import os
import shutil
import tempfile
import time
import pandas as pd
import torch
import torch.nn.functional as F
import uvicorn
import subprocess
import threading
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pose_format import Pose
from torchvision.transforms import Compose
from dataset.data_preprocessing import PoseJoinSelect, TemporalInterpolatePose, PoseNormalize
from models.vsl_net import VSLContrastiveNet

VIDEO_TO_POSE_SCRIPT = shutil.which('video_to_pose')
MAX_UPLOAD_BYTES = 5_000_000
MAX_CONCURRENT_INFERENCE = max(1, int(os.getenv('VSLNET_MAX_CONCURRENT_INFERENCE', '1')))


def run_frontend():
    try:
        subprocess.run("npm run dev", shell=True, cwd="./demo")
    except Exception as e:
        print(f"Lỗi khởi động Frontend: {e}")


def create_app(args):
    # frontend_thread = threading.Thread(target=run_frontend, daemon=True)
    # frontend_thread.start()
    # time.sleep(2)
    # webbrowser.open("http://localhost:5173")

    app = FastAPI(title="VSLNet Real-time API")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    device = torch.device("cuda" if torch.cuda.is_available() and args.GPU else "cpu")

    model = VSLContrastiveNet(
        vocab_size=args.VOCAB_SIZE,
        embedding_size=args.EMBEDDING_SIZE
    ).to(device)

    checkpoint_path = os.path.join(args.CHECKPOINT, "best_v1.pth")
    assert os.path.exists(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print("[+] Load weights successfully!")
    model.eval()

    assert os.path.exists(os.path.join(args.DATA_PATH, 'gloss.csv'))
    gloss_df = pd.read_csv(os.path.join(args.DATA_PATH, 'gloss.csv'))
    vocab_dict = dict(zip(gloss_df.index, gloss_df.iloc[:, 1]))
    print("[+] Load vocabulary successfully!")

    with torch.inference_mode():
        labels_idx = torch.arange(len(vocab_dict)).to(device)
        text_embeddings = model.text_encoder(labels_idx)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    inference_transforms = Compose([
        PoseJoinSelect(),
        TemporalInterpolatePose(frames=args.FRAMES),
        PoseNormalize()
    ])
    inference_slots = threading.BoundedSemaphore(MAX_CONCURRENT_INFERENCE)

    def process_video(video_bytes: bytes, started: float, upload_read_ms: float):
        timings = {'upload_read_ms': upload_read_ms}
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_video_path = os.path.join(tmpdir, 'raw_video.webm')
            video_tmp_path = os.path.join(tmpdir, 'input_video.mp4')
            pose_tmp_path = os.path.join(tmpdir, 'output.pose')
            with open(raw_video_path, 'wb') as stream:
                stream.write(video_bytes)

            step = time.perf_counter()
            try:
                subprocess.run(
                    ['ffmpeg', '-y', '-i', raw_video_path, '-filter:v', 'fps=30',
                     '-c:v', 'libx264', '-preset', 'ultrafast', video_tmp_path],
                    check=True, timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except subprocess.TimeoutExpired:
                return JSONResponse(status_code=504, content={'status': 'error', 'message': 'FFmpeg timeout'})
            except (OSError, subprocess.CalledProcessError):
                return JSONResponse(status_code=422, content={'status': 'error', 'message': 'Invalid video or FFmpeg unavailable'})
            timings['ffmpeg_ms'] = round((time.perf_counter() - step) * 1000, 2)

            if not VIDEO_TO_POSE_SCRIPT:
                return JSONResponse(status_code=503, content={'status': 'error', 'message': 'video_to_pose script not found'})
            step = time.perf_counter()
            try:
                subprocess.run(
                    [VIDEO_TO_POSE_SCRIPT, '-i', video_tmp_path, '-o', pose_tmp_path,
                     '--format', 'mediapipe'],
                    check=True, timeout=45, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except subprocess.TimeoutExpired:
                return JSONResponse(status_code=504, content={'status': 'error', 'message': 'Pose extraction timeout'})
            except (OSError, subprocess.CalledProcessError):
                return JSONResponse(status_code=502, content={'status': 'error', 'message': 'Pose extraction failed'})
            timings['pose_extraction_ms'] = round((time.perf_counter() - step) * 1000, 2)

            if not os.path.exists(pose_tmp_path):
                total = round((time.perf_counter() - started) * 1000, 2)
                timings['python_total_ms'] = total
                return {'status': 'success', 'action': '...', 'confidence': 0.0,
                        'processing_ms': total, 'timings': timings,
                        'message': 'Video quá ngắn, bỏ qua.'}
            with open(pose_tmp_path, 'rb') as stream:
                pose_obj = Pose.read(stream.read())

        step = time.perf_counter()
        pose_data = inference_transforms(pose_obj)
        pose_data = pose_data.reshape(pose_data.size(0), -1)
        pose_data = pose_data.unsqueeze(0).to(device)
        timings['preprocessing_ms'] = round((time.perf_counter() - step) * 1000, 2)

        step = time.perf_counter()
        with torch.inference_mode():
            video_embedding = model.video_encoder(pose_data)
            video_embedding = F.normalize(video_embedding, p=2, dim=-1)
            logits = model.logit_scale.exp() * (video_embedding @ text_embeddings.T)
            probs = F.softmax(logits, dim=-1)
            conf, pred_idx = torch.max(probs, dim=1)
        timings['model_inference_ms'] = round((time.perf_counter() - step) * 1000, 2)
        total = round((time.perf_counter() - started) * 1000, 2)
        timings['python_total_ms'] = total
        return {'status': 'success', 'action': vocab_dict.get(pred_idx.item(), 'Unknown'),
                'confidence': float(conf.item()), 'processing_ms': total,
                'timings': timings, 'message': 'Predict successfully!'}

    @app.post("/predict")
    async def predict(req: Request):
        started = time.perf_counter()
        try:
            form_data = await req.form()
            file = form_data.get("file", None)
            if file is None or not hasattr(file, 'read'):
                return JSONResponse(status_code=400, content={"status": "error", "message": "No receive file video!"})
            video_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
            await file.close()
            if not video_bytes or len(video_bytes) > MAX_UPLOAD_BYTES:
                return JSONResponse(status_code=413, content={'status': 'error', 'message': 'Video must be 1-5000000 bytes'})
            upload_read_ms = round((time.perf_counter() - started) * 1000, 2)
            if not inference_slots.acquire(blocking=False):
                return JSONResponse(status_code=429, content={'status': 'error', 'message': 'Inference capacity reached'})
            try:
                return await run_in_threadpool(process_video, video_bytes, started, upload_read_ms)
            finally:
                inference_slots.release()
        except Exception as e:
            return JSONResponse(status_code=500, content={"status": "error", "message": "Inference failed"})

    return app


def inference(args):
    uvicorn.run(create_app(args), host="127.0.0.1", port=int(os.getenv("VSLNET_PORT", "8001")))
