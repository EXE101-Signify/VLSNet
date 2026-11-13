"""Experimental VSL binary WebSocket. Never emits a model label without inference."""
import asyncio
import json
import os
import struct
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from mode.streaming_auth import SessionTokenError, verify_session_token
from mode.streaming_pose import StreamingPoseExtractor, TemporalPoseBuffer


FRAME_HEADER = struct.Struct('!BIQI')  # version, sequence, capture timestamp ms, JPEG length
MAX_JPEG_BYTES = 200_000
MAX_SESSIONS = 2
MIN_FRAME_INTERVAL_MS = 66  # 15 FPS


def parse_binary_frame(message: bytes):
    if len(message) < FRAME_HEADER.size:
        raise ValueError('Truncated VSL frame')
    version, sequence, captured_at, size = FRAME_HEADER.unpack_from(message)
    jpeg = message[FRAME_HEADER.size:]
    if (version != 1 or sequence <= 0 or captured_at <= 0 or size != len(jpeg)
            or size < 100 or size > MAX_JPEG_BYTES or not jpeg.startswith(b'\xff\xd8')):
        raise ValueError('Invalid VSL frame')
    return sequence, captured_at, jpeg


class BackendBridge:
    def __init__(self, base_url: str, secret: str):
        self.base_url = base_url.rstrip('/')
        self.secret = secret

    def post(self, path: str, payload: dict):
        request = urllib.request.Request(self.base_url + path, data=json.dumps(payload).encode(),
                headers={'Content-Type': 'application/json', 'X-VSL-Service-Token': self.secret}, method='POST')
        with urllib.request.urlopen(request, timeout=3) as response:
            content = response.read(4096)
            return json.loads(content) if content else None

    def validate(self, claims):
        result = self.post('/api/internal/vsl/session/validate', {
            'sessionId': claims.session_id, 'callId': claims.call_id, 'signerId': claims.signer_id})
        if not result or result.get('active') is not True:
            raise ValueError('Streaming session revoked')

    def prediction(self, claims, sequence: int, status: str, action: str = '',
                   confidence: float = 0, processing_ms: float = 0, captured_at: int = 0):
        self.post('/api/internal/vsl/predictions', {
            'eventId': str(uuid.uuid4()), 'sessionId': claims.session_id,
            'callId': claims.call_id, 'signerId': claims.signer_id,
            'sequenceNumber': sequence, 'status': status, 'action': action,
            'confidence': confidence, 'processingMs': processing_ms,
            'timestamp': captured_at or int(time.time() * 1000)})


def install_streaming_route(app, model, text_embeddings, vocab_dict, device):
    active_sessions = set()
    active_lock = asyncio.Lock()
    model_slot = asyncio.Semaphore(1)
    secret = os.getenv('SIGNIFY_VSL_STREAMING_SECRET', '')
    allowed_origins = {part.strip() for part in os.getenv('VSLNET_ALLOWED_ORIGINS', 'http://localhost:3000').split(',')}
    bridge = BackendBridge(os.getenv('VSLNET_BACKEND_URL', 'http://127.0.0.1:8080'), secret)

    def predict_tensor(tensor):
        started = time.perf_counter()
        with torch.inference_mode():
            video = F.normalize(model.video_encoder(tensor.unsqueeze(0).to(device)), p=2, dim=-1)
            probs = F.softmax(model.logit_scale.exp() * (video @ text_embeddings.T), dim=-1)
            confidence, index = torch.max(probs, dim=1)
        return vocab_dict[index.item()], float(confidence.item()), (time.perf_counter() - started) * 1000

    @app.websocket('/ws/vsl/stream')
    async def stream(websocket: WebSocket):
        if len(secret.encode()) < 32 or websocket.headers.get('origin') not in allowed_origins:
            await websocket.close(code=4403)
            return
        await websocket.accept()
        try:
            control = json.loads(await asyncio.wait_for(websocket.receive_text(), timeout=5))
            if control.get('type') != 'AUTH':
                raise SessionTokenError('Authentication required')
            claims = verify_session_token(control.get('token', ''), secret)
            await asyncio.to_thread(bridge.validate, claims)
        except (ValueError, KeyError, TimeoutError, OSError, WebSocketDisconnect):
            await websocket.close(code=4403)
            return
        async with active_lock:
            if claims.session_id in active_sessions or len(active_sessions) >= MAX_SESSIONS:
                await websocket.close(code=4429)
                return
            active_sessions.add(claims.session_id)

        queue = asyncio.Queue(maxsize=2)
        send_lock = asyncio.Lock()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='vsl-pose')
        loop = asyncio.get_running_loop()
        tracker = None
        received = 0
        dropped = 0
        last_seq = 0
        last_timestamp = 0
        last_receive_time = 0.0

        async def send_control(payload):
            async with send_lock:
                await websocket.send_json(payload)

        async def emit(sequence, status, action='', confidence=0.0, processing_ms=0.0, captured_at=0):
            await asyncio.to_thread(bridge.prediction, claims, sequence, status, action,
                                    confidence, processing_ms, captured_at)

        async def process_frames():
            nonlocal tracker
            tracker = await loop.run_in_executor(executor, lambda: StreamingPoseExtractor(480, 360, 15))
            buffer = TemporalPoseBuffer()
            candidate = None
            candidate_count = 0
            emitted = None
            last_infer_at = 0
            last_no_sign_at = 0
            while True:
                sequence, captured_at, jpeg = await queue.get()
                pose = await loop.run_in_executor(executor, tracker.extract_jpeg, jpeg, sequence, captured_at)
                if not buffer.append(pose):
                    continue
                window = buffer.window(duration_ms=2500, min_frames=16)
                if not window or captured_at - last_infer_at < 500:
                    continue
                last_infer_at = captured_at
                if sum(frame.hands_present for frame in window) < 8:
                    candidate, candidate_count, emitted = None, 0, None
                    if captured_at - last_no_sign_at >= 1000:
                        last_no_sign_at = captured_at
                        await emit(sequence, 'NO_SIGN', captured_at=captured_at)
                    continue
                tensor = await loop.run_in_executor(executor, tracker.to_tensor, window)
                async with model_slot:
                    action, confidence, inference_ms = await asyncio.to_thread(predict_tensor, tensor)
                if confidence < 0.8:
                    candidate, candidate_count, emitted = None, 0, None
                    if captured_at - last_no_sign_at >= 1000:
                        last_no_sign_at = captured_at
                        await emit(sequence, 'NO_SIGN', confidence=confidence, captured_at=captured_at)
                    continue
                if action != candidate:
                    candidate, candidate_count = action, 1
                else:
                    candidate_count += 1
                if action == emitted:
                    continue
                await emit(sequence, 'PARTIAL', action, confidence, inference_ms, captured_at)
                if candidate_count >= 3:
                    await emit(sequence, 'FINAL', action, confidence, inference_ms, captured_at)
                    emitted = action

        processor = asyncio.create_task(process_frames())
        try:
            await send_control({'type': 'READY', 'sessionId': claims.session_id, 'maxFps': 15})
            last_validation = time.monotonic()
            while True:
                if time.time() >= claims.expires_at or processor.done():
                    break
                if time.monotonic() - last_validation >= 5:
                    await asyncio.to_thread(bridge.validate, claims)
                    last_validation = time.monotonic()
                message = await asyncio.wait_for(websocket.receive(), timeout=10)
                if message['type'] == 'websocket.disconnect':
                    break
                if message.get('text') is not None:
                    if message['text'] == 'PING':
                        await send_control({'type': 'PONG'})
                    continue
                binary = message.get('bytes')
                if binary is None:
                    continue
                sequence, captured_at, jpeg = parse_binary_frame(binary)
                now = time.monotonic()
                if sequence <= last_seq or captured_at <= last_timestamp:
                    continue
                if last_seq and sequence > last_seq + 1:
                    dropped += sequence - last_seq - 1
                    await send_control({'type': 'GAP', 'sequenceNumber': last_seq + 1, 'droppedFrames': dropped})
                    await emit(last_seq + 1, 'GAP', captured_at=captured_at)
                if now - last_receive_time < MIN_FRAME_INTERVAL_MS / 1000:
                    dropped += 1
                    last_seq, last_timestamp = sequence, captured_at
                    await send_control({'type': 'GAP', 'sequenceNumber': sequence, 'droppedFrames': dropped})
                    await emit(sequence, 'GAP', captured_at=captured_at)
                    continue
                received += 1
                last_seq, last_timestamp, last_receive_time = sequence, captured_at, now
                if queue.full():
                    old_seq, old_time, _ = queue.get_nowait()
                    dropped += 1
                    await send_control({'type': 'GAP', 'sequenceNumber': old_seq, 'droppedFrames': dropped})
                    await emit(old_seq, 'GAP', captured_at=old_time)
                queue.put_nowait((sequence, captured_at, jpeg))
        except (ValueError, TimeoutError, WebSocketDisconnect, OSError):
            pass
        finally:
            processor.cancel()
            processor_failed = False
            try:
                await processor
            except asyncio.CancelledError:
                pass
            except Exception:
                processor_failed = True
            if processor_failed:
                try:
                    await send_control({'type': 'ERROR'})
                    if last_seq:
                        await emit(last_seq, 'ERROR')
                except (OSError, RuntimeError):
                    pass
            try:
                if tracker is not None:
                    await loop.run_in_executor(executor, tracker.close)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
                async with active_lock:
                    active_sessions.discard(claims.session_id)
            try:
                await websocket.close()
            except RuntimeError:
                pass
