FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    VSLNET_HOST=0.0.0.0 \
    VSLNET_PORT=8001 \
    VSLNET_POSE_MODE=cli \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# FFmpeg is used by the retained /predict clip route. MediaPipe/OpenCV need
# GLib/OpenGL and sounddevice needs PortAudio on the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libgomp1 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-inference.txt /app/
RUN python -m pip install --no-cache-dir \
    'torch==2.15.0+cpu' 'torchvision==0.29.2+cpu' \
    --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements-inference.txt \
    && python -m pip check

COPY serve.py /app/
COPY config/vsl_net_configuration.yaml /app/config/
COPY dataset/__init__.py dataset/data_preprocessing.py /app/dataset/
COPY dataset/root/gloss.csv /app/dataset/root/
COPY models/__init__.py models/text_encoder.py models/video_encoder.py models/vsl_net.py /app/models/
COPY mode/__init__.py mode/inferencer.py mode/streaming_auth.py mode/streaming_pose.py mode/streaming_ws.py /app/mode/
COPY checkpoint/best_v1.pth /app/checkpoint/

RUN test -s /app/checkpoint/best_v1.pth \
    && test -s /app/dataset/root/gloss.csv \
    && command -v ffmpeg \
    && command -v video_to_pose \
    && python -m compileall -q /app/serve.py /app/mode /app/models /app/dataset

RUN useradd --system --uid 10001 --create-home appuser
USER appuser

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=8s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=5).close()" || exit 1

# One process retains one checkpoint and session registry in memory.
CMD ["python", "serve.py"]
