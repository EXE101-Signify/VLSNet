"""Measure transport and Python timings for a running local VSLNet service.

Synthetic clips only measure technical performance, never recognition accuracy.
"""
import argparse
import json
import math
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def sample(url, video, filename):
    boundary = 'vsl-benchmark-boundary'
    mime = 'video/mp4' if filename.lower().endswith('.mp4') else 'video/webm'
    payload = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
               f'Content-Type: {mime}\r\n\r\n').encode() + video + f'\r\n--{boundary}--\r\n'.encode()
    request = urllib.request.Request(url, data=payload, method='POST',
                                     headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=90) as response:
        result = json.load(response)
    wall_ms = (time.perf_counter() - started) * 1000
    return {'wall_ms': round(wall_ms, 2), 'python_total_ms': result.get('processing_ms'),
            'timings': result.get('timings'), 'action': result.get('action'),
            'confidence': result.get('confidence')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8001/predict')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', type=Path)
    source.add_argument('--synthetic', action='store_true')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--runs', type=int, default=4)
    args = parser.parse_args()
    if args.runs < 2 or args.warmup < 0:
        parser.error('--runs must be at least 2 and --warmup nonnegative')
    with tempfile.TemporaryDirectory() as tmp:
        video_path = args.video
        if args.synthetic:
            if not shutil.which('ffmpeg'):
                parser.error('FFmpeg is required for --synthetic')
            video_path = Path(tmp, 'synthetic.webm')
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                            'testsrc=size=320x240:rate=30', '-t', '2.5', '-an', '-c:v', 'libvpx', str(video_path)],
                           check=True)
        video = video_path.read_bytes()
        for _ in range(args.warmup):
            sample(args.url, video, video_path.name)
        results = [sample(args.url, video, video_path.name) for _ in range(args.runs)]
    wall = [result['wall_ms'] for result in results]
    python = [result['python_total_ms'] for result in results]
    print(json.dumps({
        'source': 'synthetic' if args.synthetic else str(args.video), 'runs': args.runs,
        'wall_ms': {'p50': percentile(wall, .5), 'p95': percentile(wall, .95),
                    'min': min(wall), 'max': max(wall)},
        'python_total_ms': {'p50': percentile(python, .5), 'p95': percentile(python, .95),
                            'min': min(python), 'max': max(python)},
        'serial_throughput_clips_per_s': round(1000 * len(wall) / sum(wall), 3),
        'samples': results,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
