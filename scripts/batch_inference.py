"""Send real video clips to the local VSLNet API and record its responses."""

import argparse
import csv
import json
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


VIDEO_SUFFIXES = {".webm", ".mp4", ".avi", ".mov", ".mkv"}
FIELDS = ("video", "status", "action", "confidence", "processing_ms", "roundtrip_ms", "error")


def video_paths(inputs):
    paths = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
        elif path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            paths.append(path)
        else:
            raise ValueError(f"Not a supported video file or directory: {path}")
    return sorted(set(p.resolve() for p in paths))


def predict(path, endpoint, timeout):
    boundary = uuid.uuid4().hex
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    body = header + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = Request(endpoint, data=body, method="POST", headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"
    })
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        error = ""
    except HTTPError as exc:
        try:
            result = json.loads(exc.read())
        except (ValueError, UnicodeDecodeError):
            result = {}
        error = f"HTTP {exc.code}: {result.get('message', exc.reason)}"
    except (URLError, TimeoutError) as exc:
        result = {}
        error = str(exc)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "video": str(path),
        "status": result.get("status", "error"),
        "action": result.get("action", ""),
        "confidence": result.get("confidence", ""),
        "processing_ms": result.get("processing_ms", ""),
        "roundtrip_ms": elapsed_ms,
        "error": error or result.get("message", "") if result.get("status") != "success" else "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", help="Video files and/or directories")
    parser.add_argument("--output", default="inference_results.csv")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001/predict")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    paths = video_paths(args.videos)
    if not paths:
        parser.error("No supported video files found")
    with open(args.output, "w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        for path in paths:
            row = predict(path, args.endpoint, args.timeout)
            writer.writerow(row)
            print(f"{path.name}: {row['status']} {row['action']} ({row['processing_ms']} ms)")


if __name__ == "__main__":
    main()
