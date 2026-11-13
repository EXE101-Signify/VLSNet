import asyncio
import json
import os
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from mode import inferencer


async def post_video(app, content=b'webm'):
    boundary = b'vsl-test-boundary'
    body = (b'--' + boundary + b'\r\nContent-Disposition: form-data; name="file"; filename="clip.webm"\r\n'
            b'Content-Type: video/webm\r\n\r\n' + content + b'\r\n--' + boundary + b'--\r\n')
    scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': 'POST',
             'scheme': 'http', 'path': '/predict', 'raw_path': b'/predict', 'query_string': b'',
             'headers': [(b'content-type', b'multipart/form-data; boundary=' + boundary),
                         (b'content-length', str(len(body)).encode())],
             'client': ('test', 1), 'server': ('test', 80)}
    messages = []
    sent = False
    completed = asyncio.Event()

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {'type': 'http.request', 'body': body, 'more_body': False}
        await completed.wait()
        return {'type': 'http.disconnect'}

    async def send(message):
        messages.append(message)
        if message['type'] == 'http.response.body' and not message.get('more_body', False):
            completed.set()

    await app(scope, receive, send)
    status = next(message['status'] for message in messages if message['type'] == 'http.response.start')
    response = b''.join(message.get('body', b'') for message in messages if message['type'] == 'http.response.body')
    return status, json.loads(response)


class InferenceApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = yaml.safe_load(Path('config/vsl_net_configuration.yaml').read_text())['BASIC']
        with patch.dict(os.environ, {'VSLNET_POSE_MODE': 'cli'}), \
                patch.object(inferencer.torch, 'load', wraps=torch.load) as loader:
            cls.app = inferencer.create_app(SimpleNamespace(**settings))
            cls.checkpoint_loads = loader.call_count

    def test_checkpoint_loaded_once_and_invalid_video_rejected(self):
        self.assertEqual(self.checkpoint_loads, 1)
        def fail_ffmpeg(*args, **kwargs):
            raise inferencer.subprocess.CalledProcessError(1, args[0])
        with patch.object(inferencer.subprocess, 'run', side_effect=fail_ffmpeg):
            status, response = asyncio.run(asyncio.wait_for(post_video(self.app), 5))
        self.assertEqual(status, 422)
        self.assertEqual(response['status'], 'error')
        self.assertEqual(self.checkpoint_loads, 1)

    def test_upload_limit_and_ffmpeg_timeout(self):
        status, _ = asyncio.run(post_video(self.app, b'x' * (inferencer.MAX_UPLOAD_BYTES + 1)))
        self.assertEqual(status, 413)
        def timeout(*args, **kwargs):
            raise inferencer.subprocess.TimeoutExpired(args[0], 15)
        with patch.object(inferencer.subprocess, 'run', side_effect=timeout):
            status, response = asyncio.run(post_video(self.app))
        self.assertEqual(status, 504)
        self.assertEqual(response['message'], 'FFmpeg timeout')

    def test_no_pose_result_and_temporary_directory_cleanup(self):
        paths = []
        def fake_run(args, **kwargs):
            paths.append(Path(args[-1] if args[0] == 'ffmpeg' else args[args.index('-o') + 1]))
            if args[0] == 'ffmpeg':
                paths[-1].write_bytes(b'mp4')
        with patch.object(inferencer, 'VIDEO_TO_POSE_SCRIPT', 'video_to_pose'), \
                patch.object(inferencer.subprocess, 'run', side_effect=fake_run):
            status, response = asyncio.run(post_video(self.app))
        self.assertEqual(status, 200)
        self.assertEqual(response['action'], '...')
        self.assertIn('pose_extraction_ms', response['timings'])
        self.assertTrue(all(not path.exists() for path in paths))

    def test_second_request_rejected_while_slot_is_busy(self):
        started = threading.Event()
        release = threading.Event()
        def slow_run(args, **kwargs):
            if args[0] == 'ffmpeg':
                started.set()
                release.wait(timeout=2)
                Path(args[-1]).write_bytes(b'mp4')
        async def concurrent():
            first = asyncio.create_task(post_video(self.app))
            await asyncio.to_thread(started.wait, 2)
            second = await post_video(self.app)
            release.set()
            return await first, second
        try:
            with patch.object(inferencer, 'VIDEO_TO_POSE_SCRIPT', 'video_to_pose'), \
                    patch.object(inferencer.subprocess, 'run', side_effect=slow_run):
                first, second = asyncio.run(concurrent())
            self.assertEqual(first[0], 200)
            self.assertEqual(second[0], 429)
        finally:
            release.set()


if __name__ == '__main__':
    unittest.main()
