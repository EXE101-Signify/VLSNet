import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from pose_format.bin.pose_estimation import read_frames_exact
from pose_format.utils.holistic import load_holistic
from torchvision.transforms import Compose

from dataset.data_preprocessing import PoseJoinSelect, TemporalInterpolatePose, PoseNormalize
from mode.streaming_pose import PoseFrame, StreamingPoseExtractor, TemporalPoseBuffer


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
class StreamingPoseRegressionTests(unittest.TestCase):
    def test_same_rgb_frames_match_original_pose_transforms(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp, 'same_frames.mp4')
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                            'testsrc=size=320x240:rate=30', '-t', '2.5', '-c:v', 'libx264', str(video)], check=True)
            rgb_frames = list(read_frames_exact(str(video)))
        original_pose = load_holistic(rgb_frames, fps=30, width=320, height=240,
                                      progress=False, pose_workers=1, reuse=False)
        transform = Compose([PoseJoinSelect(), TemporalInterpolatePose(frames=64), PoseNormalize()])
        original = transform(original_pose).reshape(64, 108)
        extractor = StreamingPoseExtractor(width=320, height=240, fps=30)
        try:
            streamed = [extractor.extract_rgb(rgb, i, i * 33 + 1) for i, rgb in enumerate(rgb_frames)]
            result = extractor.to_tensor(streamed)
        finally:
            extractor.close()
        self.assertEqual(result.shape, torch.Size([64, 108]))
        self.assertTrue(torch.equal(result, original))

    def test_buffer_rejects_old_frames_and_requires_real_temporal_span(self):
        buffer = TemporalPoseBuffer(max_frames=20, max_age_ms=3000)
        frame = lambda i, ms: PoseFrame(i, ms, np.zeros((576, 3)), np.zeros(576), False, False, 1.0)
        self.assertTrue(buffer.append(frame(1, 1000)))
        self.assertFalse(buffer.append(frame(1, 1100)))
        self.assertFalse(buffer.append(frame(2, 900)))
        self.assertEqual(buffer.window(duration_ms=2500, min_frames=16), [])
        for i in range(2, 19):
            self.assertTrue(buffer.append(frame(i, 1000 + i * 150)))
        self.assertTrue(buffer.append(frame(19, 3850)))
        self.assertGreaterEqual(len(buffer.window(duration_ms=2500, min_frames=16)), 16)
        self.assertLessEqual(len(buffer.frames), 20)
        buffer.clear()
        self.assertEqual(len(buffer.frames), 0)


if __name__ == '__main__':
    unittest.main()
