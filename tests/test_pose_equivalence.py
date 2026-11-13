import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from pose_format import Pose
from pose_format.bin.pose_estimation import pose_video
from torchvision.transforms import Compose

from dataset.data_preprocessing import PoseJoinSelect, TemporalInterpolatePose, PoseNormalize
from models.vsl_net import VSLContrastiveNet


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('video_to_pose'), 'FFmpeg and video_to_pose required')
class PoseEquivalenceTests(unittest.TestCase):
    def test_cli_and_inprocess_produce_identical_tensor_and_logits_on_synthetic_video(self):
        transform = Compose([PoseJoinSelect(), TemporalInterpolatePose(frames=64), PoseNormalize()])
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp, 'test.webm')
            mp4 = Path(tmp, 'test.mp4')
            cli_pose = Path(tmp, 'cli.pose')
            inprocess_pose = Path(tmp, 'inprocess.pose')
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                            'testsrc=size=320x240:rate=30', '-t', '2.5', '-an', '-c:v', 'libvpx', str(clip)], check=True)
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(clip),
                            '-filter:v', 'fps=30', '-c:v', 'libx264', '-preset', 'ultrafast', str(mp4)], check=True)
            subprocess.run(['video_to_pose', '-i', str(mp4), '-o', str(cli_pose), '--format', 'mediapipe'],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pose_video(str(mp4), str(inprocess_pose), 'mediapipe', {}, progress=False, pose_workers=1)
            cli_tensor = transform(Pose.read(cli_pose.read_bytes())).reshape(64, 108)
            inprocess_tensor = transform(Pose.read(inprocess_pose.read_bytes())).reshape(64, 108)

        self.assertEqual(tuple(cli_tensor.shape), (64, 108))
        self.assertTrue(torch.equal(cli_tensor, inprocess_tensor))
        model = VSLContrastiveNet(vocab_size=400, embedding_size=256)
        model.load_state_dict(torch.load('checkpoint/best_v1.pth', map_location='cpu', weights_only=True), strict=True)
        model.eval()
        labels = torch.arange(400)
        with torch.inference_mode():
            text = F.normalize(model.text_encoder(labels), p=2, dim=-1)
            def logits(tensor):
                video = F.normalize(model.video_encoder(tensor.unsqueeze(0)), p=2, dim=-1)
                return model.logit_scale.exp() * (video @ text.T)
            original = logits(cli_tensor)
            experimental = logits(inprocess_tensor)
        self.assertTrue(torch.equal(original, experimental))
        self.assertEqual(original.argmax().item(), experimental.argmax().item())


if __name__ == '__main__':
    unittest.main()
