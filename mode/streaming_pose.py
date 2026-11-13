"""Experimental in-memory MediaPipe pose path for VSLNet.

The frame-to-Pose conversion uses pose_format's own holistic component layout
and extraction routine. The model transforms remain unchanged.
"""
from collections import deque
from dataclasses import dataclass
from time import perf_counter

import cv2
import mediapipe as mp
import numpy as np
import torch
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_header import PoseHeader, PoseHeaderDimensions
from pose_format.utils.holistic import _extract_frame_data, holistic_components
from torchvision.transforms import Compose

from dataset.data_preprocessing import PoseJoinSelect, PoseNormalize, TemporalInterpolatePose


@dataclass(frozen=True)
class PoseFrame:
    sequence: int
    timestamp_ms: int
    data: np.ndarray
    confidence: np.ndarray
    hands_present: bool
    pose_present: bool
    pose_ms: float


class StreamingPoseExtractor:
    def __init__(self, width: int, height: int, fps: float = 15.0):
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError('Invalid frame dimensions or FPS')
        self.width, self.height, self.fps = width, height, fps
        self.tracker = mp.solutions.holistic.Holistic(static_image_mode=False)
        self.header = PoseHeader(version=0.2,
                                 dimensions=PoseHeaderDimensions(width=width, height=height, depth=0),
                                 components=holistic_components('XYZC', 0))
        self.transforms = Compose([PoseJoinSelect(), TemporalInterpolatePose(frames=64), PoseNormalize()])

    def extract_rgb(self, rgb: np.ndarray, sequence: int, timestamp_ms: int) -> PoseFrame:
        if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
            raise ValueError('Unexpected RGB frame format')
        started = perf_counter()
        result = self.tracker.process(rgb)
        data, confidence = _extract_frame_data(result, self.width, self.height, 0)
        return PoseFrame(sequence, timestamp_ms, data, confidence,
                         result.left_hand_landmarks is not None or result.right_hand_landmarks is not None,
                         result.pose_landmarks is not None, (perf_counter() - started) * 1000)

    def extract_jpeg(self, jpeg: bytes, sequence: int, timestamp_ms: int) -> PoseFrame:
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (self.height, self.width):
            raise ValueError('Invalid JPEG frame or dimensions')
        return self.extract_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), sequence, timestamp_ms)

    def to_tensor(self, frames: list[PoseFrame]) -> torch.Tensor:
        if len(frames) < 2 or any(b.timestamp_ms <= a.timestamp_ms for a, b in zip(frames, frames[1:])):
            raise ValueError('At least two ordered, distinct frames are required')
        pose = Pose(self.header, NumPyPoseBody(fps=self.fps,
                    data=np.expand_dims(np.stack([frame.data for frame in frames]), axis=1),
                    confidence=np.expand_dims(np.stack([frame.confidence for frame in frames]), axis=1)))
        return self.transforms(pose).reshape(64, 108)

    def close(self):
        self.tracker.close()


class TemporalPoseBuffer:
    def __init__(self, max_frames: int = 96, max_age_ms: int = 5000):
        self.frames: deque[PoseFrame] = deque(maxlen=max_frames)
        self.max_age_ms = max_age_ms

    def append(self, frame: PoseFrame) -> bool:
        if self.frames and (frame.sequence <= self.frames[-1].sequence
                            or frame.timestamp_ms <= self.frames[-1].timestamp_ms):
            return False
        self.frames.append(frame)
        while self.frames and frame.timestamp_ms - self.frames[0].timestamp_ms > self.max_age_ms:
            self.frames.popleft()
        return True

    def window(self, duration_ms: int = 2500, min_frames: int = 16) -> list[PoseFrame]:
        if not self.frames:
            return []
        end = self.frames[-1].timestamp_ms
        selected = [frame for frame in self.frames if frame.timestamp_ms >= end - duration_ms]
        if len(selected) < min_frames or selected[-1].timestamp_ms - selected[0].timestamp_ms < duration_ms * .7:
            return []
        return selected

    def clear(self):
        self.frames.clear()
