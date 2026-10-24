"""Webcam -> MediaPipe Hand Landmarker (live-stream mode) -> 21 landmarks per frame."""
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"

# Pairs of landmark indices forming the hand skeleton, for drawing.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
    (18, 19), (19, 20), (0, 17),
]


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class HandTracker:
    """Returns (21, 3) normalised landmarks (x, y in [0,1], z relative) or None."""

    def __init__(self, max_hands: int = 1):
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._t0 = time.monotonic()

    def process(self, bgr: np.ndarray) -> np.ndarray | None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.monotonic() - self._t0) * 1000)
        result = self._lm.detect_for_video(image, ts_ms)
        if not result.hand_landmarks:
            return None
        return np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)

    def close(self) -> None:
        self._lm.close()


def draw(frame: np.ndarray, lm: np.ndarray) -> None:
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in lm]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)
