"""Passive negative-data logger: python -m handmouse.negative_logger SUBJECT_ID [--lighting L] [--hand H]

Run it while you work; Ctrl-C stops it. Needs the camera, so don't run it alongside preview.
Writes raw landmarks only (frozen contract 1), never video (contract 3).
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .tracker import HandTracker

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
FLUSH_FRAMES = 900  # ~30 s at 30 FPS


def save_clip(out_dir: Path, subject_id: str, label: str, lighting: str, handedness: str,
              timestamps: list[float], landmarks: list[np.ndarray]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now():%Y%m%dT%H%M%S%f}.npz"
    np.savez_compressed(
        path,
        subject_id=subject_id, label=label, lighting=lighting, handedness=handedness,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        landmarks=np.stack(landmarks).astype(np.float32),  # (T, 21, 3)
    )
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject_id")
    ap.add_argument("--lighting", default="unknown")
    ap.add_argument("--hand", default="unknown")
    args = ap.parse_args()

    out_dir = DATA_DIR / args.subject_id / "negative"
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    tracker = HandTracker()
    t0, ts, lms = time.monotonic(), [], []

    def flush() -> None:
        nonlocal ts, lms
        if lms:
            print("saved", save_clip(out_dir, args.subject_id, "negative", args.lighting,
                                     args.hand, ts, lms))
        ts, lms = [], []

    print("logging negatives, Ctrl-C to stop")
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            lm = tracker.process(cv2.flip(frame, 1))
            if lm is not None:
                ts.append(time.monotonic() - t0)
                lms.append(lm)
                if len(lms) >= FLUSH_FRAMES:
                    flush()
    except KeyboardInterrupt:
        pass
    finally:
        flush()
        tracker.close()
        cap.release()


if __name__ == "__main__":
    main()
