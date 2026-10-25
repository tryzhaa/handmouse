"""Latency profiler: python -m handmouse.latency [--seconds 20]

Times each stage per frame on this machine: MediaPipe tracking and spotting decision.
Measures processing only (frame in hand -> decision); camera exposure and transfer happen
before the frame reaches software and are not included. Keep a hand in view while it runs.
"""
import argparse
import json
import time
from datetime import datetime

import cv2
import numpy as np

from .evaluate import RESULTS_DIR, git_state
from .spotting import Spotter
from .tracker import HandTracker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20)
    args = ap.parse_args()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    tracker, spotter = HandTracker(), Spotter()
    track_ms, decide_ms, total_ms = [], [], []
    frames = hand_frames = 0
    t_end = time.monotonic() + args.seconds
    print(f"profiling for {args.seconds:.0f} s, keep a hand in view")
    try:
        while time.monotonic() < t_end:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            lm = tracker.process(cv2.flip(frame, 1))
            t1 = time.perf_counter()
            frames += 1
            track_ms.append((t1 - t0) * 1000)
            if lm is not None:
                spotter.push(lm, time.monotonic())
                t2 = time.perf_counter()
                hand_frames += 1
                decide_ms.append((t2 - t1) * 1000)
                total_ms.append((t2 - t0) * 1000)
    finally:
        tracker.close()
        cap.release()

    def pct(xs):
        return {"p50": round(float(np.percentile(xs, 50)), 2),
                "p95": round(float(np.percentile(xs, 95)), 2)} if xs else None

    result = {
        "kind": "latency",
        "frames": frames,
        "hand_frames": hand_frames,
        "fps": round(frames / args.seconds, 1),
        "track_ms": pct(track_ms),
        "decide_ms": pct(decide_ms),
        "processing_ms": pct(total_ms),
        "note": "processing only; excludes camera exposure/transfer",
        "git": git_state(),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"latency-{datetime.now():%Y%m%dT%H%M%S}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("fps", "track_ms", "decide_ms", "processing_ms")}))
    print("wrote", out)


if __name__ == "__main__":
    main()
