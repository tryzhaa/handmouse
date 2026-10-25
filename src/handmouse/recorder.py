"""Prompted recorder: python -m handmouse.recorder SUBJECT_ID --lighting daylight --hand right

Shows "Next: <gesture>", counts down, records landmarks, labels the clip automatically.
Motion gestures show an arrow and record only the sweep (a rest beat after it is not recorded),
so the return stroke does not end up in the clip. Landmarks only; no video (frozen contract 3).
Re-record just the motion gestures with --motion-only. q quits.
"""
import argparse
import random
import time

import cv2

from .negative_logger import DATA_DIR, save_clip
from .tracker import HandTracker, draw

# Labels follow docs/plan.md gesture table; swipe left/right are separate classes.
GESTURES = ["point", "palm", "pinch", "pinch_hold", "two_fingers",
            "swipe_left", "swipe_right", "swipe_up", "draw_x"]
DYNAMIC = {"swipe_left", "swipe_right", "swipe_up", "draw_x"}
# Arrow direction on the (mirrored) screen: what you see is your own left/right.
ARROW = {"swipe_left": (-1, 0), "swipe_right": (1, 0), "swipe_up": (0, -1)}
HINT = {
    "swipe_left": "ONE fast sweep LEFT, then stop",
    "swipe_right": "ONE fast sweep RIGHT, then stop",
    "swipe_up": "ONE fast sweep UP, then stop",
    "draw_x": "Draw ONE big X, then stop",
}
COUNTDOWN_S = 3.0
STATIC_S = 1.5
DYNAMIC_S = {"swipe_left": 1.2, "swipe_right": 1.2, "swipe_up": 1.2, "draw_x": 2.0}
REST_S = 1.0  # after a motion clip; not recorded, lets the hand come back
MIN_FRAMES = 10
CYAN = (255, 230, 0)


def build_schedule(gestures: list[str], repeats: int, seed: int = 0) -> list[str]:
    s = [g for g in gestures for _ in range(repeats)]
    random.Random(seed).shuffle(s)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject_id")
    ap.add_argument("--lighting", required=True)
    ap.add_argument("--hand", required=True, choices=["left", "right"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--motion-only", action="store_true",
                    help="record only swipes and draw_x")
    args = ap.parse_args()

    gestures = [g for g in GESTURES if g in DYNAMIC] if args.motion_only else GESTURES
    queue = build_schedule(gestures, args.repeats)
    total = len(queue)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    tracker = HandTracker()

    def show(frame, text, lm=None, label=None, sub=None) -> bool:
        if lm is not None:
            draw(frame, lm)
        h, w = frame.shape[:2]
        if label in ARROW:
            dx, dy = ARROW[label]
            cv2.arrowedLine(frame, (w // 2 - dx * 110, h // 2 - dy * 110),
                            (w // 2 + dx * 110, h // 2 + dy * 110), CYAN, 8, cv2.LINE_AA,
                            tipLength=0.35)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        if sub:
            cv2.putText(frame, sub, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CYAN, 2)
        cv2.imshow("handmouse recorder", frame)
        return cv2.waitKey(1) & 0xFF != ord("q")

    def phase(seconds, text, label=None, sub=None, record=False):
        """Run one timed phase; text may use {left} for seconds remaining."""
        ts, lms, t0 = [], [], time.monotonic()
        while (elapsed := time.monotonic() - t0) < seconds:
            ok, frame = cap.read()
            if not ok:
                return False, ts, lms
            frame = cv2.flip(frame, 1)
            lm = tracker.process(frame)  # always run, keeps tracker timestamps advancing
            if record and lm is not None:
                ts.append(elapsed)
                lms.append(lm)
            if not show(frame, text.format(left=seconds - elapsed), lm, label, sub):
                return False, ts, lms
        return True, ts, lms

    try:
        while queue:
            label = queue.pop(0)
            done = total - len(queue) - 1
            sub = HINT.get(label)
            ok, _, _ = phase(COUNTDOWN_S, f"[{done}/{total}] {label} in {{left:.0f}}", label, sub)
            if not ok:
                return
            dur = DYNAMIC_S[label] if label in DYNAMIC else STATIC_S
            ok, ts, lms = phase(dur, f"GO: {label}", label, sub, record=True)
            if not ok:
                return
            if label in DYNAMIC:
                ok, _, _ = phase(REST_S, "REST: hand back to the middle")
                if not ok:
                    return
            if len(lms) < MIN_FRAMES:
                print(f"hand lost during {label}; will redo")
                queue.append(label)
                continue
            save_clip(DATA_DIR / args.subject_id / label, args.subject_id, label,
                      args.lighting, args.hand, ts, lms)
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
