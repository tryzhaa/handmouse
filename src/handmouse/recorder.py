"""Prompted recorder: python -m handmouse.recorder SUBJECT_ID --lighting daylight --hand right

Shows the current and next gesture and records clips back to back (one get-ready at launch),
labelling each automatically.
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
    "point": "Index finger out, hold steady",
    "palm": "Open palm, hold steady",
    "two_fingers": "Index + middle up, hold steady",
    "pinch": "Tap thumb+index once, release at once",
    "pinch_hold": "Pinch and KEEP it closed",
    "swipe_left": "ONE fast sweep LEFT, then stop",
    "swipe_right": "ONE fast sweep RIGHT, then stop",
    "swipe_up": "ONE fast sweep UP, then stop",
    "draw_x": "Draw ONE big X, then stop",
}
START_S = 3.0  # once, before the first clip only
READY_S = 0.6  # shows the gesture before recording so the pose is formed when GO starts
STATIC_S = 1.2
DYNAMIC_S = {"swipe_left": 1.2, "swipe_right": 1.2, "swipe_up": 1.2, "draw_x": 2.0}
REST_S = 0.6  # after every clip; not recorded, lets the hand return to open
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
    ap.add_argument("--only", nargs="+", choices=GESTURES, help="record just these gestures")
    args = ap.parse_args()

    gestures = args.only or ([g for g in GESTURES if g in DYNAMIC] if args.motion_only
                             else GESTURES)
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
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
        if not phase(START_S, "Get ready: {left:.0f}")[0]:
            return
        while queue:
            label = queue.pop(0)
            done = total - len(queue) - 1
            sub = HINT.get(label)
            nxt = queue[0] if queue else "done"
            ok, _, _ = phase(READY_S, f"READY: {label}", label, sub)  # form the pose before GO
            if not ok:
                return
            dur = DYNAMIC_S[label] if label in DYNAMIC else STATIC_S
            ok, ts, lms = phase(dur, f"[{done}/{total}] GO: {label}  (next: {nxt})", label, sub,
                                record=True)
            if not ok:
                return
            # Rest after every clip: without it the next clip starts in the previous pose
            # (all pinch clips started already pinched).
            ok, _, _ = phase(REST_S, f"REST: open your hand - next: {nxt}")
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
