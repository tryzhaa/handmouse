"""Cursor path (signal processing, no ML): python -m handmouse.cursor [--live]

Index fingertip -> One Euro filter -> screen mapping -> pynput move.
Dry-run by default (logs intended moves). Esc on the keyboard is the kill switch, q in the window too.
"""
import argparse
import math
import time

import cv2
import numpy as np

from .tracker import HandTracker, draw

INDEX_TIP = 8


class OneEuro:
    """One Euro filter (Casiez 2012): low jitter when still, low lag when fast."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x = self._dx = self._t = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._t is None:
            self._x, self._dx, self._t = x, 0.0, t
            return x
        dt = max(t - self._t, 1e-6)
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * (x - self._x) / dt + (1 - a_d) * self._dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self._dx), dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x


def map_to_screen(xy, screen: tuple[int, int], margin: float = 0.2, gamma: float = 1.0):
    """Map the comfortable camera region [margin, 1-margin]^2 to the screen.

    gamma > 1 is a simple acceleration curve: slow near the centre, fast toward the edges.
    """
    u = (np.asarray(xy, dtype=np.float64) - margin) / (1 - 2 * margin)
    u = np.clip(u, 0.0, 1.0) * 2 - 1
    u = np.sign(u) * np.abs(u) ** gamma
    return (u + 1) / 2 * (np.array(screen) - 1)


def screen_size() -> tuple[int, int]:
    import Quartz  # macOS only; pulled in by pynput

    b = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    return int(b.size.width), int(b.size.height)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="actually move the mouse (default: dry-run)")
    ap.add_argument("--gamma", type=float, default=1.0)
    args = ap.parse_args()

    from pynput import keyboard, mouse

    killed = False

    def on_press(key):
        nonlocal killed
        if key == keyboard.Key.esc:
            killed = True
            return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    ctl = mouse.Controller() if args.live else None
    screen = screen_size()
    fx, fy = OneEuro(), OneEuro()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    tracker = HandTracker()
    last_log = 0.0
    print("LIVE" if args.live else "DRY-RUN", "- press Esc to stop")
    try:
        while cap.isOpened() and not killed:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            lm = tracker.process(frame)
            now = time.monotonic()
            if lm is not None:
                draw(frame, lm)
                x, y = lm[INDEX_TIP, :2]
                target = map_to_screen((fx(float(x), now), fy(float(y), now)), screen,
                                       gamma=args.gamma)
                if args.live:
                    ctl.position = (int(target[0]), int(target[1]))
                elif now - last_log > 0.25:
                    print(f"would move to ({target[0]:.0f}, {target[1]:.0f})")
                    last_log = now
            cv2.imshow("handmouse cursor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        listener.stop()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
