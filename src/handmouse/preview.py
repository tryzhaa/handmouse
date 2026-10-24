"""Live preview: python -m handmouse.preview  (press q to quit)."""
import time

import cv2

from .tracker import HandTracker, draw


def main() -> None:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    tracker = HandTracker()
    last, fps = time.monotonic(), 0.0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror so movement feels natural
            lm = tracker.process(frame)
            if lm is not None:
                draw(frame, lm)
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            cv2.putText(frame, f"{fps:4.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.imshow("handmouse", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
