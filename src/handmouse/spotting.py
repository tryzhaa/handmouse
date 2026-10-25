"""Streaming gesture spotting: a sliding buffer over the live landmark stream, hysteresis and a
cooldown so one gesture fires once, plus the streaming metrics (false triggers per hour,
event precision/recall/delay). The same Spotter runs live and on recorded streams.

Rejection is the rules' motion threshold plus hysteresis; a calibrated confidence threshold
needs a probabilistic classifier and is not here yet.
"""
from collections import Counter, deque

import numpy as np

from .data import Clip
from .rules import classify_clip, frame_label

BUFFER_S = 2.0  # classifier sees the last BUFFER_S seconds
GAP_S = 0.5  # a gap this long (hand lost) resets the buffer
EVAL_EVERY = 3  # classify every N frames (~10 Hz at 30 fps)
MIN_FRAMES = 10
CONFIRM = 3  # consecutive agreeing evaluations before an event fires (hysteresis)
COOLDOWN_S = 1.0  # no second event within this time
# Discrete commands. point / two_fingers / pinch_hold are continuous modes, not events.
EVENT_LABELS = {"pinch", "palm", "swipe_left", "swipe_right", "swipe_up", "draw_x"}


class Spotter:
    def __init__(self, frame_label_fn=frame_label):
        self.frame_label_fn = frame_label_fn
        self.last_fire = -np.inf
        self._reset()

    def _reset(self) -> None:
        self.lms, self.ts, self.labels = deque(), deque(), deque()
        self.n, self.streak = 0, (None, 0)

    def push(self, lm: np.ndarray, t: float) -> str | None:
        """Feed one frame; returns an event label when one fires, else None."""
        if self.ts and t - self.ts[-1] > GAP_S:
            self._reset()
        self.lms.append(lm)
        self.ts.append(t)
        self.labels.append(self.frame_label_fn(lm))  # each frame labelled once
        while t - self.ts[0] > BUFFER_S:
            self.lms.popleft()
            self.ts.popleft()
            self.labels.popleft()
        self.n += 1
        if self.n % EVAL_EVERY or len(self.ts) < MIN_FRAMES:
            return None
        labels = list(self.labels)
        label = classify_clip(np.stack(self.lms), np.array(self.ts), frame_labels=lambda _: labels)
        prev, k = self.streak
        k = k + 1 if label == prev else 1
        self.streak = (label, k)
        if label in EVENT_LABELS and k == CONFIRM and t - self.last_fire >= COOLDOWN_S:
            self.last_fire = t
            return label
        return None


def spot(landmarks: np.ndarray, timestamps: np.ndarray) -> list[tuple[float, str]]:
    sp = Spotter()
    out = []
    for lm, t in zip(landmarks, timestamps):
        if (e := sp.push(lm, float(t))) is not None:
            out.append((float(t), e))
    return out


def hand_visible_hours(timestamps: np.ndarray) -> float:
    d = np.diff(np.asarray(timestamps, dtype=float))
    return float(d[d <= GAP_S].sum() / 3600)


def false_triggers(negatives: list[Clip]) -> dict:
    """Every event fired on non-gesture recordings is a false trigger."""
    events = [e for c in negatives for _, e in spot(c.landmarks, c.timestamps)]
    hours = sum(hand_visible_hours(c.timestamps) for c in negatives)
    return {
        "events": len(events),
        "hand_visible_hours": round(hours, 4),
        "per_hour": len(events) / hours if hours > 0 else None,
        "by_label": dict(Counter(events)),
    }


def match_events(pred: list[tuple[float, str]], truth: list[tuple[float, str]],
                 window_s: float = 2.5) -> dict:
    """Greedy match: a prediction counts for a prompt if same label and 0..window_s s after it."""
    used, delays = set(), []
    for tt, tl in sorted(truth):
        for i, (pt, pl) in enumerate(sorted(pred)):
            if i not in used and pl == tl and 0 <= pt - tt <= window_s:
                used.add(i)
                delays.append(pt - tt)
                break
    return {
        "precision": len(used) / len(pred) if pred else None,
        "recall": len(used) / len(truth) if truth else None,
        "median_delay_s": float(np.median(delays)) if delays else None,
    }
