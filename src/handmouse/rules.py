"""Rung 1: hand-written rule baselines. No training; thresholds are first guesses.

shortcut: thresholds are untuned guesses, tune on training conditions only once data exists.
"""
from collections import Counter

import numpy as np

from .features import normalise

PINCH_DIST = 0.35  # thumb tip to index tip, in palm lengths
WINDOW_S = 0.4  # a swipe's stroke is found as the fastest window of this length
MOTION_MIN = 0.15  # frame-width displacement in that window above which a clip is dynamic
MULTI_STROKE = 2.5  # total path / fastest-stroke displacement above which it is draw_x
HOLD_FRAC = 0.6  # fraction of pinched frames above which a pinch is a hold
TAP_FRAC = 0.05

# MCP, PIP, tip indices for index, middle, ring, pinky
_FINGERS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]


def frame_label(lm: np.ndarray) -> str:
    p = normalise(lm)
    if np.linalg.norm(p[4] - p[8]) < PINCH_DIST:
        return "pinch"
    d = np.linalg.norm(p, axis=1)
    idx, mid, ring, pinky = (d[tip] > d[pip] for _, pip, tip in _FINGERS)
    if idx and mid and ring and pinky:
        return "palm"
    if idx and mid and not ring and not pinky:
        return "two_fingers"
    if idx and not (mid or ring or pinky):
        return "point"
    return "other"


def fastest_stroke(c: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Displacement vector of the largest move within any WINDOW_S window."""
    best = np.zeros(2)
    for i in range(len(c)):
        j = int(np.searchsorted(t, t[i] + WINDOW_S))
        if j >= len(c):
            break
        if np.linalg.norm(c[j] - c[i]) > np.linalg.norm(best):
            best = c[j] - c[i]
    return best


def classify_clip(landmarks: np.ndarray, timestamps: np.ndarray) -> str:
    c = landmarks[:, :, :2].mean(axis=1)
    t = np.asarray(timestamps)
    if len(c) >= 5:  # 3-frame median drops single-frame tracking glitches
        c = np.median([c[:-2], c[1:-1], c[2:]], axis=0)
        t = t[1:-1]
    stroke = fastest_stroke(c, t)
    peak = float(np.linalg.norm(stroke))
    if peak > MOTION_MIN:
        path = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
        if path / peak > MULTI_STROKE:
            return "draw_x"
        dx, dy = stroke
        if abs(dx) >= abs(dy):
            return "swipe_right" if dx > 0 else "swipe_left"
        return "swipe_up" if dy < 0 else "other"  # no swipe-down class
    labels = [frame_label(f) for f in landmarks]
    pinch_frac = labels.count("pinch") / len(labels)
    if pinch_frac > HOLD_FRAC:
        return "pinch_hold"
    if pinch_frac > TAP_FRAC:
        return "pinch"
    return Counter(label for label in labels if label != "pinch").most_common(1)[0][0]
