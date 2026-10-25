"""KNN-DTW over the hand's screen-space trajectory decides motion vs static and which motion.

Static clips are pooled into one "static" class; those then go to the rule-based pose logic,
so the score difference against the rules baseline isolates what KNN-DTW adds for motion.
Trajectories are not mirrored for left hands: swipe direction is screen-space.
"""
import numpy as np

from .data import Clip
from .dtw import KNNDTW
from .recorder import DYNAMIC
from .rules import classify_clip

N_POINTS = 20
FLOOR = 0.15  # normalise mode: extents below this are not blown up, so a still hand stays small


def trajectory(c: Clip, normalise: bool = False) -> np.ndarray:
    m = c.landmarks[:, :, :2].mean(axis=1)
    if len(m) >= 5:  # 3-frame median drops single-frame tracking glitches
        m = np.median([m[:-2], m[1:-1], m[2:]], axis=0)
    m = m - m[0]
    if normalise:
        m = m / max(np.abs(m).max(), FLOOR)
    t = np.linspace(0, 1, len(m))
    u = np.linspace(0, 1, N_POINTS)
    return np.stack([np.interp(u, t, m[:, d]) for d in range(2)], axis=1)


def fit_knn_dtw(train: list[Clip], normalise: bool = False, k: int = 1):
    model = KNNDTW(k=k).fit([trajectory(c, normalise) for c in train],
                            [c.label if c.label in DYNAMIC else "static" for c in train])

    def predict(c: Clip) -> str:
        label = model.predict_one(trajectory(c, normalise))
        if label == "static":
            return classify_clip(c.landmarks, c.timestamps, motion_min=float("inf"))
        return label

    return predict
