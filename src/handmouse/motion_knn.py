"""KNN-DTW over the hand's screen-space trajectory decides motion vs static and which motion.

Static clips are pooled into one "static" class; those then go to the rule-based pose logic,
so the score difference against the rules baseline isolates what KNN-DTW adds for motion.
Trajectories are not mirrored for left hands: swipe direction is screen-space.
"""
import numpy as np

from .data import Clip
from .dtw import KNNDTW
from .recorder import DYNAMIC
from .rules import classify_clip, fastest_window

N_POINTS = 20
FLOOR = 0.15  # normalise mode: extents below this are not blown up, so a still hand stays small
STROKE_GRID_S = [0.4, 0.6, 0.8, 1.0]
K_GRID = [1, 3]


def trajectory(c: Clip, normalise: bool = False, stroke_s: float | None = None) -> np.ndarray:
    """Resampled centroid path; with stroke_s, only the stroke_s seconds around the fastest move."""
    m = c.landmarks[:, :, :2].mean(axis=1)
    t = np.asarray(c.timestamps, dtype=float)
    if len(m) >= 5:  # 3-frame median drops single-frame tracking glitches
        m = np.median([m[:-2], m[1:-1], m[2:]], axis=0)
        t = t[1:-1]
    if stroke_s is not None:
        i, j = fastest_window(m, t)
        keep = np.abs(t - (t[i] + t[j]) / 2) <= stroke_s / 2
        keep[i : j + 1] = True
        m, t = m[keep], t[keep]
    m = m - m[0]
    if normalise:
        m = m / max(np.abs(m).max(), FLOOR)
    s = np.linspace(0, 1, len(m))
    u = np.linspace(0, 1, N_POINTS)
    return np.stack([np.interp(u, s, m[:, d]) for d in range(2)], axis=1)


def fit_knn_dtw(train: list[Clip], normalise: bool = False, k: int = 1,
                stroke_s: float | None = None):
    model = KNNDTW(k=k).fit([trajectory(c, normalise, stroke_s) for c in train],
                            [c.label if c.label in DYNAMIC else "static" for c in train])

    def predict(c: Clip) -> str:
        label = model.predict_one(trajectory(c, normalise, stroke_s))
        if label == "static":
            return classify_clip(c.landmarks, c.timestamps, motion_min=float("inf"))
        return label

    return predict


def fit_knn_dtw_stroke(train: list[Clip], log: list | None = None):
    """Stroke-window KNN-DTW; (stroke_s, k) picked by inner leave-one-condition-out on train."""
    from .evaluate import macro_f1  # local: evaluate imports this module

    def cond(c):
        return c.lighting, c.handedness

    conds = sorted({cond(c) for c in train})
    best = (0.6, 1)
    if len(conds) >= 2:
        scores = {}
        for s in STROKE_GRID_S:
            for k in K_GRID:
                f1s = []
                for g in conds:
                    inner_tr = [c for c in train if cond(c) != g]
                    val = [c for c in train if cond(c) == g]
                    predict = fit_knn_dtw(inner_tr, k=k, stroke_s=s)
                    f1s.append(macro_f1([c.label for c in val], [predict(c) for c in val]))
                scores[(s, k)] = float(np.mean(f1s))
        best = max(scores, key=scores.get)
    if log is not None:
        log.append({"stroke_s": best[0], "k": best[1]})
    return fit_knn_dtw(train, k=best[1], stroke_s=best[0])
