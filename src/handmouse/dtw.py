"""Rung 2 for motion: k-nearest-neighbour with dynamic time warping, from scratch in NumPy.

Multi-dimensional DTW with squared-Euclidean local cost and a Sakoe-Chiba window; returns the
square root of the path cost (same convention as dtaidistance.dtw_ndim.distance).
LB_Keogh lower bounds let 1-NN search skip most full DTW computations.
"""
import numpy as np


def dtw(a: np.ndarray, b: np.ndarray, window: int | None = None) -> float:
    n, m = len(a), len(b)
    w = max(n, m) if window is None else max(window, abs(n - m))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - w), min(m, i + w)
        cost = ((a[i - 1] - b[lo - 1 : hi]) ** 2).sum(axis=1)
        for k, j in enumerate(range(lo, hi + 1)):
            D[i, j] = cost[k] + min(D[i - 1, j], D[i - 1, j - 1], D[i, j - 1])
    return float(np.sqrt(D[n, m]))


def envelope(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension running min/max over +-window, for LB_Keogh."""
    n = len(x)
    lo = np.stack([x[max(0, i - window) : i + window + 1].min(axis=0) for i in range(n)])
    hi = np.stack([x[max(0, i - window) : i + window + 1].max(axis=0) for i in range(n)])
    return lo, hi


def lb_keogh(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Lower bound on dtw(q, c, window) given c's envelope; q and c must be the same length."""
    above = np.clip(q - hi, 0, None)
    below = np.clip(lo - q, 0, None)
    return float(np.sqrt((above**2 + below**2).sum()))


class KNNDTW:
    def __init__(self, k: int = 1, window: int = 4):
        self.k, self.window = k, window

    def fit(self, X: list[np.ndarray], y: list[str]):
        self.X, self.y = X, y
        self.env = [envelope(x, self.window) for x in X]
        return self

    def predict_one(self, q: np.ndarray) -> str:
        best: list[tuple[float, str]] = []  # k smallest (distance, label), sorted
        for x, label, (lo, hi) in zip(self.X, self.y, self.env):
            if len(best) == self.k and lb_keogh(q, lo, hi) >= best[-1][0]:
                continue  # cannot beat the current k-th neighbour
            d = dtw(q, x, self.window)
            if len(best) < self.k or d < best[-1][0]:
                best = sorted([*best, (d, label)])[: self.k]
        votes = [label for _, label in best]
        return max(set(votes), key=lambda v: (votes.count(v), -votes.index(v)))
