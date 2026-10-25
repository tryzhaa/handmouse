"""Frame-level pose MLP that replaces the hand-written finger rules inside classify_clip.

Everything else (tap vs hold, swipes, draw-X) stays rule-based, so the score difference against
the rules baseline isolates what learning the pose adds.
"""
import numpy as np

from .data import Clip
from .features import mirror_for, pose_features
from .mlp import MLP
from .rules import classify_clip

# Steady-pose clips only; a quick-tap clip is mostly an open hand, so it cannot label frames.
FRAME_CLASS = {"point": "point", "palm": "palm", "two_fingers": "two_fingers", "pinch_hold": "pinch"}
TRIM_S = 0.4  # skip the start of a clip, where the hand is still forming the pose
AUG_COPIES = 4
STRETCH = 0.15  # per-copy anisotropic stretch range, stands in for out-of-plane tilt
JITTER = 0.005  # landmark noise in frame units
L2_GRID = [1e-3, 1e-2, 1e-1]
EPOCH_STEPS = [3, 7, 15, 25, 50]  # cumulative 3, 10, 25, 50, 100 epochs; best is early stopping


def _augment(lms: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotation/scale/translation are already normalised away by the features, so only
    perturbations that change the normalised shape are useful."""
    s = rng.uniform(1 - STRETCH, 1 + STRETCH, 2)
    out = lms.copy()
    out[:, :, :2] = lms[:, :1, :2] + (lms[:, :, :2] - lms[:, :1, :2]) * s
    out[:, :, :2] += rng.normal(0, JITTER, out[:, :, :2].shape)
    return out


def _dataset(clips: list[Clip], copies: int, rng: np.random.Generator):
    X, y = [], []
    for c in clips:
        if c.label not in FRAME_CLASS:
            continue
        lms = c.landmarks[c.timestamps >= TRIM_S]
        mirror = mirror_for(c.handedness)
        for k in range(copies + 1):
            frames = lms if k == 0 else _augment(lms, rng)
            X.append(np.stack([pose_features(f, mirror=mirror) for f in frames]))
            y += [FRAME_CLASS[c.label]] * len(lms)
    return np.vstack(X), np.array(y)


def _condition(c: Clip) -> tuple[str, str]:
    return c.lighting, c.handedness


def _select(train: list[Clip], classes: list[str], rng) -> tuple[float, int]:
    """Inner leave-one-condition-out: pick (l2, epochs) without touching the held-out condition."""
    conds = sorted({_condition(c) for c in train})
    if len(conds) < 2:
        return 1e-2, 40
    scores: dict[tuple[float, int], list[float]] = {}
    for g in conds:
        Xtr, ytr = _dataset([c for c in train if _condition(c) != g], AUG_COPIES, rng)
        Xva, yva = _dataset([c for c in train if _condition(c) == g], 0, rng)
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
        yt = np.array([classes.index(v) for v in ytr])
        yv = np.array([classes.index(v) for v in yva])
        for l2 in L2_GRID:
            model, done = MLP([Xtr.shape[1], 32, 16, len(classes)]), 0
            for step in EPOCH_STEPS:
                model.fit((Xtr - mu) / sd, yt, epochs=step, l2=l2, seed=done)
                done += step
                acc = float((model.predict((Xva - mu) / sd) == yv).mean())
                scores.setdefault((l2, done), []).append(acc)
    return max(scores, key=lambda k: np.mean(scores[k]))


def fit_pose_mlp(train: list[Clip], log: list | None = None):
    rng = np.random.default_rng(0)
    X, y = _dataset(train, AUG_COPIES, rng)
    classes = sorted(set(y))
    l2, epochs = _select(train, classes, rng)
    if log is not None:
        log.append({"l2": l2, "epochs": epochs})
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
    yi = np.array([classes.index(v) for v in y])
    model = MLP([X.shape[1], 32, 16, len(classes)]).fit((X - mu) / sd, yi, epochs=epochs, l2=l2)

    def predict(c: Clip) -> str:
        mirror = mirror_for(c.handedness)
        F = np.stack([pose_features(f, mirror=mirror) for f in c.landmarks])
        labels = [classes[i] for i in model.predict((F - mu) / sd)]
        return classify_clip(c.landmarks, c.timestamps, frame_labels=lambda _: labels)

    return predict
