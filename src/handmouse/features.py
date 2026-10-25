"""Pure feature functions over landmark arrays. Shared by live inference and offline training.

Default is x/y only (HaGRID has no z); pass use_z=True for the ablated extra.
"""
import numpy as np

# Interior joints of each finger chain (wrist->tip): angle is measured at the middle index.
_TRIPLES = [
    (a, b, c)
    for chain in ([0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
                  [0, 13, 14, 15, 16], [0, 17, 18, 19, 20])
    for a, b, c in zip(chain, chain[1:], chain[2:])
]
N_ANGLES = len(_TRIPLES)  # 15


def _dims(use_z: bool) -> int:
    return 3 if use_z else 2


def mirror_for(handedness: str) -> bool:
    """Canonical hand is right; left-hand shapes get mirrored so one model serves both."""
    return handedness == "left"


def normalise(lm: np.ndarray, use_z: bool = False, mirror: bool = False) -> np.ndarray:
    """Wrist-relative, palm-scaled, rotated so wrist->middle-MCP points up. Returns (21, d)."""
    if mirror:
        lm = lm.copy()
        lm[:, 0] = 1.0 - lm[:, 0]
    p = lm[:, : _dims(use_z)].astype(np.float64) - lm[0, : _dims(use_z)]
    axis = p[9, :2]
    scale = np.linalg.norm(axis)
    if scale < 1e-6:
        return p.astype(np.float32)
    p = p / scale
    # image y points down, so "up" is (0, -1); rotate the palm axis onto it
    theta = np.arctan2(axis[1], axis[0]) - np.arctan2(-1.0, 0.0)
    c, s = np.cos(-theta), np.sin(-theta)
    p[:, :2] = p[:, :2] @ np.array([[c, s], [-s, c]])
    return p.astype(np.float32)


def joint_angles(lm: np.ndarray, use_z: bool = False) -> np.ndarray:
    """(15,) joint angles in radians, in [0, pi]; straight finger is pi."""
    p = lm[:, : _dims(use_z)].astype(np.float64)
    out = np.empty(N_ANGLES, dtype=np.float32)
    for i, (a, b, c) in enumerate(_TRIPLES):
        u, v = p[a] - p[b], p[c] - p[b]
        cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
        out[i] = np.arccos(np.clip(cos, -1.0, 1.0))
    return out


def pose_features(lm: np.ndarray, use_z: bool = False, mirror: bool = False) -> np.ndarray:
    """Single-frame vector: normalised coords of the 20 non-wrist points + joint angles."""
    return np.concatenate([normalise(lm, use_z, mirror)[1:].ravel(), joint_angles(lm, use_z)])


def sequence_features(landmarks: np.ndarray, timestamps: np.ndarray,
                      use_z: bool = False, mirror: bool = False) -> np.ndarray:
    """(T, F) per-frame features: pose, its velocity, and wrist velocity (frame units / s).

    Pose features are wrist-relative, so the wrist velocity carries the swipe direction.
    mirror flips hand shape only; wrist velocity stays in screen space so "swipe left" keeps
    its meaning for a left hand.
    """
    pose = np.stack([pose_features(f, use_z, mirror) for f in landmarks])
    wrist = landmarks[:, 0, :2].astype(np.float64)
    dt = np.maximum(np.diff(timestamps), 1e-3)[:, None]
    pose_v = np.vstack([np.zeros((1, pose.shape[1])), np.diff(pose, axis=0) / dt])
    wrist_v = np.vstack([np.zeros((1, 2)), np.diff(wrist, axis=0) / dt])
    return np.hstack([pose, pose_v, wrist_v]).astype(np.float32)
