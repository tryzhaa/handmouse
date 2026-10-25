import numpy as np

from handmouse.cursor import OneEuro, map_to_screen
from handmouse.features import (
    N_ANGLES,
    joint_angles,
    mirror_for,
    normalise,
    pose_features,
    sequence_features,
)
from handmouse.recorder import GESTURES, build_schedule


def _hand(seed=0):
    return np.random.default_rng(seed).uniform(0.2, 0.8, (21, 3)).astype(np.float32)


def test_normalise_invariant_to_translation_scale_rotation():
    lm = _hand()
    th = 0.7
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    moved = lm.copy()
    moved[:, :2] = (lm[:, :2] @ rot.T) * 1.7 + 0.3
    assert np.allclose(normalise(lm), normalise(moved), atol=1e-4)
    assert np.allclose(normalise(lm)[9], [0, -1], atol=1e-5)


def test_joint_angles_straight_finger_is_pi():
    lm = np.zeros((21, 3), np.float32)
    for i in (5, 6, 7, 8):
        lm[i] = [0.5, 0.5 - 0.1 * (i - 4), 0]
    lm[0] = [0.5, 0.5, 0]
    assert np.isclose(joint_angles(lm)[3:6], np.pi, atol=1e-2).all()
    assert joint_angles(lm).shape == (N_ANGLES,)


def test_sequence_features_shape_and_wrist_velocity():
    seq = np.stack([_hand(0)] * 5)
    seq[:, 0, 0] += np.arange(5) * 0.1  # wrist moves +x at 0.1 per 0.1 s
    f = sequence_features(seq, np.arange(5) * 0.1)
    d = pose_features(seq[0]).shape[0]
    assert f.shape == (5, 2 * d + 2)
    assert np.isclose(f[1:, -2], 1.0, atol=1e-4).all()


def test_one_euro_smooths_jitter_and_passes_constant():
    f = OneEuro()
    assert f(0.5, 0.0) == 0.5
    rng = np.random.default_rng(0)
    xs = [f(0.5 + rng.normal(0, 0.01), 0.033 * i) for i in range(1, 200)]
    assert np.std(xs[50:]) < 0.01


def test_map_to_screen_corners_and_clip():
    s = (1000, 500)
    assert np.allclose(map_to_screen((0.2, 0.2), s), [0, 0])
    assert np.allclose(map_to_screen((0.8, 0.8), s), [999, 499])
    assert np.allclose(map_to_screen((0.0, 1.0), s), [0, 499])


def test_schedule_balanced_and_deterministic():
    s = build_schedule(GESTURES, 10)
    assert len(s) == 10 * len(GESTURES) and s == build_schedule(GESTURES, 10)
    assert all(s.count(g) == 10 for g in GESTURES)


def test_mirror_makes_left_hand_match_right_but_keeps_wrist_velocity():
    right = _hand()
    left = right.copy()
    left[:, 0] = 1.0 - left[:, 0]
    assert np.allclose(pose_features(right), pose_features(left, mirror=True), atol=1e-5)
    assert not np.allclose(pose_features(right), pose_features(left), atol=1e-3)
    seq_r = np.stack([right] * 4)
    seq_r[:, 0, 0] += np.arange(4) * 0.1
    seq_l = seq_r.copy()
    seq_l[:, :, 0] = 1.0 - seq_l[:, :, 0]
    t = np.arange(4) * 0.1
    # screen-space wrist velocity is unchanged by mirror=True (a left-hand swipe keeps its sign)
    assert np.allclose(sequence_features(seq_l, t, mirror=True)[:, -2:],
                       sequence_features(seq_l, t)[:, -2:])
    assert mirror_for("left") and not mirror_for("right")
