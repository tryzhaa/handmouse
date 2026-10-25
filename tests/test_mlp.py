import numpy as np

from handmouse.mlp import MLP, gradient_check


def test_backprop_matches_numerical_gradient():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 6))
    y = rng.integers(0, 3, 20)
    assert gradient_check(MLP([6, 8, 5, 3], seed=2), X, y) < 1e-5


def test_learns_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 25, dtype=float)
    y = np.array([0, 1, 1, 0] * 25)
    m = MLP([2, 8, 2], seed=0).fit(X, y, epochs=300, lr=0.1, batch=16, l2=0.0)
    assert (m.predict(X) == y).all()


def test_pose_mlp_end_to_end_on_synthetic_clips():
    from test_rules_eval import clip_of, hand

    from handmouse.pose_mlp import _augment, fit_pose_mlp

    poses = {"palm": hand((True,) * 4), "point": hand((True, False, False, False)),
             "two_fingers": hand((True, True, False, False)), "pinch_hold": hand(pinch=True)}
    clips = []
    for light in ("daylight", "tube"):
        for label, lm in poses.items():
            c = clip_of([lm] * 30, label)
            c.lighting = light
            clips.append(c)
    a = np.stack([poses["palm"]] * 3)
    assert np.array_equal(_augment(a, np.random.default_rng(1)), _augment(a, np.random.default_rng(1)))
    log = []
    predict = fit_pose_mlp(clips, log)
    assert [predict(c) for c in clips] == [c.label for c in clips] and "l2" in log[0]
