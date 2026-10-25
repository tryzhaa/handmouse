import numpy as np
import pytest

from handmouse.dtw import KNNDTW, dtw, envelope, lb_keogh


def _series(rng, n=20):
    return np.cumsum(rng.normal(size=(n, 2)), axis=0)


@pytest.mark.parametrize("window", [None, 2, 5])
def test_dtw_matches_dtaidistance(window):
    dtw_ndim = pytest.importorskip("dtaidistance.dtw_ndim")
    rng = np.random.default_rng(0)
    for _ in range(10):
        a, b = _series(rng), _series(rng)
        ref = dtw_ndim.distance(a, b, window=None if window is None else window + 1, use_c=False)
        assert np.isclose(dtw(a, b, window), ref)


def test_dtw_basic_properties():
    a = np.array([[0.0, 0], [1, 0], [2, 0]])
    assert dtw(a, a) == 0.0
    stretched = np.array([[0.0, 0], [0, 0], [1, 0], [2, 0], [2, 0]])
    assert dtw(a, stretched) == 0.0  # warping absorbs repeated points


def test_lb_keogh_never_exceeds_dtw():
    rng = np.random.default_rng(1)
    for _ in range(50):
        q, c = _series(rng), _series(rng)
        lo, hi = envelope(c, 3)
        assert lb_keogh(q, lo, hi) <= dtw(q, c, 3) + 1e-9


def test_pruned_knn_equals_brute_force():
    rng = np.random.default_rng(2)
    X = [_series(rng) for _ in range(40)]
    y = [str(i % 4) for i in range(40)]
    knn = KNNDTW(k=1, window=3).fit(X, y)
    for _ in range(10):
        q = _series(rng)
        brute = y[int(np.argmin([dtw(q, x, 3) for x in X]))]
        assert knn.predict_one(q) == brute


def test_stroke_window_crops_to_the_fast_move():
    from test_rules_eval import along, clip_of

    from handmouse.motion_knn import N_POINTS, trajectory
    from handmouse.rules import fastest_window

    # 1 s still, 0.3 s fast sweep left, 1 s still (30 fps)
    frames = along([0, 30, 39, 69], [0, 0, -0.4, -0.4], [0, 0, 0, 0], n=70)
    c = clip_of(frames, "swipe_left")
    full, crop = trajectory(c), trajectory(c, stroke_s=0.6)
    assert crop.shape == (N_POINTS, 2)
    assert np.isclose(crop[-1, 0], -0.4, atol=0.05)  # whole sweep kept
    # the cropped path spends far fewer resampled points standing still
    assert (np.abs(np.diff(crop[:, 0])) > 1e-3).sum() > (np.abs(np.diff(full[:, 0])) > 1e-3).sum()
    i, j = fastest_window(np.stack(frames)[:, :, :2].mean(axis=1), c.timestamps)
    assert 25 <= i <= 40 and j > i
