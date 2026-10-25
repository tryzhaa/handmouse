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
