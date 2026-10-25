import handmouse
from handmouse.tracker import CONNECTIONS


def test_version():
    assert handmouse.__version__


def test_connections_cover_21_landmarks():
    assert {i for pair in CONNECTIONS for i in pair} == set(range(21))


def test_save_clip_roundtrip(tmp_path):
    import numpy as np

    from handmouse.negative_logger import save_clip

    lms = [np.zeros((21, 3), np.float32)] * 3
    p = save_clip(tmp_path, "s01", "negative", "daylight", "right", [0.0, 0.03, 0.06], lms)
    d = np.load(p)
    assert d["landmarks"].shape == (3, 21, 3) and str(d["subject_id"]) == "s01"
