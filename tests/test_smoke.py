import handmouse
from handmouse.tracker import CONNECTIONS


def test_version():
    assert handmouse.__version__


def test_connections_cover_21_landmarks():
    assert {i for pair in CONNECTIONS for i in pair} == set(range(21))
