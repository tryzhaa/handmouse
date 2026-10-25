import numpy as np
from test_rules_eval import along, clip_of, hand

from handmouse.spotting import false_triggers, hand_visible_hours, match_events, spot

POINT = hand((True, False, False, False))


def _ts(n, start=0.0):
    return start + np.arange(n) / 30


def test_held_palm_fires_once_not_every_frame():
    lms = np.stack([hand((True,) * 4)] * 150)  # 5 s of open palm
    events = spot(lms, _ts(150))
    assert [e for _, e in events] == ["palm"]


def test_pointing_is_a_mode_not_an_event():
    assert spot(np.stack([POINT] * 150), _ts(150)) == []


def test_swipe_in_a_stream_fires_once():
    still = [POINT] * 30
    sweep = [f - hand((True,) * 4) + POINT for f in along([0, 9], [0, -0.4], [0, 0], n=10)]
    after = [POINT + np.array([-0.4, 0, 0], np.float32)] * 40
    lms = np.stack(still + sweep + after)
    events = spot(lms, _ts(len(lms)))
    assert [e for _, e in events] == ["swipe_left"]
    assert 1.0 <= events[0][0] <= 2.0


def test_gap_resets_buffer_and_is_not_counted_as_visible_time():
    t = np.concatenate([_ts(30), _ts(30, start=10.0)])
    assert np.isclose(hand_visible_hours(t) * 3600, 2 * 29 / 30)
    # a sweep split across a gap must not be stitched into one swipe
    lms = np.stack([POINT] * 30 + [POINT + np.array([-0.4, 0, 0], np.float32)] * 30)
    assert spot(lms, t) == []


def test_false_triggers_on_still_pointing_is_zero():
    ft = false_triggers([clip_of([POINT] * 300, "negative")])
    assert ft["events"] == 0 and ft["per_hour"] == 0.0


def test_match_events():
    truth = [(1.0, "swipe_left"), (5.0, "draw_x")]
    pred = [(1.4, "swipe_left"), (3.0, "pinch")]
    m = match_events(pred, truth)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and np.isclose(m["median_delay_s"], 0.4)
