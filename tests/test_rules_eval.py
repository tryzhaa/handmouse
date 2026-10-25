import numpy as np

from handmouse.data import Clip
from handmouse.evaluate import confusion_matrix, loso, macro_f1
from handmouse.rules import classify_clip, frame_label

_X = {5: 0.44, 9: 0.50, 13: 0.56, 17: 0.62}
_EXT_Y, _CURL_Y = (0.52, 0.46, 0.40), (0.56, 0.62, 0.66)


def hand(ext=(False, False, False, False), pinch=False):
    lm = np.zeros((21, 3), np.float32)
    lm[0] = [0.5, 0.8, 0]
    lm[1:5] = [[0.42, 0.74, 0], [0.38, 0.70, 0], [0.35, 0.66, 0], [0.33, 0.62, 0]]
    for (mcp, x), e in zip(_X.items(), ext):
        ys = _EXT_Y if e else _CURL_Y
        lm[mcp] = [x, 0.6, 0]
        for k, y in enumerate(ys, 1):
            lm[mcp + k] = [x, y, 0]
    if pinch:
        lm[4] = lm[8] + 0.01
    return lm


def clip_of(frames, label="x", subject="s1"):
    lms = np.stack(frames)
    return Clip(subject, label, "d", "right", np.arange(len(lms)) / 30, lms)


def moving(dx, dy, n=30):
    return [hand((True,) * 4) + np.array([dx * k / (n - 1), dy * k / (n - 1), 0], np.float32)
            for k in range(n)]


def test_frame_labels():
    assert frame_label(hand((True,) * 4)) == "palm"
    assert frame_label(hand((True, False, False, False))) == "point"
    assert frame_label(hand((True, True, False, False))) == "two_fingers"
    assert frame_label(hand((True, False, False, False), pinch=True)) == "pinch"


def cc(frames):
    lms = np.stack(frames)
    return classify_clip(lms, np.arange(len(lms)) / 30)


def along(frame_idx, xs, ys, n=30):
    """Open hand whose position follows piecewise-linear waypoints over n frames at 30 fps."""
    k = np.arange(n)
    return [hand((True,) * 4) + np.array([np.interp(i, frame_idx, xs), np.interp(i, frame_idx, ys), 0],
                                         np.float32) for i in k]


def test_static_clips():
    assert cc([hand((True,) * 4)] * 20) == "palm"
    assert cc([hand(pinch=True)] * 20) == "pinch_hold"
    tap = [hand((True, False, False, False), pinch=True)] * 3 + [hand((True, False, False, False))] * 17
    assert cc(tap) == "pinch"


def test_swipes_by_fastest_stroke():
    assert cc(along([0, 10, 29], [0, -0.5, -0.5], [0, 0, 0])) == "swipe_left"
    assert cc(along([0, 10, 29], [0, 0.5, 0.5], [0, 0, 0])) == "swipe_right"
    assert cc(along([0, 10, 29], [0, 0, 0], [0, -0.5, -0.5])) == "swipe_up"


def test_swipe_with_slow_return_keeps_its_direction():
    # the old whole-clip rule saw net displacement ~0 here and called it something else
    assert cc(along([0, 10, 29], [0, -0.5, 0], [0, 0, 0])) == "swipe_left"


def test_draw_x_is_multi_stroke():
    x = along([0, 15, 25, 40, 60], [0, 0.3, 0.3, 0, 0], [0, 0.3, 0, 0.3, 0.3], n=60)
    assert cc(x) == "draw_x"


def test_macro_f1_and_confusion():
    assert macro_f1(["a", "b"], ["a", "b"]) == 1.0
    assert macro_f1(["a", "b"], ["a", "a"]) == (2 / 3 + 0) / 2
    assert confusion_matrix(["a", "b"], ["a", "a"], ["a", "b"]) == [[1, 0], [1, 0]]


def test_loso_never_shares_subjects():
    clips = [clip_of([hand((True,) * 4)] * 3, "palm", s) for s in ("a", "b", "c")]
    seen = []

    def fit(train):
        seen.append({c.subject_id for c in train})
        return lambda c: "palm"

    _, _, per = loso(clips, fit)
    assert seen == [{"b", "c"}, {"a", "c"}, {"a", "b"}] and per == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_solo_data_splits_by_condition_not_subject():
    from handmouse.evaluate import split_by

    solo = [clip_of([hand()] * 3, "palm", "s1")]
    solo[0].lighting, solo[0].handedness = "tube", "left"
    key, name = split_by(solo)
    assert name == "leave-one-condition-out" and key(solo[0]) == "tube/left"
    duo = solo + [clip_of([hand()] * 3, "palm", "s2")]
    assert split_by(duo)[1] == "leave-one-subject-out"


def test_motion_threshold_is_a_parameter_and_tuning_uses_train_only():
    from handmouse.evaluate import tune_rules

    small = along([0, 10, 29], [0, -0.12, -0.12], [0, 0, 0])  # 0.12-wide swipe
    lms = np.stack(small)
    ts = np.arange(len(lms)) / 30
    assert classify_clip(lms, ts) != "swipe_left"  # below the default 0.15
    assert classify_clip(lms, ts, motion_min=0.08) == "swipe_left"
    train = [clip_of(small, "swipe_left"), clip_of([hand((True,) * 4)] * 30, "palm")]
    assert tune_rules(train)[0] <= 0.10
