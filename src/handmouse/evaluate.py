"""Evaluation harness: python -m handmouse.evaluate rules

Leave-one-subject-out (frozen contract 2). Clip-level only for now: false triggers per hour
(contract 5) needs the streaming harness, so it is written as null rather than omitted.
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .data import Clip, load_clips
from .rules import classify_clip

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def confusion_matrix(y_true, y_pred, labels):
    i = {l: k for k, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        m[i[t]][i[p]] += 1
    return m


def macro_f1(y_true, y_pred) -> float:
    classes = sorted(set(y_true))
    f1s = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s)


def split_by(clips: list[Clip]):
    """Subject-disjoint when there are several people. A solo dataset cannot be, so it is split
    by recording condition instead and is NOT evidence of working on new people."""
    if len({c.subject_id for c in clips}) > 1:
        return (lambda c: c.subject_id), "leave-one-subject-out"
    return (lambda c: f"{c.lighting}/{c.handedness}"), "leave-one-condition-out"


def loso(clips: list[Clip], fit, key=lambda c: c.subject_id):
    """fit(train_clips) -> predict(clip) -> label. Returns (y_true, y_pred, per-group macro-F1)."""
    y_true, y_pred, per_group = [], [], {}
    for g in sorted({key(c) for c in clips}):
        train = [c for c in clips if key(c) != g]
        test = [c for c in clips if key(c) == g]
        assert not {key(c) for c in train} & {key(c) for c in test}
        predict = fit(train)
        yt, yp = [c.label for c in test], [predict(c) for c in test]
        per_group[g] = macro_f1(yt, yp)
        y_true += yt
        y_pred += yp
    return y_true, y_pred, per_group


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], check=False, capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "rules"
    fits = {"rules": lambda train: (lambda c: classify_clip(c.landmarks, c.timestamps))}
    clips = [c for c in load_clips() if c.label != "negative"]
    if not clips:
        sys.exit("no gesture clips in data/raw; run the recorder first")
    key, split = split_by(clips)
    n_subjects = len({c.subject_id for c in clips})
    y_true, y_pred, per_group = loso(clips, fits[model], key)
    labels = sorted(set(y_true) | set(y_pred))
    result = {
        "model": model,
        "split": split,
        "unseen_person": n_subjects > 1,
        "n_clips": len(clips),
        "n_subjects": n_subjects,
        "macro_f1": macro_f1(y_true, y_pred),
        "per_group_macro_f1": per_group,
        "labels": labels,
        "confusion": confusion_matrix(y_true, y_pred, labels),
        "false_triggers_per_hour": None,  # not measured: clip-level eval only
        "git": git_state(),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{model}-{datetime.now():%Y%m%dT%H%M%S}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"{model}: macro-F1 {result['macro_f1']:.3f} over {len(clips)} clips, "
          f"{n_subjects} subject(s), {split}; false triggers/hour: not measured")
    if n_subjects == 1:
        print("NOTE: single subject, so this says nothing about new people.")
    print("rows=true, cols=pred:", labels)
    for l, row in zip(labels, result["confusion"]):
        print(f"{l:>12}", row)
    print("wrote", out)


if __name__ == "__main__":
    main()
