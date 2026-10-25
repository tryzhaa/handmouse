# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
source .venv/bin/activate        # .venv already exists (python3.12)
pip install -e ".[dev]"
pytest                           # all tests
pytest tests/test_smoke.py::test_version   # single test
ruff check .                     # CI runs ruff then pytest
python -m handmouse.preview      # live webcam preview, q to quit
```

CI (`.github/workflows/ci.yml`) runs headless on ubuntu, so **nothing under test may open a camera
or a window**. Keep webcam/GUI code in entry points (`preview.py` and future `__main__`-style
modules) and keep the testable logic in pure functions over numpy arrays.

## Architecture

One landmark stream, two independent consumers. This split is the core design decision and it is
load-bearing for both latency and the project's claims:

```
webcam -> HandTracker.process() -> (21,3) landmarks
                                      |
            +-------------------------+-------------------------+
            |                                                   |
      cursor path                                        gesture path
   (signal processing, no ML)                        (trained models, ML)
   normalise -> One Euro filter ->                   features -> static pose clf
   accel curve -> pynput move                                 -> dynamic gesture clf
                                                              -> spotting (threshold,
                                                                 hysteresis, cooldown)
                                                              -> pynput click/keys
```

- `tracker.py` is the only place that touches MediaPipe. `HandTracker.process()` returns `(21,3)`
  float32 normalised landmarks or `None`; everything downstream consumes that array, never frames.
  It uses `RunningMode.VIDEO` with a monotonic timestamp, which requires monotonically increasing
  timestamps — do not reuse one `HandTracker` across two video sources.
- The cursor path is deliberately **not** learned. Don't "improve" it with a model; its value is
  that pointing stays fast and predictable.
- Feature extraction (wrist-relative, palm-scaled, rotation-normalised + joint angles + velocities)
  must be a pure function of a landmark array/sequence, shared by live inference and offline
  training. Live and training features drifting apart is the main source of "works in eval, fails
  on camera" in this project.

## Frozen contracts

Changing these invalidates recorded data or published results, so change them only deliberately.

1. **Store raw landmarks, never features.** A clip on disk is `subject_id`, `label`, `lighting`,
   `handedness`, per-frame timestamps, and `(T,21,3)` landmarks. Features are computed on load.
   This way feature changes cost a re-run, not a re-recording — and re-recording is impossible
   once participants have gone.
2. **Subject-disjoint splits always.** Split by `subject_id` (leave-one-subject-out). No subject
   may appear in both train and test, for any model or any reported number.
   **v1 exception (deliberate, solo dataset):** with one subject, `evaluate.split_by` splits by
   recording condition (lighting/hand) instead, results carry `unseen_person: false`, and no
   v1 number may be described as generalising to new people. Back to strict LOSO once a second
   subject exists.
3a. **One shared model for both hands.** Left-hand shapes are mirrored (`features.mirror_for`)
   before features; wrist velocity stays in screen space so swipe labels keep their meaning.
   Never train per-hand models.
3. **Release landmarks only.** Never write raw video to disk unless the participant opted in.
4. **A non-gesture kill switch, and a dry-run mode.** Anything that drives `pynput` must be
   killable from the keyboard (not from a gesture, which may be exactly what is misfiring) and
   must support a mode that logs intended actions instead of sending them. Debugging a cursor
   controller that owns your cursor is otherwise a lockout.
5. **Report false-triggers-per-hour alongside every accuracy number.** Accuracy on isolated clips
   is the tutorial-grade metric; the streaming false-trigger rate is the real one.

## External data constraints

- **HaGRID** annotations include 21 landmarks per hand as `[x, y]` relative image coords
  (MediaPipe-generated) plus `user_id`, `bboxes`, `labels`, `leading_hand`. Two consequences:
  **(a)** there is no `z`, so the pretraining feature set must be the x/y-only subset of ours;
  **(b)** HaGRID's label set is its own (call, like, ok, palm, peace, stop…), not ours, so this is
  pretrain-the-body / replace-the-head, not direct transfer. Download annotations only (~700GB of
  images is not needed).
- **IPN Hand** is held out as an external benchmark for spotting and false triggers. Don't train
  on it.

## macOS notes (dev machine is darwin)

`pynput` silently does nothing until the terminal/IDE is granted Accessibility permission
(System Settings → Privacy & Security → Accessibility), and `cv2.VideoCapture(0)` needs Camera
permission. Both fail as "my code is broken" bugs. Grant them before debugging input code.

## Project docs

`docs/plan.md` holds the dated 1-month plan (schedule, metrics, targets, cut order). When asked
what to work on, read it — it names the week's goal and what gets cut first if behind.
