# Webcam Hand-Gesture Control — 1-Month Plan

Oct 25, 2026 · @trish · runs Mon 26 Oct – Wed 25 Nov 2026, solo, laptop CPU.

## Amendments (strategy pass, 25 Oct)

These override the week-by-week schedule below where they conflict.

1. **Recording is the critical path, not modelling.** Participants cannot be recruited
   retroactively; models can be trained any night. The recorder moves from Wed 28 to Mon 26–Tue 27,
   and the first outside participants are recorded from Wed 28, not Mon 2 Nov. This buys four extra
   recording days of slack before the Wed 4 Nov "fewer than 6 people" trigger.
2. **Negative data collection starts Mon 26 and runs passively all month.** A background logger
   that writes landmarks while you work costs no participant time and is what the
   false-triggers-per-hour claim rests on. It is the highest leverage-per-effort item in the plan
   and was previously buried inside participant sessions.
3. **Gesture set cut from 12 to 8** (see table below). Dropped: thumb–middle pinch (indistinguishable
   from thumb–index at landmark noise levels), swipe down, draw-a-circle, thumbs-up. Right click
   becomes two-finger pinch; undo moves to a keyboard fallback. Fewer, separable classes beat more,
   confusable ones — and gesture-set design is itself part of the ML work, so this gets written up.
4. **MLflow replaced by `results/<run>.json` + git commit hash**, rendered into the README by a
   script. A solo month-long project does not repay a tracking server; a results file that is
   actually kept current does.
5. **No system tray / settings GUI.** The preview window is the app: live overlay, on-screen state
   text, and a `config.yaml` for gesture→shortcut mapping. Saves most of Mon 16 for the user study.
6. **Feature set is x/y-only by default**, with `z` as an explicitly separate, ablated extra —
   because HaGRID's landmarks have no `z` (see CLAUDE.md, External data constraints).

## Overview

Replace the trackpad with webcam hand gestures: point to move the cursor, pinch to click, swipe and
draw shortcuts, recognised by models trained, evaluated and shipped like a production ML system.

"Virtual mouse with OpenCV and MediaPipe" is one of the most copied tutorials online. Those projects
hard-code thresholds and work only for their creator. This project wins on what they skip:

1. **Own multi-person dataset**, recorded with automatic labels, released as landmarks only.
2. **A ladder of models** per task: hand-written rules → from-scratch classical ML → compact nets.
3. **Streaming recognition**: spotting gestures in continuous video, measured by false triggers per
   hour, not just accuracy.
4. **Real-time deployment**: optimised models, a latency budget, a packaged app, a browser demo.
5. **A user study** against the trackpad, using standard HCI measures.

**What is ML here and what isn't.** Moving the cursor is signal processing (filtering and mapping
hand position), not learning. The ML is recognising poses and gestures, telling deliberate gestures
from ordinary hand movement, and adapting to new users. Saying this plainly shows judgement.

Assumptions: Python, one built-in webcam, no GPU needed for the core. If the course deadline falls
before 25 Nov, the course core is complete by 1 Nov.

## Gestures and system design

Eight built-in gestures plus user-taught custom ones. An open palm pauses and resumes control,
which addresses the core problem of gesture interfaces: every hand movement risks being read as a
command (the "Midas touch" problem).

| Gesture | Kind | Action |
| --- | --- | --- |
| Index finger pointing | Static pose + tracking | Move cursor |
| Open palm, held 0.5 s | Static pose | Pause / resume control |
| Thumb–index pinch, quick | Static pose | Left click |
| Thumb–index pinch, held | Static pose + tracking | Drag |
| Two fingers up, move up/down | Static pose + tracking | Scroll |
| Swipe left / right | Dynamic | Previous / next tab |
| Swipe up | Dynamic | Show all windows |
| Draw an X in the air | Dynamic | Close tab |
| Custom (5 examples) | Dynamic | Any shortcut the user picks |

Only the gesture path uses trained models; the cursor path is filtering and mapping, which keeps
pointing fast and predictable. See the diagram in CLAUDE.md.

## Models

Each recognition task gets three rungs, so every result answers "was the complex model worth it?"
The from-scratch rungs are the course submission.

| Task | Rung 1: rule baseline | Rung 2: from scratch (course) | Rung 3: compact neural net |
| --- | --- | --- | --- |
| Static poses (point, pinch, palm…) | Finger-angle thresholds | MLP with hand-written backprop in NumPy (Module II) | PyTorch MLP pretrained on HaGRID landmarks, head replaced, fine-tuned on ours |
| Dynamic gestures (swipes, X) | Velocity and direction thresholds | KNN with DTW distance in NumPy, lower-bound pruning (Module IV) | TCN or GRU in PyTorch |
| Custom gestures, 5 examples | — | KNN with DTW on the 5 examples | Nearest prototype in the TCN embedding |

**Gesture spotting** sits on top of the dynamic models: a sliding window over the live stream, an
explicit "no gesture" class, a rejection threshold on confidence, plus hysteresis and a cooldown so
one gesture fires once.

**Cursor control** is not learned: hand position is mapped from a comfortable region of the camera
frame to the screen, smoothed with a One Euro filter (low jitter when still, low lag when moving
fast) and an acceleration curve.

**Deployment:** export the neural models to ONNX, apply INT8 quantisation, report accuracy against
latency for every rung.

**Course mapping:** Module II (perceptron, multilayer networks, backpropagation) and Module IV
(k-nearest neighbour). Optionally Module IV's locally weighted regression as a cursor smoother
compared against the One Euro filter.

## Data plan

**Public data**

| Dataset | What it gives | How to use it |
| --- | --- | --- |
| [HaGRID](https://github.com/hukenovs/hagrid) | Static hand gesture images from tens of thousands of people; annotations include 21 `[x,y]` landmarks per hand and `user_id`. CC BY-SA 4.0. | Annotations only (not ~700GB of images). Pretrain the static-pose MLP body; replace the head (different label set; no `z`). |
| [IPN Hand](https://github.com/GibranBenitez/IPN-hand) | Videos of 50 people doing 13 touchless gestures continuously, including natural non-gesture movement. | Run MediaPipe on a subset for landmarks; **held out** as an external benchmark for spotting and false triggers. Never train on it. |

**Own recordings**

- **Prompted recorder:** an on-screen prompt says "Swipe left in 3, 2, 1", so every clip is labelled
  automatically. Save landmarks with timestamps; never raw video unless the person opts in.
- **People:** 10–12, varied by hand size, skin tone, left/right-handedness, seated distance.
- **Per person (~20 min):** 8 gestures × 10 repeats, in two lighting conditions (daylight, tube light).
- **Negative data:** 2–3 hours total of typing, talking with hands, drinking, touching the face,
  phone use in front of the camera. Collected passively from Mon 26 (amendment 2).
- **Continuous sessions:** 3–4 scripted runs with gestures back to back in a normal browsing task,
  with prompt timestamps, for event-level scoring.

**Rules that keep results honest:** split by person (leave-one-subject-out); signed consent,
anonymous IDs, a datasheet, landmarks-only release; dataset versioned and v1 frozen end of week 2,
so every result names the data it used.

## Evaluation

Targets are written down before any results are seen. Missing a target honestly is still a result.

| Question | Metric | How it's measured | Target |
| --- | --- | --- | --- |
| Does it recognise gestures? | Macro-F1, confusion matrix | Isolated clips, leave-one-subject-out | Decide in week 1 |
| Does it work on new people? | F1 unseen vs after 5-shot personalisation | LOSO, then 5 examples from the held-out person | Gap shrinks after 5 shots |
| Does it work on a live stream? | Event precision, recall, detection delay (ms) | Continuous scripted sessions | Decide in week 2 |
| Does it fire by accident? | False triggers per hour | Negative recordings + IPN Hand non-gesture segments | Under 1/hour |
| Is it fast enough? | End-to-end latency p50/p95 (ms), FPS, CPU % | Profiler on this laptop, per rung | p95 under 100 ms |
| Is it fair across people? | F1 per slice | Lighting, handedness, distance, skin tone (with consent) | No slice far below the rest |
| Is it usable? | Fitts's law throughput (bits/s), error rate, task time, SUS | User study, gesture vs trackpad | Report honestly |

**Expect the trackpad to win on speed.** Gesture input is not meant to beat a trackpad for an
able-bodied user at a desk. Its value is hands-free, touch-free use (cooking, presenting,
accessibility). Showing that trade-off with real numbers is more credible than claiming otherwise.

Every experiment is logged with its config, dataset version and git commit, so any number in the
report can be reproduced.

## Week 1 (26 Oct – 1 Nov): working pipeline, recorder, course core

Goal by Sunday: the system works end to end for you, both from-scratch models exist, **and outside
people are already recorded**.

- [x] **Mon 26:** Repo, venv, pytest, GitHub Actions. Webcam → MediaPipe Hand Landmarker at steady
      frame rate with a live landmark preview.
- [ ] **Mon 26 (cont.):** Consent form drafted. Passive negative-data logger started (amendment 2).
- [ ] **Tue 27:** Feature extraction (wrist-relative, palm-scaled, rotation-normalised; joint
      angles; velocities) as pure functions. Prompted recorder with automatic labels. Record
      yourself on the full set. Cursor mode with One Euro filter + pynput, **with keyboard kill
      switch and dry-run mode** (frozen contract 4).
- [ ] **Wed 28 – Fri 30:** Record every participant who is available now, ~2/day alongside other
      work. Rule baselines for poses and gestures. Evaluation harness skeleton: LOSO splits,
      macro-F1, confusion matrix. Write down the targets.
- [ ] **Sat 31:** MLP from scratch in NumPy: forward, backprop, gradient check against numerical
      gradients.
- [ ] **Sun 1:** KNN + DTW from scratch, checked against `dtaidistance`. Book remaining recording
      slots for 2–4 Nov. Week log: what worked, what broke, numbers so far.

If the course deadline is early, the from-scratch MLP and KNN + DTW with a short report on your own
data is a complete course submission by this Sunday.

## Week 2 (2 – 8 Nov): data at scale and the evaluation harness

Goal by Sunday: dataset v1 frozen, every baseline and from-scratch model scored on it.

- [ ] **Mon 2 – Wed 4:** Record remaining participants: gesture repeats, two lighting conditions, a
      negative session and a continuous scripted session each. Evenings: pull HaGRID annotations,
      convert to the feature format.
- [ ] **Thu 5:** Data quality pass — drop clips where the hand was lost, check labels against
      prompts, write the datasheet, freeze v1.
- [ ] **Fri 6:** Finish the harness: event-level spotting metrics, false triggers per hour, latency
      profiler. Process an IPN Hand subset into landmarks.
- [ ] **Sat 7:** Run rule baselines, from-scratch MLP and KNN + DTW on v1. First results table.
- [ ] **Sun 8:** Speed up DTW with lower-bound pruning and a warping window. Wire up
      `results/<run>.json` logging (amendment 4).

Recruitment is the biggest risk this week; fewer than 6 people by Wed 4 triggers the fallback.

## Week 3 (9 – 15 Nov): neural models, spotting and optimisation

Goal by Sunday: all three rungs compared, streaming recognition tuned, models frozen.

- [ ] **Mon 9:** PyTorch MLP pretrained on HaGRID landmarks (x/y only, head replaced), fine-tuned on
      ours. Compare against the from-scratch MLP and against training on our data alone.
- [ ] **Tue 10:** TCN (or GRU) for dynamic gestures, with landmark augmentation: rotation, scaling,
      noise, time stretching. Small enough for CPU; Kaggle's free GPU if needed.
- [ ] **Wed 11:** Gesture spotting: sliding window, rejection threshold, hysteresis, cooldown. Tune
      on validation people only; plot recall against false triggers/hour and pick the operating point.
- [ ] **Thu 12:** Few-shot custom gestures: KNN + DTW on 5 examples vs nearest prototype in the TCN
      embedding. Accuracy at 1, 3 and 5 shots.
- [ ] **Fri 13:** Export to ONNX, INT8 quantisation, latency and accuracy for every rung.
- [ ] **Sat 14:** Error analysis by slice. Fix the two biggest failure modes and re-measure.
- [ ] **Sun 15:** Freeze models v1; complete results tables; write the methods section while fresh.

## Week 4 (16 – 25 Nov): product, user study, write-up

Goal by Wed 25: an app anyone can install, a user study, and a public repo judgeable in two minutes.

- [ ] **Mon 16:** App: preview window as the app — live overlay, on-screen action confirmation,
      pause gesture, `config.yaml` gesture→shortcut map, "teach a new gesture" flow (amendment 5).
      Spend the saved time on the browser demo.
- [ ] **Tue 17:** Packaging (pip install or single executable), README quick start, tests and lint.
- [ ] **Wed 18 – Thu 19:** User study, 6–8 people, at least half not in the training data: Fitts's
      law pointing task (gesture vs trackpad), a real browsing task, SUS questionnaire.
- [ ] **Fri 20:** Analyse the study. Write the report: problem, data, models, results, limitations.
- [ ] **Sat 21:** 2-minute demo video, model card, datasheet; publish the consented landmark dataset.
- [ ] **Sun 22:** Resume bullet with real numbers; blog post or write-up linking repo and video.
- [ ] **Mon 23 – Wed 25:** Buffer. Rehearse explaining DTW, backprop, LOSO and the One Euro filter
      out loud.

## Tools and resources

| Need | Tool | Use |
| --- | --- | --- |
| Hand landmarks | [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python) | 21 landmarks per frame, live-stream mode |
| Static gesture data | [HaGRID](https://github.com/hukenovs/hagrid) · [paper](https://arxiv.org/abs/2206.08219) | Landmark annotations for pretraining |
| Continuous gesture data | [IPN Hand](https://github.com/GibranBenitez/IPN-hand) · [paper](https://arxiv.org/abs/2005.02134) | External benchmark for spotting / false triggers |
| Prior work to compare | [ai-forever dynamic gestures](https://github.com/ai-forever/dynamic_gestures) | Cite it; explain how ours differs |
| Cursor smoothing | [One Euro filter](https://gery.casiez.net/1euro/) | Low-jitter, low-lag position filtering |
| Clicks and keys | [pynput](https://pynput.readthedocs.io/) | Mouse and keyboard control |
| Checking the DTW | [dtaidistance](https://github.com/wannesm/dtaidistance) | Reference to test the from-scratch version |
| Neural models | [PyTorch tutorials](https://pytorch.org/tutorials/) | MLP and TCN/GRU training |
| Optimisation | [ONNX Runtime quantisation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) | INT8 models, CPU latency |
| Dataset release | [Hugging Face datasets](https://huggingface.co/docs/datasets) | Publish landmarks with a datasheet |

For the user study, read up on the Fitts's law pointing task (throughput in bits/s) and the System
Usability Scale (SUS).

## Suggestions, in priority order

1. **Browser demo with nothing to install.** MediaPipe for web + ONNX Runtime Web, so a recruiter
   clicks a link and tries it in 10 seconds. The single biggest multiplier on how many people
   actually try the project; amendment 5 frees time for it.
2. **Learn from corrections.** When a user undoes an action right after it fires, save that window
   as a "should not have fired" example and adjust that user's thresholds. Report false triggers
   before and after a few days of use — a closed feedback loop.
3. **Calibrated confidence.** Reliability diagram, temperature scaling, trigger on calibrated
   probability. Few student projects do this.
4. **Accessibility mode.** Dwell-to-click and stronger smoothing for unsteady hands, for people who
   cannot pinch reliably. Test with one person with limited hand mobility, with consent.
5. **Context-aware actions.** Detect the active app so a swipe means "next slide" in a presentation
   and "next tab" in a browser.
6. Locally weighted regression as a cursor smoother, compared with One Euro (ties in Module IV).
7. Two-hand gestures (zoom, rotate).
8. Idle power saving: drop frame rate when no hand is present; measure CPU and battery.
9. Push-to-click using estimated depth instead of a pinch.
10. A short paper-style report if the results are strong.

## Risks and what to cut

| Risk | Early warning | Fallback |
| --- | --- | --- |
| Too few participants | Fewer than 6 recorded by Wed 4 Nov | 20-minute slots from club volunteers; 6 is the minimum; lean on IPN Hand for unseen-person testing |
| Hand tracking drops in poor light | >~10% of frames with no hand detected | Record a controlled-light subset too; report light as a failure slice rather than hiding it |
| Gestures confused with each other | Two classes dominate the confusion matrix | Redesign or drop one gesture; gesture design is part of the ML work, so write it up |
| Too slow for real time | p95 latency above ~150 ms | Lower camera resolution, classify every second frame, use the INT8 model |
| Course deadline earlier than planned | — | Week 1 already ends with a complete course submission |
| Can't explain code Claude wrote | You can't sketch a module on paper | After each module, write a one-paragraph explanation in your own words in `docs/` |

**Cut in this order if behind:** two-hand gestures → GRU (keep the TCN) → quantisation →
custom-gesture prototype method (keep KNN + DTW) → user study down to 4 people.

**Never cut:** testing on unseen people, false triggers per hour, and the live demo. Those three are
what separate this from the tutorial version.

## Deliverables

- [ ] Public repo: README with GIF, quick start, results tables, architecture diagram
- [ ] Installable app, and a browser demo link
- [ ] 2-minute demo video
- [ ] Technical report (course submission + portfolio)
- [ ] Model card and dataset datasheet
- [ ] Consented landmark dataset on Hugging Face
- [ ] Every result reproducible from a commit

**Resume bullet** (replace each [X] with real numbers):

> Built a webcam hand-gesture system that replaces the trackpad (cursor, clicks, 8 shortcuts,
> 5-shot custom gestures). Recorded and released an [N]-person landmark dataset; compared
> from-scratch MLP and KNN-DTW against HaGRID-pretrained MLP and TCN models ([X] macro-F1 on unseen
> users, [X] false triggers/hour, [X] ms p95 latency on CPU after INT8 quantisation). User study vs
> trackpad with [N] participants (SUS [X]).
