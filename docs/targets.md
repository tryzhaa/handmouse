# Targets (written before any results)

Status: **PROPOSED, not yet confirmed by Trish.** Confirm or edit, then commit before running the
first real evaluation. Changing a target after seeing results defeats the point of this file.

| Question | Metric | Target | Source |
| --- | --- | --- | --- |
| Recognises gestures | Macro-F1, clip level, LOSO | >= 0.85 for the best rung; rule baseline expected well below | proposed |
| Works on new people | F1 unseen vs after 5-shot personalisation | **Not measured in v1 (solo data)**; v2 | plan.md amendment 7 |
| Works across conditions | Macro-F1, leave-one-condition-out (lighting, hand) | Each held-out condition within 0.10 of the overall figure | proposed |
| Works on a live stream | Event F1, detection delay | >= 0.80 event F1, median delay <= 500 ms | proposed |
| Fires by accident | False triggers per hour | < 1 / hour | plan.md |
| Fast enough | End-to-end latency | p95 < 100 ms on this laptop | plan.md |
| Fair across people | Per-slice macro-F1 | No slice more than 0.10 below the overall figure | proposed |
| Usable | Fitts's law throughput, SUS | Report honestly; trackpad expected to win on speed | plan.md |

Every accuracy number is reported next to its false-triggers-per-hour figure (CLAUDE.md contract 5).
`results/*.json` carries `false_triggers_per_hour: null` until the streaming harness measures it.
