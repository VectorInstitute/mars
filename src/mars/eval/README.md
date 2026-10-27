# eval

Scoring, metrics and result files shared by every model, so all numbers come from one
code path.

```
eval/
├── evaluator.py   # predict, score_battery, run_split: scoring for training and run_eval
├── metrics.py     # compute: accuracy, AP, AUROC, EER, P/R at 0.5; per-method tables; macro
├── summary.py     # summarise: per-clip predictions -> clip and video metrics, per video rows
├── thresholds.py  # source threshold (Youden J on source validation) and operating points
├── protocols.py   # both operating points per corpus; summary.md tables and log bullets
└── write.py       # results/<model>/<run>/<corpus>/{metrics.json, predictions.jsonl, ...}
```

Conventions:

- A video's score is the mean of its clip probabilities.
- Per-method (per-generator) metrics score each fake method against all reals of the
  corpus.
- The source threshold is chosen on FF++ validation videos only and applied unchanged
  to every test corpus.
