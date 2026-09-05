# Phase 9 — Final 2025/26 Holdout

Frozen candidate: **model_b**, evaluated once on **380 EPL fixtures**.

| Metric | 2024/25 test | 2025/26 final holdout | Holdout minus test |
|---|---:|---:|---:|
| log_loss | 1.031918963256 | 1.075037170847 | +0.043118207590 |
| macro_f1 | 0.366447842342 | 0.353287940919 | -0.013159901423 |
| accuracy | 0.471052631579 | 0.463157894737 | -0.007894736842 |

Primary-metric log loss worsened by 0.043118 (4.18% relative to test). Lower log loss is better; higher F1 and accuracy are better.

## Interpretation and limitations

This is a sequential pre-match backtest: rolling features use only earlier completed fixtures, including earlier matches in the holdout season. It is not a simultaneous preseason forecast. Possession is lagged from 2024/25, never taken from the final 2025/26 season averages. Missing previous-season possession for promoted clubs uses the saved training-only medians.

Only the previously selected model was evaluated. The result does not measure a holdout possession uplift against other candidates and does not establish statistical significance or future performance. Class-level metrics and the confusion matrix expose weaknesses that aggregate accuracy can hide. No model selection, parameter adjustment, feature change, median fitting, or retraining followed this evaluation. Any future modeling change needs a new future holdout season.

## Integrity and acceptance

- Original Phase 8 freeze: `37f4b67d7dbc1c391aa57ede2796834fec488eb2`.
- Committed pre-holdout evaluator: `829286c73919db7045670eb4613f14696e545e8f`.
- Unchanged freeze record: `f3b9ac19656dc1a8218176250f182406895b4fa0bcff0cb3a8432a18cf79a530`.
- Evaluation started at `2026-09-05T07:25:13.081143+00:00`.
- Saved best iteration: 267; no fit or early-stopping calls.
- Frozen artifact/source checksums and exact holdout fixture membership verified before inference.
- Three finite, nonnegative class probabilities per fixture; sums checked within 0.000001.
- Original report column contract and target encoding (away=0, draw=1, home=2) preserved.
- Model configuration and all original training/feature/metric implementations unchanged. The separate Phase 9 protocol permits only evaluation/verification adapters and evaluation tests.
- A one-time start record prevents repeat inference. Repeating the CLI verifies and returns saved results.
- The completion receipt hashes all final outputs. The original metadata's holdout_evaluated=false remains an immutable statement of its pre-holdout state; the completion receipt records the current state.

![Final holdout confusion matrix](confusion_matrix_final_holdout.png)

Metrics: `final_holdout_results.csv`. Per-fixture probabilities: `final_holdout_predictions.json`. Audit: `final_holdout_started.json` and `final_holdout_receipt.json`.

Phase 10 upcoming-fixture prediction is not implemented as part of this phase.
