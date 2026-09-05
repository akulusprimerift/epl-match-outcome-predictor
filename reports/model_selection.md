# Phase 8 Model Selection Freeze

Selected candidate: **model_b**.

Recorded at 2026-09-05T07:04:25.916556+00:00, before final holdout evaluation.

| Candidate | Test fixtures | Log loss | Macro F1 | Accuracy |
|---|---:|---:|---:|---:|
| model_a | 380 | 1.039491851757 | 0.348426348426 | 0.471052631579 |
| model_a_matched | 380 | 1.047621434138 | 0.330595955332 | 0.450000000000 |
| model_b | 380 | 1.031918963256 | 0.366447842342 | 0.471052631579 |

## Decision and limitations

Section 10.6 selects model_b using the lowest test log loss among candidates passing integrity and macro-F1 guardrails. The macro-F1 check uses the configured maximum 0.02 drop relative to the best candidate F1; Model B also must pass the Section 10.5 comparison against Model A-Matched. Model B has both the lowest log loss and highest macro F1 of all three candidates, so the choice does not depend on a borderline guardrail interpretation.

All three candidates were compared on the identical 380 fixtures from 2024/25. Model A used 4,940 training fixtures; the matched candidates each used 1,900. Model B adds only previous-season possession for each team and their difference. Promoted-team missing values use training-only medians. The improvement measures predictive association, not causation.

The improvement is modest and comes from one test season; it does not establish statistical significance or future performance. Logistic regression remains a useful benchmark (test log loss 1.033743, accuracy 0.489474), but the Section 10.6 candidate set is Model A, Model A-Matched, and Model B.

## Frozen record

The existing model is retained without refitting: `models/model_b_xgb.json`. Its zero-based best iteration is 267; inference uses iterations [0, 268).
`config/model_config.json` records the selected feature order, full preprocessing values, parameters, labels, prediction JSON schema, probability-sum tolerance, test decision, source/artifact checksums, and implementation checksum. The original top-level baseline feature list remains for earlier phase compatibility; the candidate's authoritative list is `frozen_candidate.feature_columns`.

Implementation hashes normalize CRLF to LF; artifact and raw-data hashes use exact bytes. The configuration checksum covers the entire config except its own `freeze_record_sha256` field. `git_commit` identifies the pre-freeze parent 0980258f828c63744ab7863eb3a032fdf67b3808; the freeze commit is the commit containing this record, avoiding a self-referential commit hash. Model JSON files retain the existing local/ignored policy; keep the saved artifacts to verify this exact freeze.

Verification: `python -m src.freeze_model --verify`. Full acceptance: `python -m unittest discover -s tests -v`.

The 2025/26 holdout remains unevaluated. Phase 9 requires explicit user approval after this freeze is committed. The prediction schema is frozen here; the upcoming-fixture command is still Phase 10 work.
