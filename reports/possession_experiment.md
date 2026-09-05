# Phase 7 Matched Possession Experiment

Model A-Matched and Model B use one frozen possession-eligible fixture cohort. This comparison measures predictive association on that cohort; it does not establish that possession causes match outcomes.

| Model | Split | Rows | Log loss | Macro F1 | Accuracy |
|---|---|---:|---:|---:|---:|
| model_a_matched | validation | 380 | 0.930106 | 0.426953 | 0.578947 |
| model_a_matched | test | 380 | 1.047621 | 0.330596 | 0.450000 |
| model_b | validation | 380 | 0.928749 | 0.470055 | 0.578947 |
| model_b | test | 380 | 1.031919 | 0.366448 | 0.471053 |

## Model B acceptance rule

- PASS — Test log loss is lower than Model A-Matched.
- PASS — Test macro F1 is no more than 0.02 below Model A-Matched.
- PASS — Frozen-row, feature-contract, class-order, and probability-normalization checks pass.

**Phase 7 result: Model B passes the declared incremental-value rule.**

This is an experiment result only. Production-candidate selection belongs to Phase 8, and the 2025/26 holdout remains unopened.

Per-class precision, recall, F1, support, and confusion-matrix counts are stored in `reports/model_results.csv`; model-specific confusion-matrix images are stored alongside this report.
