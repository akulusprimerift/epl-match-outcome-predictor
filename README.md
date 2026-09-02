# EPL Match Outcome Predictor

A reproducible Python project for predicting English Premier League match
outcomes as away-win, draw, and home-win probabilities using only information
available before kickoff.

The repository has completed **Phase 6: Possession Collection and Join** and has
the **Phase 7 matched-experiment workflow implemented**. It contains
immutable raw EPL match files, leakage-safe pre-match features, frozen
chronological splits, majority/logistic-regression benchmarks, and a tuned
long-history XGBoost model, plus a resumable API-Football possession enrichment
pipeline. No API response data is fabricated or committed. The current 0%
possession coverage does not support a matched cohort, so Model A-Matched and
Model B have not been trained. The final 2025/26 holdout has not been evaluated.

## Requirements

- Python 3.11 or later
- pandas
- scikit-learn
- XGBoost
- matplotlib

## Setup

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Exact versions used for the bootstrapped environment are recorded in
`requirements.lock.txt`.

## Download EPL history

Download every configured season from 2010/11 through 2025/26:

```bash
python -m src.download_data --all
```

Or download one configured season:

```bash
python -m src.download_data --season 2425
```

Valid cached files are not requested again. Source URLs, retrieval timestamps,
row counts, and SHA-256 checksums are stored in `data/raw/manifest.json`.
Football-Data CSVs are committed as binary files so Git cannot change their
downloaded bytes through line-ending normalization.

## Build the canonical match table

Validate and clean every immutable raw season into the fixed canonical schema:

```bash
python -m src.clean_data
```

The command writes `data/processed/canonical_matches.csv` atomically and reports
input rows, output rows, duplicate rows, missing shots, and unresolved teams.
Team names are resolved exactly through `config/team_name_map.csv`; unknown
provider names fail validation and are never matched fuzzily.

## Build team history and baseline features

Expand each canonical fixture into chronological home- and away-team history
rows:

```bash
python -m src.build_history
```

Then build the non-possession baseline feature table:

```bash
python -m src.build_features --feature-set baseline
```

Every rolling statistic uses the previous five EPL matches, requires at least
three observations, and excludes the current fixture. Same-date fixtures are
treated as contemporaneous. Phase 3 retains all canonical fixtures, including
cold starts, and records overall and venue-specific history counts. Missing
rolling values remain unfilled in `model_dataset.csv`; the reusable imputation
helpers are designed to fit medians on a future training split only.

## Train the baseline models

Build the frozen season splits and train the majority and logistic-regression
benchmarks:

```bash
python -m src.train_baselines --feature-set baseline
```

The split policy is fixed in `config/model_config.json`: 2010/11–2022/23 for
training, 2023/24 for validation, 2024/25 for testing, and 2025/26 as the final
holdout. Median imputation is fitted on training rows only. The command writes
the reproducible row assignments to `data/processed/split_manifest.csv`, the
validation/test comparison to `reports/model_results.csv`, and local imputation
state to `models/preprocessing.json`. It does not calculate or display holdout
metrics.

## Train and evaluate Model A

Run the bounded, validation-only XGBoost search and evaluate the selected model
on the 2024/25 test split:

```bash
python -m src.train_xgboost --model-name model_a --feature-set baseline
```

Reload the saved native XGBoost artifact and reproduce its test metrics without
refitting:

```bash
python -m src.evaluate --model-name model_a --split test
```

The seven-candidate search uses only the approved parameter ranges in the
specification. Model A achieved test log loss `1.03949`, compared with `1.08222`
for the majority baseline and `1.03374` for logistic regression. It therefore
passes the required majority benchmark but does not outperform logistic
regression on test log loss. Detailed class metrics and all attempted settings
are saved in `reports/model_results.csv` and `reports/tuning_results.csv`.
Generated feature-importance values describe model gain and must not be read as
causal effects. Model JSON and preprocessing metadata remain local generated
artifacts under the repository's existing ignore policy.

## Collect and join possession

Set `API_FOOTBALL_KEY` in the local environment, then collect one configured
EPL season within a strict request budget:

```bash
# PowerShell: $env:API_FOOTBALL_KEY = "your-local-key"
# macOS/Linux: export API_FOOTBALL_KEY="your-local-key"
python -m src.collect_possession --season 2025 --max-requests 90
```

The collector requests only API-Football league `39` fixtures with status `FT`,
caches the exact fixture and statistics responses, records their checksums and
endpoints in `data/raw/manifest.json`, and skips verified caches on rerun. If the
request budget is reached, rerun the same command to resume. The key is sent only
as a request header and is never written to the manifest or command output.

Rebuild the canonical table and join every available cache by exact season,
date, home team, and away team:

```bash
python -m src.clean_data --include-possession
```

The rebuild writes `reports/possession_unmatched.csv`,
`reports/possession_ambiguous.csv`, and `reports/possession_coverage.csv`.
Coverage is reported by season and team against the configured 95% threshold.
With no local API caches, the report correctly declares no valid Model B period;
missing possession remains missing rather than becoming zero.

## Run the matched possession experiment

After possession has reached the configured 95% season threshold, train the two
models on one frozen match-ID cohort:

```bash
python -m src.train_xgboost --model-name model_a_matched --feature-set baseline_matched
python -m src.train_xgboost --model-name model_b --feature-set possession
python -m src.compare_models --models model_a_matched model_b
```

Model A-Matched uses only the baseline columns. Model B uses those exact rows
and columns plus `home_possession_avg_5`, `away_possession_avg_5`, and
`possession_edge`. Both variants independently use the approved validation-only
search, while test metrics are reserved for their final comparison. The command
refuses to train when train, validation, test, or holdout coverage is empty and
never evaluates holdout metrics. When both models exist, the comparison writes
`reports/possession_experiment.md` and applies the declared log-loss, macro-F1,
integrity, and probability checks without selecting a production candidate.

## Validate the project

With the virtual environment active:

```bash
python --version
python -c "import pandas, sklearn, xgboost, matplotlib"
python -m unittest discover -s tests -v
git status --short
```

See `EPL_MATCH_OUTCOME_PREDICTOR_SPEC.md` for the authoritative phased
implementation plan. Later phases must not be started without explicit
approval.
