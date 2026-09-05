# EPL Match Outcome Predictor

A reproducible Python project for predicting English Premier League match
outcomes as away-win, draw, and home-win probabilities using only information
available before kickoff.

The repository has completed Phases 0 through 5 and contains a revised
**Phase 6 SofaScore team-season possession collector**. It contains
immutable raw EPL match files, leakage-safe pre-match features, frozen
chronological splits, majority/logistic-regression benchmarks, and a tuned
long-history XGBoost model. Model A-Matched and Model B have not been trained,
and the final 2025/26 holdout has not been evaluated. The existing Phase 7
implementation still represents the old match-level design and must be revised
before it is run against the new team-season table.

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

## Collect team-season possession averages

No key or account is required. Collect one SofaScore EPL season:

```bash
python -m src.collect_possession --season 2024 --max-requests 25
```

Or collect the default 2017/18 through 2025/26 range with a strict request
budget:

```bash
python -m src.collect_possession --all --max-requests 250
```

The collector discovers SofaScore season IDs, gets the teams from each season's
standings, and retrieves one `averageBallPossession` value per team. Exact JSON
responses are cached under `data/raw/sofascore/`, checksummed in
`data/raw/manifest.json`, and skipped on rerun. Requests are throttled and use
bounded retries. Rerunning the same command safely resumes a budget-limited run.

The deterministic output is `data/processed/team_season_possession.csv`, with
coverage recorded in `reports/possession_coverage.csv`. Each source-season row
also names its following target season. A 2023/24 team average may therefore be
used for 2024/25 fixtures, but never for fixtures within 2023/24 itself. This
one-season lag is mandatory to prevent future leakage.

## Run the matched possession experiment

Phase 7 must first be revised to join the new lagged team-season table. After
that revision and sufficient 95% source-season team coverage, the intended
commands remain:

```bash
python -m src.train_xgboost --model-name model_a_matched --feature-set baseline_matched
python -m src.train_xgboost --model-name model_b --feature-set possession
python -m src.compare_models --models model_a_matched model_b
```

Model A-Matched will use only the baseline columns. Model B will use those exact
rows and columns plus `home_previous_season_possession`,
`away_previous_season_possession`, and `possession_edge`. Both variants must use
identical match IDs, and neither may use a final possession average from the
fixture's own season.

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
