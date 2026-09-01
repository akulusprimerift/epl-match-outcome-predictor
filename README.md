# EPL Match Outcome Predictor

A reproducible Python project for predicting English Premier League match
outcomes as away-win, draw, and home-win probabilities using only information
available before kickoff.

The repository has completed **Phase 3: Team History and Leakage-Safe
Features**. It contains immutable raw EPL match files, their checksum manifest,
explicit team-name mappings, a validated one-row-per-fixture canonical dataset,
two team-perspective history rows per fixture, and a baseline pre-match feature
table. Train/test splitting and model training have not been implemented.

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
