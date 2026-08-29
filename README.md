# EPL Match Outcome Predictor

A reproducible Python project for predicting English Premier League match
outcomes as away-win, draw, and home-win probabilities using only information
available before kickoff.

The repository has completed **Phase 1: Football-Data Ingestion**. It contains
the project structure, dependency setup, season configuration, immutable raw
EPL match files, and their checksum manifest. Data cleaning, feature
engineering, and model training have not been implemented.

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
