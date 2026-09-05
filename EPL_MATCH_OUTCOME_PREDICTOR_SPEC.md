# EPL Match Outcome Predictor — Engineering Specification

**Document status:** Approved implementation plan  
**Version:** 1.0  
**Last updated:** 2026-08-29  
**Owner:** Akshaj Sinha  
**Primary use:** Source-of-truth specification for Codex, VS Code, and project contributors

---

## 1. How Codex Must Use This Document

This specification is the authoritative description of the project. Codex must read this file before proposing or making project changes.

### Execution rules

1. Work on **one numbered phase at a time**.
2. Do not begin a later phase unless the user explicitly requests it.
3. Before editing, inspect the repository and report whether the requested phase is unstarted, partially complete, or complete.
4. Preserve completed phase behavior unless the current phase requires a change.
5. Do not silently change data sources, label encoding, feature definitions, season splits, model-selection rules, or output schemas.
6. Never use future-match information in a pre-match feature.
7. Never modify files under `data/raw/` after they are downloaded. Corrections belong in the cleaning pipeline.
8. Never commit API keys, authentication files, `.env`, cached credentials, or private tokens.
9. Use only English Premier League data. Do not use other leagues for training, priors, imputation, ratings, comparisons, or calibration.
10. At the end of each phase:
    - run its required validation commands;
    - summarize changed files;
    - report acceptance criteria as pass/fail;
    - stop and wait for approval before starting the next phase.



## 2. Product Definition

### 2.1 Problem statement

Build a reproducible machine-learning pipeline that predicts the outcome of an upcoming English Premier League fixture from the home team's perspective:

- `0` — away win
- `1` — draw
- `2` — home win

The final model must return three probabilities:

```text
P(away win), P(draw), P(home win)
```

The probabilities must be based only on information available before kickoff.

### 2.2 Primary user story

> Given a scheduled EPL home team, away team, and match date, generate pre-match rolling features from completed historical fixtures and return home-win, draw, and away-win probabilities.

### 2.3 Project goals

- Build a complete data-to-prediction workflow.
- Use free, reproducible, legally accessible data sources.
- Demonstrate robust pandas data engineering.
- Prevent temporal leakage by design and through automated tests.
- Train and evaluate a multiclass XGBoost classifier.
- Compare a long-history baseline model with a possession-enriched model.
- Produce inspectable reports with matplotlib.
- Save a reusable model and provide a deterministic prediction entry point.
- Present strong software-engineering and data-science evidence for SWE/data internship applications.

### 2.4 Non-goals for version 1

- Predicting exact scores.
- Betting recommendations or expected-value calculations.
- Using bookmaker odds as model features.
- Supporting leagues other than the EPL.
- Player-level modeling, injuries, lineups, xG, xA, or event-sequence models.
- A web application, API, database, or live production deployment.
- Collecting match-level possession when a leakage-safe team-season average is sufficient.
- Opening or optimizing against the final 2025/26 holdout before the pipeline is frozen.

These may be considered only after all phases in this document are complete.

---

## 3. Fixed Technical Decisions

### 3.1 Required stack

- Python 3.11 or later
- pandas
- scikit-learn
- xgboost
- matplotlib
- Python standard library for downloading, file hashing, JSON, CLI parsing, and tests

No additional runtime dependency should be introduced unless the user approves it.

### 3.2 Reproducibility rules

- Global random seed: `42`.
- Sort all match data chronologically before feature generation.
- Store raw inputs separately from processed outputs.
- Cache every external API response.
- Record source URL, retrieval time, row count, and SHA-256 checksum.
- Write processed tables with explicit column ordering.
- Save the trained XGBoost model with `save_model()`.
- Save metrics and configuration used for each model run.

### 3.3 Model architecture

Three model variants will exist:

1. **Model A — Long baseline**
   - Trained with the complete Football-Data history.
   - Features: goals, goals conceded, shots, recent form, venue edge, history counts.

2. **Model A-Matched — Coverage-matched baseline**
   - Uses the same rows, seasons, and fixtures available to Model B.
   - Excludes possession features.
   - Exists to isolate the incremental value of possession.

3. **Model B — Possession-enriched**
   - Uses the exact same training and evaluation rows as Model A-Matched.
   - Adds home possession, away possession, and possession-edge features.

The value of possession is measured by comparing **Model B vs Model A-Matched**, not only Model B vs the long-history Model A.

---

## 4. Data Sources and Contracts

### 4.1 Source A: Football-Data.co.uk

**Purpose:** Primary historical EPL match dataset.  
**League code:** `E0`  
**Season range for version 1:** 2010/11 through 2025/26.  
**URL pattern:**

```text
https://www.football-data.co.uk/mmz4281/{SEASON_CODE}/E0.csv
```

Examples:

```text
2010/11 -> 1011
2024/25 -> 2425
2025/26 -> 2526
```

Required source columns:

| Source column | Meaning | Required |
|---|---|---:|
| `Div` | League division | Yes |
| `Date` | Match date | Yes |
| `Time` | Kickoff time | No |
| `HomeTeam` | Home team name | Yes |
| `AwayTeam` | Away team name | Yes |
| `FTHG` | Full-time home goals | Yes |
| `FTAG` | Full-time away goals | Yes |
| `FTR` | Full-time result: H/D/A | Yes |
| `HS` | Home shots | Yes when available |
| `AS` | Away shots | Yes when available |
| `HST` | Home shots on target | Optional version-1 diagnostic |
| `AST` | Away shots on target | Optional version-1 diagnostic |

Football-Data betting-odds columns may remain in raw files but must not enter the version-1 feature table.

### 4.2 Source B: SofaScore team-season statistics

**Purpose:** Possession-style enrichment using one EPL season average per team.
**EPL unique-tournament ID:** `17`.
**Authentication:** None.
**Reliability note:** These are public JSON endpoints used by SofaScore's site, not a
versioned public API contract. Cache exact responses and fail clearly if the schema
changes. When automated JSON access is blocked, a manifested, checksummed export
of the same team-season statistics displayed by SofaScore may be used; it must
retain the season ID, team ID, match count, and source URL for every row.

Required endpoints:

```text
GET /unique-tournament/17/seasons
GET /unique-tournament/17/season/{SEASON_ID}/standings/total
GET /team/{TEAM_ID}/unique-tournament/17/season/{SEASON_ID}/statistics/overall
```

Required extracted values:

- Source season and SofaScore season ID
- Team ID and exact provider team name
- `statistics.averageBallPossession`
- `statistics.matches`
- Source URL

Possession parsing and timing rules:

- Parse `averageBallPossession` as a floating-point percentage from 0 through 100.
- Preserve `null` as missing; never silently convert it to zero.
- Cache every original JSON response before deterministic extraction.
- Skip a request when a manifested cache has the expected checksum.
- Throttle requests, retry transient failures, and stop before a configured request budget.
- A fixture in season `N` may use only the team average from completed season `N-1`.
- Never attach a final season average to fixtures within that same season.

### 4.3 Source C: StatsBomb Open Data

StatsBomb Open Data is explicitly excluded from the version-1 training pipeline because its open EPL seasons are not continuous enough for the main model. It may be used in a future event-data research phase only.

### 4.4 Raw-data immutability

The following directories are append-only:

```text
data/raw/football_data/
data/raw/sofascore/
```

Downloaded files must never be manually corrected. Every normalization must occur in `src/clean_data.py` or a later deterministic transformation.

### 4.5 Data manifest

Create `data/raw/manifest.json` with one record per retrieved file or response:

```json
{
  "source": "football-data",
  "season": "2425",
  "source_url": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
  "local_path": "data/raw/football_data/E0_2425.csv",
  "retrieved_at_utc": "ISO-8601 timestamp",
  "sha256": "hexadecimal checksum",
  "row_count": 380
}
```

SofaScore records additionally include `endpoint`, `sofascore_season_id`, and,
for team-statistics responses, `sofascore_team_id` and `sofascore_team_name`.

---

## 5. Canonical Data Model

### 5.1 Canonical match identifier

Create a deterministic `match_id` after team-name normalization:

```text
{season}|{YYYY-MM-DD}|{home_team_slug}|{away_team_slug}
```

Example:

```text
2425|2024-08-17|arsenal|wolverhampton-wanderers
```

Requirements:

- Unique within the canonical match table.
- Stable across repeated pipeline runs.
- Not dependent on DataFrame row order.
- Used as the stable Football-Data fixture identity. The SofaScore source joins
  separately at team-season level through canonical team slugs.

### 5.2 Team-name mapping

Create `config/team_name_map.csv`:

```csv
provider,provider_team_name,canonical_team_name,canonical_team_slug
football_data,Man United,Manchester United,manchester-united
sofascore,Manchester United,Manchester United,manchester-united
```

Rules:

- All source names must be resolved before feature generation.
- Unknown names cause a clear validation error.
- Do not use fuzzy matching automatically in the production pipeline.
- Fuzzy matching may generate suggestions for manual review but may not select a mapping silently.

### 5.3 Canonical match table

Output: `data/processed/canonical_matches.csv`

Required columns, in order:

```text
match_id
season
date
kickoff_time
home_team
away_team
home_team_slug
away_team_slug
home_goals
away_goals
result_code
home_shots
away_shots
home_possession
away_possession
football_data_source_file
api_fixture_id
```

`home_possession`, `away_possession`, and `api_fixture_id` may be missing before the enrichment phase.

The nullable per-fixture possession fields remain in the canonical schema for
backward compatibility but are not populated by the SofaScore team-season source.
Team-season averages are stored separately in
`data/processed/team_season_possession.csv`.

### 5.4 Team-season possession table

Output: `data/processed/team_season_possession.csv`

Required columns, in order:

```text
source_season
target_season
source_season_start_year
sofascore_season_id
team
team_slug
sofascore_team_name
sofascore_team_id
average_possession_pct
matches_recorded
source_url
```

`target_season` is always the season immediately following `source_season`.
This makes the leakage-safe lag explicit in the data contract.

### 5.5 Team-match history table

Output: `data/processed/team_match_history.csv`

Each completed fixture produces exactly two rows: one for each team.

Required columns:

```text
team_match_id
match_id
season
date
team
team_slug
opponent
opponent_slug
is_home
goals_for
goals_against
shots
possession
points
```

Derived values:

```text
points = 3 if goals_for > goals_against
points = 1 if goals_for == goals_against
points = 0 otherwise
```

### 5.6 Final model table

Output: `data/processed/model_dataset.csv`

Each fixture produces exactly one row from the home team's perspective.

Required identity and label columns:

```text
match_id
season
date
home_team
away_team
target
```

Target encoding:

```text
0 = away win
1 = draw
2 = home win
```

The current match's goals, shots, possession, points, or result must never appear as model inputs.

---

## 6. Feature Specification

### 6.1 Rolling window

- Default window: previous 5 completed EPL matches for that team.
- Minimum observations: 3.
- Current match exclusion: mandatory `shift(1)` before `rolling(...)`.
- Windows may carry across EPL seasons.
- Only rows with `date < current_match_date` may contribute.
- When multiple EPL matches occur on the same date, source ordering must not permit one match to influence another match on that date. Grouped feature calculations must treat same-date matches as contemporaneous unless kickoff timestamps establish a safe order.

### 6.2 Team-level rolling features

For both home and away teams:

| Feature | Definition |
|---|---|
| `goals_for_avg_5` | Mean goals scored over previous five matches |
| `goals_against_avg_5` | Mean goals conceded over previous five matches |
| `shots_avg_5` | Mean shots over previous five matches |
| `previous_season_possession` | Team's average possession from the immediately preceding completed EPL season |
| `form_points_5` | Sum of league points over previous five matches |
| `overall_ppg_5` | Mean points over previous five matches |
| `venue_ppg_5` | Mean points from previous five matches at the same home/away venue role |
| `history_matches` | Count of prior EPL matches available before the fixture |
| `venue_history_matches` | Count of prior same-venue-role EPL matches available |

Final model columns use prefixes:

```text
home_goals_for_avg_5
away_goals_for_avg_5
home_goals_against_avg_5
away_goals_against_avg_5
...
```

### 6.3 Directional edge features

Positive edge values must consistently favor the home team.

```text
goals_scored_edge = home_goals_for_avg_5 - away_goals_for_avg_5
defensive_edge = away_goals_against_avg_5 - home_goals_against_avg_5
shots_edge = home_shots_avg_5 - away_shots_avg_5
possession_edge = home_previous_season_possession - away_previous_season_possession
form_edge = home_form_points_5 - away_form_points_5
venue_edge = home_venue_ppg_5 - away_venue_ppg_5
history_edge = home_history_matches - away_history_matches
```

### 6.4 Home advantage

Do not create a constant `home_advantage = 1` feature. Every row is already home-oriented, so that value has no variance.

Version 1 represents home advantage through:

- Separate home and away feature columns.
- `venue_edge`.
- Optional league-level rolling home-win rate calculated only from earlier EPL fixtures.

If the league-level rate is implemented:

```text
league_home_win_rate_380 = mean(home_win) over the previous 380 completed EPL fixtures
```

It must use `shift(1)` and must not include the current match.

### 6.5 Cold-start policy

Because other leagues are prohibited, newly promoted teams may have little or no recent EPL history.

Policy:

1. Use available prior EPL history if the club previously played in the EPL.
2. Do not import Championship statistics.
3. For missing rolling values, impute the EPL training-set median for that feature.
4. Add `history_matches` and `venue_history_matches` so the model can distinguish an imputed cold start from a mature history.
5. Fit imputation values on the training set only.
6. Save imputation values to `models/preprocessing.json`.

### 6.6 Possession coverage policy

- Model A does not use possession.
- Model B uses each club's immediately preceding completed EPL season average.
- Generate source-season and team coverage rows.
- Declare the first target season whose preceding source season has acceptable coverage.
- Default minimum acceptable coverage: 95% of EPL teams in the source season.
- A promoted club without a preceding EPL average is missing and follows training-only median imputation; no Championship data may be substituted.
- Model A-Matched and Model B must use identical row IDs in training, validation, and testing.

---

## 7. Leakage and Integrity Requirements

The pipeline is invalid if any requirement below fails.

### 7.1 Temporal leakage invariants

For feature row `r` representing match `m`:

```text
max(source_match_date used by r) < m.date
```

When exact kickoff timestamps are reliable:

```text
max(source_kickoff_timestamp used by r) < m.kickoff_timestamp
```

### 7.2 Required automated tests

Create standard-library `unittest` tests under `tests/`:

1. **Current-result mutation test**
   - Change the current match's goals and result.
   - Rebuild its features.
   - Assert its pre-match features are unchanged.

2. **Future-result mutation test**
   - Change a future match.
   - Assert all earlier feature rows are unchanged.

3. **Window membership test**
   - For a known team and fixture, verify the exact previous match IDs used.

4. **One-row-per-fixture test**
   - Assert `model_dataset.match_id` is unique.

5. **Two-history-rows-per-fixture test**
   - Assert every canonical match produces exactly two team-history rows.

6. **Target mapping test**
   - Verify A/D/H maps to 0/1/2.

7. **Probability normalization test**
   - Assert predicted probabilities are finite, nonnegative, and sum to 1 within `1e-6`.

8. **Chronological split test**
   - Assert training dates precede validation, validation precedes test, and test precedes final holdout.

9. **Matched-model row test**
   - Assert Model A-Matched and Model B use identical `match_id` sets for every split.

10. **Unknown-team mapping test**
    - Assert an unmapped provider name raises a clear error.

11. **Duplicate-match test**
    - Assert canonical `match_id` values are unique.

12. **Raw immutability test**
    - Compare raw file checksums against the data manifest.

Required test command:

```bash
python -m unittest discover -s tests -v
```

---

## 8. Season Splits and Holdout Policy

Version-1 frozen splits:

| Split | Seasons | Purpose |
|---|---|---|
| Train | 2010/11–2022/23 | Fit preprocessing and models |
| Validation | 2023/24 | Early stopping and approved tuning |
| Test | 2024/25 | Compare baselines and model variants |
| Final holdout | 2025/26 | One final untouched backtest |

Rules:

- Do not use random `train_test_split`.
- Do not shuffle before splitting.
- Fit all imputation statistics on training rows only.
- Use validation data for early stopping and limited hyperparameter selection.
- Use test data for the final Model A / A-Matched / B comparison.
- Do not inspect holdout metrics until all features, preprocessing, and model settings are frozen in `config/model_config.json`.
- After the holdout is opened, no changes may be justified by improved holdout performance. Any later change requires a new future holdout season.

---

## 9. Modeling Specification

### 9.1 Majority baseline

Predict the most frequent training label for every row.

Purpose:

- Establish the minimum accuracy and log-loss benchmark.
- Detect a sophisticated model that fails to improve over class frequency.

### 9.2 Logistic-regression baseline

Use scikit-learn multinomial logistic regression with the exact feature table used by XGBoost.

Requirements:

- Apply training-only median imputation.
- Apply standardization through a scikit-learn pipeline.
- Use `random_state=42` where supported.
- Save baseline metrics to the same comparison report as XGBoost.

### 9.3 XGBoost starting configuration

```python
XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=500,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=3,
    subsample=0.80,
    colsample_bytree=0.80,
    reg_lambda=2.0,
    eval_metric="mlogloss",
    early_stopping_rounds=40,
    random_state=42,
)
```

### 9.4 Hyperparameter policy

Version 1 permits a small, documented search over:

```text
max_depth: [2, 3, 4]
learning_rate: [0.02, 0.03, 0.05]
min_child_weight: [1, 3, 5]
subsample: [0.75, 0.90]
colsample_bytree: [0.75, 0.90]
reg_lambda: [1.0, 2.0, 5.0]
```

Constraints:

- Optimize validation multiclass log loss.
- Do not use the test or holdout sets for tuning.
- Limit total search size; exhaustive search is not required.
- Save every attempted configuration and metric to `reports/tuning_results.csv`.
- Prefer the simpler model when validation metrics are effectively tied.

### 9.5 Output class order

Never assume `predict_proba` column order without reading `model.classes_`.

Required mapping:

```python
probability_by_class = dict(zip(model.classes_, model.predict_proba(X)[0]))
result = {
    "away_win": probability_by_class[0],
    "draw": probability_by_class[1],
    "home_win": probability_by_class[2],
}
```

---

## 10. Evaluation and Model Selection

### 10.1 Primary metric

**Multiclass log loss** — lower is better.

This is primary because the product returns probabilities, not only labels.

### 10.2 Guardrail metric

**Macro F1** — higher is better.

Draws are commonly underpredicted, so macro F1 prevents overall accuracy from hiding weak class-specific performance.

### 10.3 Secondary metrics

- Accuracy
- Per-class precision
- Per-class recall
- Per-class F1
- Confusion matrix
- Class distribution
- Best iteration selected by early stopping

### 10.4 Required reports

```text
reports/model_results.csv
reports/class_distribution.png
reports/confusion_matrix_model_a.png
reports/confusion_matrix_model_a_matched.png
reports/confusion_matrix_model_b.png
reports/feature_importance_model_a.png
reports/feature_importance_model_b.png
reports/possession_coverage.csv
reports/tuning_results.csv
reports/final_holdout_results.csv
```

### 10.5 Model B acceptance rule

Model B demonstrates useful incremental possession value only if:

1. Its test log loss is lower than Model A-Matched on identical test fixtures.
2. Its macro F1 is not more than `0.02` below Model A-Matched.
3. It passes every integrity and probability-normalization test.

If Model B fails this rule, possession remains an experiment and Model A stays the preferred model.

### 10.6 Production-candidate selection

Compare:

- Model A long-history performance.
- Model A-Matched performance.
- Model B performance.

The preferred final model is the candidate with the strongest test log loss, subject to macro-F1 and integrity guardrails. The decision and reasoning must be written to `reports/model_selection.md` before the final holdout is opened.

---

## 11. Repository Structure

```text
epl-match-outcome-predictor/
├── .gitignore
├── .env.example
├── AGENTS.md
├── EPL_MATCH_OUTCOME_PREDICTOR_SPEC.md
├── README.md
├── requirements.txt
├── requirements.lock.txt
├── config/
│   ├── model_config.json
│   ├── seasons.json
│   └── team_name_map.csv
├── data/
│   ├── raw/
│   │   ├── manifest.json
│   │   ├── football_data/
│   │   └── sofascore/
│   └── processed/
│       ├── canonical_matches.csv
│       ├── team_match_history.csv
│       ├── model_dataset.csv
│       └── team_season_possession.csv
├── models/
│   ├── model_a_xgb.json
│   ├── model_a_matched_xgb.json
│   ├── model_b_xgb.json
│   ├── preprocessing.json
│   └── model_metadata.json
├── reports/
│   ├── model_results.csv
│   ├── model_selection.md
│   ├── possession_coverage.csv
│   ├── tuning_results.csv
│   └── *.png
├── src/
│   ├── __init__.py
│   ├── constants.py
│   ├── download_data.py
│   ├── collect_possession.py
│   ├── clean_data.py
│   ├── build_history.py
│   ├── build_features.py
│   ├── split_data.py
│   ├── train_baselines.py
│   ├── train_xgboost.py
│   ├── evaluate.py
│   ├── compare_models.py
│   └── predict.py
└── tests/
    ├── __init__.py
    ├── fixtures/
    ├── test_data_contracts.py
    ├── test_features.py
    ├── test_leakage.py
    ├── test_splits.py
    ├── test_model_outputs.py
    └── test_possession_join.py
```

### 11.1 `.gitignore` minimum

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.DS_Store
data/raw/sofascore/*.json
models/*.json
```

Raw Football-Data CSV policy may be decided before the first commit. If raw data is not committed, keep manifests and reproducible download scripts committed.

### 11.2 `AGENTS.md` minimum

```markdown
# Project instructions

Read `EPL_MATCH_OUTCOME_PREDICTOR_SPEC.md` completely before changing code.
Treat it as the source of truth. Work on one requested phase at a time.
Never change fixed data contracts, season splits, target encoding, or
model-selection rules without explicit user approval. Run the phase's required
tests and report acceptance criteria before proceeding.
```

---

## 12. Phased Implementation Plan

## Phase 0 — Repository Bootstrap

### Objective

Create a clean, reproducible Python project skeleton without downloading or modeling data.

### Tasks

- Initialize Git if the folder is not already a repository.
- Create the repository tree from Section 11.
- Create `.gitignore`, `.env.example`, `requirements.txt`, `AGENTS.md`, and `README.md`.
- Copy this specification into the repository root.
- Create a virtual environment.
- Install the four required libraries.
- Capture exact installed versions in `requirements.lock.txt`.
- Add `src/__init__.py`, `tests/__init__.py`, and placeholder modules with module docstrings only.
- Add a basic test confirming imports and required directories.

### Required files

```text
.gitignore
.env.example
AGENTS.md
README.md
requirements.txt
requirements.lock.txt
src/__init__.py
tests/test_project_structure.py
```

### Validation

```bash
python --version
python -c "import pandas, sklearn, xgboost, matplotlib"
python -m unittest discover -s tests -v
git status --short
```

### Acceptance criteria

- Required imports succeed.
- Secrets and virtual-environment files are ignored.
- Project structure test passes.
- No match data has been downloaded.
- No model code has been implemented.

### Stop condition

Stop after reporting Phase 0 results. Do not start Phase 1.

---

## Phase 1 — Football-Data Ingestion

### Objective

Download and preserve EPL season CSVs from 2010/11 through 2025/26.

### Tasks

- Implement `config/seasons.json` with season labels and codes.
- Implement `src/download_data.py` with a CLI.
- Download each `E0.csv` to a season-specific filename.
- Use a temporary file and atomic rename so interrupted downloads do not create valid-looking partial files.
- Calculate SHA-256 and row count.
- Append or update `data/raw/manifest.json` deterministically.
- Skip a valid file whose checksum already matches the manifest.
- Fail clearly on HTTP, decoding, empty-file, or missing-column errors.
- Support `--season` and `--all` modes.

### CLI contract

```bash
python -m src.download_data --all
python -m src.download_data --season 2425
```

### Acceptance criteria

- Every configured season has one raw file.
- Files are nonempty and readable by pandas.
- `Div` contains only `E0` for required rows.
- Required columns exist.
- Manifest checksums match local files.
- Re-running the downloader performs no unnecessary downloads.
- Raw files remain byte-identical after a repeated run.

### Required tests

- Season URL generation.
- Manifest checksum verification.
- Missing-column failure.
- Empty-download failure using a local test fixture.
- Idempotent rerun behavior.

### Stop condition

Do not clean or concatenate data in this phase.

---

## Phase 2 — Cleaning and Canonical Match Table

### Objective

Transform immutable raw CSVs into one validated canonical EPL match table.

### Tasks

- Implement `config/team_name_map.csv`.
- Parse dates with explicit day-first handling.
- Parse optional kickoff times without fabricating missing values.
- Convert goals and shots to numeric fields.
- Reject invalid `FTR` values.
- Normalize team names through the mapping file.
- Generate deterministic `match_id` values.
- Concatenate seasons in chronological order.
- Detect duplicates and conflicting duplicate records.
- Preserve source-filename provenance.
- Write `data/processed/canonical_matches.csv` atomically.
- Produce a cleaning summary: input rows, output rows, duplicates, missing shots, unresolved teams.

### CLI contract

```bash
python -m src.clean_data
```

### Acceptance criteria

- Exactly one canonical row exists per fixture.
- Every row is EPL-only.
- Dates are valid and chronological.
- Goals are nonnegative integers.
- Result code agrees with final goals.
- No unknown team names remain.
- `match_id` is unique and deterministic.
- Output row count is explainable from input counts.

### Required tests

- Date parsing across source formats.
- Target/result consistency.
- Team-name mapping.
- Duplicate detection.
- Deterministic `match_id` generation.
- Invalid result and negative-goal rejection.

### Stop condition

Do not create rolling features in this phase.

---

## Phase 3 — Team History and Leakage-Safe Features

### Objective

Create chronological team histories and pre-match rolling features.

### Tasks

- Implement `src/build_history.py`.
- Generate exactly two team-history rows per canonical fixture.
- Calculate team-perspective goals, shots, points, and venue role.
- Implement `src/build_features.py`.
- Sort by team and date.
- Apply `shift(1)` before every rolling calculation.
- Build all non-possession features in Section 6.
- Track source match IDs for testability, either directly in a debug output or through a deterministic helper.
- Join home and away pre-match features back to one fixture row.
- Compute directional edge features.
- Implement training-only cold-start imputation as a reusable preprocessing function, but do not fit it across future splits.
- Save `team_match_history.csv` and the initial `model_dataset.csv`.

### CLI contract

```bash
python -m src.build_history
python -m src.build_features --feature-set baseline
```

### Acceptance criteria

- Team history has exactly twice as many rows as canonical matches.
- Final dataset has one row per canonical fixture after documented minimum-history filtering.
- Current-match mutations do not change current pre-match features.
- Future-match mutations do not change earlier features.
- Edge signs follow the home-positive convention.
- No current result or raw current-match statistic is present in `FEATURE_COLUMNS`.
- Cold-start rows retain history-count indicators.

### Required tests

All leakage and feature tests from Section 7 that do not require a trained model.

### Stop condition

Do not create train/test splits or train models.

---

## Phase 4 — Frozen Splits and Baseline Models

### Objective

Freeze chronological datasets and establish majority/logistic-regression benchmarks.

### Tasks

- Implement `src/split_data.py` using the fixed seasons in Section 8.
- Assert split disjointness and chronological ordering.
- Fit median imputation on training data only.
- Implement the majority-class predictor.
- Implement a scikit-learn logistic-regression pipeline.
- Evaluate both on validation and test sets.
- Save metrics to `reports/model_results.csv`.
- Save split row IDs or a split manifest for reproducibility.

### CLI contract

```bash
python -m src.train_baselines --feature-set baseline
```

### Acceptance criteria

- Splits match the exact season policy.
- No `match_id` occurs in more than one split.
- Validation/test/holdout values do not affect preprocessing fit.
- Both baselines return three probabilities.
- Probabilities pass normalization tests.
- Holdout metrics are not computed or displayed.

### Stop condition

Do not train XGBoost.

---

## Phase 5 — Model A XGBoost Baseline

### Objective

Train, tune, and evaluate the long-history XGBoost model without possession.

### Tasks

- Implement `src/train_xgboost.py`.
- Use `multi:softprob` and fixed label encoding.
- Use validation data for early stopping.
- Perform only the approved hyperparameter search.
- Record every attempted configuration.
- Select by validation log loss.
- Evaluate the selected model on the 2024/25 test set.
- Save model, preprocessing values, class mapping, feature order, best iteration, and training metadata.
- Implement matplotlib reports.

### CLI contract

```bash
python -m src.train_xgboost --model-name model_a --feature-set baseline
python -m src.evaluate --model-name model_a --split test
```

### Acceptance criteria

- Model A beats the majority baseline on test log loss.
- Model A is compared with logistic regression.
- All probability and class-order tests pass.
- Model artifact can be loaded in a fresh Python process.
- Repeated prediction on identical input is deterministic.
- Holdout remains unopened.

### Stop condition

Do not collect possession data.

---

## Phase 6 — SofaScore Team-Season Possession Collection

### Objective

Build a resumable, rate-limited pipeline for EPL team-season possession averages.

### Tasks

- Implement `src/collect_possession.py` using the standard library.
- Discover SofaScore season IDs rather than hard-coding them.
- Retrieve the EPL standings to enumerate exact season teams.
- Cache the season directory, standings, and team-statistics JSON responses, or
  an equivalent manifested SofaScore web export when automated JSON access is
  blocked.
- Implement safe resume, retry, throttling, and request-budget behavior.
- Parse `averageBallPossession` and `matches` without converting missing values to zero.
- Maintain exact SofaScore team-name mappings.
- Produce `data/processed/team_season_possession.csv` deterministically.
- Generate `reports/possession_coverage.csv` by source season and team.
- Record the next target season explicitly so Phase 7 cannot use a same-season final average.
- Determine the first eligible Model B target season using the 95% team-coverage rule.

### CLI contract

```bash
python -m src.collect_possession --season 2024 --max-requests 25
python -m src.collect_possession --all --max-requests 250
```

### Acceptance criteria

- No credential is required or stored.
- Manifested caches are not requested twice.
- Collector can stop and resume without data loss.
- Transient request failures use bounded retries and modest throttling.
- Season IDs and teams are derived from the SofaScore responses.
- Average-possession values parse correctly.
- Missing possession remains missing, not zero.
- Every provider team name resolves through the explicit mapping table.
- The processed table has one row per source season and team.
- Coverage is measured against expected teams from the season standings.
- Every row explicitly maps source season `N-1` to target season `N`.
- Coverage report identifies the first eligible Model B target season.

### Stop condition

Do not train Model B.

---

## Phase 7 — Matched Possession Experiment

### Objective

Measure the incremental value of lagged team-season possession without sample-selection confounding.

### Tasks

- Join each target fixture to both clubs' immediately preceding EPL season averages.
- Construct one possession-eligible row set after the declared promoted-team policy.
- Freeze identical match-ID lists for Model A-Matched and Model B.
- Train Model A-Matched with baseline features only.
- Train Model B with baseline plus possession features.
- Use identical splits and approved tuning policy.
- Compare log loss, macro F1, accuracy, class metrics, and confusion matrices.
- Apply the Model B acceptance rule.
- Document findings without overstating causality.

### CLI contract

```bash
python -m src.train_xgboost --model-name model_a_matched --feature-set baseline_matched
python -m src.train_xgboost --model-name model_b --feature-set possession
python -m src.compare_models --models model_a_matched model_b
```

### Acceptance criteria

- Both models use identical match IDs per split.
- Only `home_previous_season_possession`,
  `away_previous_season_possession`, and `possession_edge` differ between their
  feature sets.
- No target fixture uses a possession average calculated from its own season.
- Comparison report states whether Model B passes each acceptance rule.
- Holdout remains unopened.

### Stop condition

Do not select the production candidate or open the holdout.

---

## Phase 8 — Model Selection Freeze

### Objective

Choose and document the final candidate before examining 2025/26.

### Tasks

- Compare Model A, Model A-Matched, and Model B test results.
- Verify all integrity tests.
- Select the final candidate using Section 10.6.
- Write `reports/model_selection.md`.
- Freeze feature order, preprocessing, model parameters, label mapping, and prediction schema in `config/model_config.json`.
- Commit the freeze before running the holdout.

### Acceptance criteria

- Selection rationale is written before holdout evaluation.
- Config includes a checksum or commit reference for the frozen implementation.
- No holdout metrics exist in reports or logs.
- All tests pass at the freeze commit.

### Stop condition

Wait for explicit user approval to open the final holdout.

---

## Phase 9 — Final 2025/26 Holdout

### Objective

Perform one final untouched evaluation of the frozen candidate.

### Tasks

- Confirm the working tree and freeze configuration.
- Load the frozen model and preprocessing.
- Evaluate on 2025/26 without retraining.
- Save metrics to `reports/final_holdout_results.csv`.
- Generate the final confusion matrix.
- Compare test and holdout degradation.
- Do not tune after viewing results.

### CLI contract

```bash
python -m src.evaluate --model-name selected --split holdout --frozen
```

### Acceptance criteria

- Frozen configuration matches the pre-holdout record.
- Model is not refit on holdout rows.
- Results are clearly labeled final holdout.
- Probability and integrity tests pass.
- No post-holdout parameter or feature changes occur.

### Stop condition

Do not implement upcoming-fixture prediction until holdout reporting is complete.

---

## Phase 10 — Upcoming-Fixture Prediction Command

### Objective

Provide a deterministic CLI that builds current pre-match features and returns three probabilities.

### CLI contract

```bash
python -m src.predict \
  --home "Arsenal" \
  --away "Chelsea" \
  --date 2026-09-12
```

Expected JSON output:

```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "match_date": "2026-09-12",
  "model_name": "selected model name",
  "model_version": "version or commit",
  "probabilities": {
    "away_win": 0.23,
    "draw": 0.28,
    "home_win": 0.49
  },
  "predicted_outcome": "home_win",
  "feature_as_of": "latest completed EPL match timestamp",
  "warnings": []
}
```

### Tasks

- Load saved model and preprocessing metadata.
- Validate canonical team names.
- Rebuild the exact frozen feature order.
- Use completed fixtures strictly before the requested date.
- Report cold-start or stale-data warnings.
- Map probabilities through `model.classes_`.
- Emit machine-readable JSON.

### Acceptance criteria

- Probabilities sum to one.
- Repeated calls with identical stored data are deterministic.
- Unknown teams fail clearly.
- A prediction date before the latest source match is rejected or explicitly handled as a historical backtest.
- Feature order matches training metadata exactly.
- No result or statistic from the predicted fixture is required.

---

## Phase 11 — Documentation and Reproducibility

### Objective

Make the repository understandable and runnable by another developer.

### Tasks

- Complete `README.md` with problem, architecture, setup, data sources, commands, results, limitations, and project structure.
- Include the Model A vs Model B experiment and final model decision.
- Document free-tier API quota behavior.
- Add a data dictionary.
- Add a reproducibility section with exact commands from clean clone to reports.
- Add sample prediction output.
- Include charts without claiming that feature importance proves causation.
- State that the project is EPL-only.
- State known cold-start and missing-data limitations.

### Acceptance criteria

- A new developer can follow the README without this chat.
- All documented commands are valid.
- No secret or local absolute path appears in documentation.
- Reported metrics match saved report files.

---

## Phase 12 — Final Quality Gate

### Objective

Verify the repository is ready for portfolio and résumé use.

### Required validation

```bash
python -m unittest discover -s tests -v
python -m src.download_data --help
python -m src.clean_data --help
python -m src.build_features --help
python -m src.train_xgboost --help
python -m src.evaluate --help
python -m src.predict --help
git status --short
```

### Final checklist

- [ ] All automated tests pass.
- [ ] Raw sources are reproducible.
- [ ] Leakage tests pass.
- [ ] Frozen season splits are documented.
- [ ] Baselines are reported.
- [ ] Model selection follows the declared rule.
- [ ] Holdout was opened only after freezing.
- [ ] Prediction CLI returns normalized probabilities.
- [ ] README commands run successfully.
- [ ] No secrets are tracked.
- [ ] No non-EPL data enters the pipeline.
- [ ] Results and limitations are stated honestly.

### Definition of done

The project is complete when a fresh environment can reproduce the processed dataset, train the declared models, regenerate reports, load the selected model, and produce an upcoming-fixture prediction without consulting this chat.

---

## 13. Global Error-Handling Requirements

Every command-line module must:

- Exit nonzero on failure.
- Print a concise actionable error.
- Avoid partial final outputs by writing to a temporary path and renaming atomically.
- Validate inputs before expensive work.
- Avoid swallowing exceptions without context.
- Preserve cached/raw data after downstream failures.
- Never print API keys or complete authentication headers.

Expected failures that require explicit messages:

- Missing raw files.
- Missing required columns.
- Invalid or ambiguous team mappings.
- Duplicate fixtures.
- Malformed dates or results.
- Missing API key.
- API quota exhaustion.
- Possession coverage below threshold.
- Empty model split.
- Unexpected model class order.
- Feature-order mismatch during prediction.

---

## 14. Configuration Contracts

### 14.1 `config/model_config.json`

Minimum fields:

```json
{
  "random_seed": 42,
  "target_mapping": {"A": 0, "D": 1, "H": 2},
  "rolling_window": 5,
  "rolling_min_periods": 3,
  "train_seasons": ["1011", "1112", "...", "2223"],
  "validation_seasons": ["2324"],
  "test_seasons": ["2425"],
  "holdout_seasons": ["2526"],
  "possession_coverage_threshold": 0.95,
  "max_macro_f1_drop": 0.02,
  "feature_columns": [],
  "selected_model": null,
  "frozen_at_utc": null,
  "git_commit": null
}
```

### 14.2 `models/model_metadata.json`

Minimum fields:

```json
{
  "model_name": "model_a",
  "trained_at_utc": "ISO-8601 timestamp",
  "feature_columns": [],
  "classes": [0, 1, 2],
  "best_iteration": 0,
  "parameters": {},
  "training_match_count": 0,
  "validation_match_count": 0,
  "test_match_count": 0,
  "training_date_min": "YYYY-MM-DD",
  "training_date_max": "YYYY-MM-DD",
  "source_manifest_sha256": "checksum",
  "git_commit": "commit hash"
}
```

---

## 15. Decision Log

| Decision | Reason |
|---|---|
| EPL-only data | Preserve scope and avoid cross-league distribution shifts. |
| Football-Data as primary history | Free, stable, season-based CSVs with results and shots. |
| SofaScore team-season averages for enrichment | Avoids hundreds of match-statistics requests per season while retaining a broad possession-style signal. |
| Previous-season lag for possession | Prevents a final season average from leaking future fixtures into earlier predictions. |
| Two matched models for possession experiment | Prevent possession value from being confounded by different row coverage. |
| One row per fixture | Avoid contradictory mirrored predictions. |
| Home-oriented target encoding | Directly maps one fixture to H/D/A probabilities. |
| `shift(1)` before rolling | Prevent current-match leakage. |
| Chronological splits | Match real future-prediction conditions. |
| Log loss as primary metric | The product outputs probabilities. |
| Macro F1 as guardrail | Prevent poor draw performance from being hidden. |
| 2025/26 final holdout | Most recent complete unseen season in the version-1 plan. |
| No current-season 2026/27 training | The season is incomplete and would create unstable evaluation. |

---

## 16. Reference Links

- [Football-Data EPL downloads](https://www.football-data.co.uk/englandm.php)
- [Football-Data column definitions](https://www.football-data.co.uk/notes.txt)
- [SofaScore EPL season directory](https://www.sofascore.com/api/v1/unique-tournament/17/seasons)
- [StatsBomb Open Data](https://github.com/hudl/open-data)
- [pandas GroupBy documentation](https://pandas.pydata.org/docs/user_guide/groupby.html)
- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [scikit-learn log loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)
- [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [Official Codex IDE extension guide](https://learn.chatgpt.com/docs/codex/ide)

---

## 17. Current Next Action

Phases 0 through 10 are complete. The validated SofaScore export contains 180
team-season rows across 2017/18 through 2025/26, all with possession values and
38 recorded matches. Model B passed all declared Phase 7 incremental-value
rules on the frozen matched test cohort.
Phase 8 selected and froze Model B using Section 10.6. The rationale is in
`reports/model_selection.md` and the frozen configuration can be verified with
`python -m src.freeze_model --verify`. Following explicit approval, Phase 9
evaluated the frozen Model B once on all 380 fixtures from 2025/26, without
refitting or tuning. Results, test-to-holdout degradation, and limitations are
in `reports/final_holdout.md`; the completion receipt and separate Phase 9
evaluation protocol preserve the original Phase 8 freeze record.
Phase 10 implements `python -m src.predict --home "Arsenal" --away "Chelsea"
--date 2026-09-12` using the frozen model and local snapshot. It preserves the
feature definitions, rejects historical/same-date requests, and warns about
cold starts, stale history, and unverified fixture scheduling. Its separate
implementation protocol leaves the pre-holdout records untouched.
Phase 11 is the next action and requires a separate user request. No modeling
change may be optimized against the now-opened 2025/26 holdout.
