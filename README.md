# EPL Match Outcome Predictor

A reproducible Python project for predicting English Premier League match
outcomes as away-win, draw, and home-win probabilities using only information
available before kickoff.

The repository has completed Phases 0 through 10 and contains a validated
**SofaScore team-season possession dataset**. It contains
immutable raw EPL match files, leakage-safe pre-match features, frozen
chronological splits, majority/logistic-regression benchmarks, and a tuned
long-history XGBoost model. The coverage-matched baseline and possession model
have also been trained and compared. Model B is the frozen final candidate.
The final 2025/26 holdout has been evaluated once without retraining. Results
and limitations are recorded in `reports/final_holdout.md`.

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

The training commands below describe the pre-freeze workflow. Once Phase 8 is
frozen, training entry points refuse to overwrite its artifacts. Preserve the
local model JSON files under `models/`; the existing ignore policy keeps them
out of Git, while the freeze records their exact checksums.

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

The collector can discover SofaScore season IDs, get the teams from each
season's standings, and retrieve one `averageBallPossession` value per team.
Because SofaScore currently returns HTTP 403 to this environment's automated
JSON requests, the repository also accepts the manifested SofaScore web export
already stored under `data/raw/sofascore/`. Raw inputs are checksummed in
`data/raw/manifest.json`, never overwritten, and skipped on rerun. The current
export contains all 20 EPL teams for every source season from 2017/18 through
2025/26, with 38 matches recorded for every team-season.

The deterministic output is `data/processed/team_season_possession.csv`, with
coverage recorded in `reports/possession_coverage.csv`. Each source-season row
also names its following target season. A 2023/24 team average may therefore be
used for 2024/25 fixtures, but never for fixtures within 2023/24 itself. This
one-season lag is mandatory to prevent future leakage.

## Run the matched possession experiment

The validated source-season coverage is 100%, so the first eligible Model B
target season is 2018/19. Phase 7 joins the lagged team-season table and can be
reproduced with these commands:

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

The completed comparison used 1,900 training, 380 validation, and 380 test
fixtures for each model. Model B lowered test log loss from `1.047621` to
`1.031919`, increased test macro F1 from `0.330596` to `0.366448`, and increased
test accuracy from `0.450000` to `0.471053`. It therefore passes every declared
Phase 7 incremental-value rule. Full results are in
`reports/possession_experiment.md`.

## Model selection freeze

Phase 8 selected Model B under Section 10.6: its test log loss (`1.031919`) is
lower than Model A (`1.039492`) and Model A-Matched (`1.047621`), and it has the
highest macro F1 of those three candidates. The comparison uses the same 380
test fixtures. The test advantage is modest; only the selected candidate was
subsequently evaluated on the final holdout, so no holdout comparison against
other candidates is claimed. The rationale is recorded in
`reports/model_selection.md`.

`config/model_config.json` records the selection and its timestamp. Its
`frozen_candidate` section contains the exact 25-feature order, training-only
preprocessing values, parameters, best iteration, labels, prediction schema,
and checksums for the implementation, saved models, source data, and reports.
The existing top-level `feature_columns` remains the baseline contract.

Verify the freeze without running inference or changing files:

```bash
python -m src.freeze_model --verify
```

The creation command is `python -m src.freeze_model`; if already frozen, it
only verifies the existing record. Implementation hashes normalize CRLF to LF;
raw data and artifact checksums use exact bytes. The `git_commit` field records
the pre-freeze parent, and the commit containing the configuration is the freeze
commit. The configuration also has its own checksum, excluding that checksum
field itself.

## Final holdout evaluation (Phase 9)

Model B was evaluated on all 380 fixtures from 2025/26:

| Metric | 2024/25 test | 2025/26 final holdout |
|---|---:|---:|
| Log loss (lower is better) | 1.031919 | 1.075037 |
| Macro F1 | 0.366448 | 0.353288 |
| Accuracy | 47.11% | 46.32% |

Log loss worsened by 4.18%; accuracy dropped by 0.79 percentage points.
Draw recognition is weak: just 2 of 104 draws were correctly classified
(1.92% draw recall). This is a sequential pre-match backtest using earlier
completed matches, not a preseason forecast. Possession for these fixtures
comes from 2024/25; final 2025/26 averages are not prediction features.
The result does not justify retuning against this now-opened holdout.

The pre-evaluation suite passed all 132 tests. Probability, leakage, frozen
artifact, and one-time evaluation checks are included. The evaluated model,
feature order, parameters, split membership, and training-only medians remain
unchanged.

The approved one-time command now verifies and returns the saved result:

```bash
python -m src.evaluate --model-name selected --split holdout --frozen
```

The command requires a clean working tree and a committed evaluation protocol
in `config/phase9_protocol.json`. This separate record preserves the original
Phase 8 configuration and all model, preprocessing, data, feature-building, and
training checksums. It permits only the evaluation/verification adapters and
their tests; the original inference functions are also checked against the
Phase 8 commit. The model is loaded, never refitted.

Outputs are `reports/final_holdout_results.csv`, a final confusion matrix,
per-fixture probabilities, and `reports/final_holdout.md` comparing test and
holdout results. A start record prevents repeat or concurrent inference; the
completion receipt checksums every final output. Repeating the same command
only verifies and returns saved results. An interrupted attempt remains locked:
preserve its files for investigation, and do not delete the record or rerun
inference. A fresh evaluation requires a new future holdout, not retuning on
2025/26.

The original model metadata and Phase 8 report remain immutable historical
records. Current completion status is stored in
`reports/final_holdout_receipt.json` and reported by the freeze verifier.

## Upcoming-fixture prediction (Phase 10)

With the project dependencies installed, request a prediction using exact
canonical team names:

```bash
python -m src.predict --home "Arsenal" --away "Chelsea" --date 2026-09-12
python -m src.predict --help
```

This is an illustrative user-supplied fixture, not a verified scheduled match.
The command prints one JSON object containing `home_team`, `away_team`,
`match_date`, `model_name`, `model_version`, `probabilities` (away win, draw,
home win), `predicted_outcome`, `feature_as_of`, and `warnings`. The model
version is the unchanged Phase 8 freeze-record checksum. Errors go to standard
error with a nonzero exit status. No files are written or downloaded.

The predictor loads the saved Model B and its training-only medians, builds
the exact frozen 25-feature order, and maps probabilities through the model's
class labels. It never refits the model or preprocessing. The prediction
extension is recorded in `config/phase10_protocol.json`; the original Phase 8
configuration, Phase 9 protocol, holdout reports, and training/feature-building
implementations remain unchanged. The existing holdout command still returns
its saved result without repeating inference.

Important limits:

- Only completed EPL fixtures strictly before the requested date contribute.
  Dates on or before the latest stored match date are rejected; this command
  does not offer historical backtesting.
- The current snapshot ends on **2026-05-24**. Team histories more than 14 days
  old produce warnings; this warning threshold does not change predictions.
  The example September prediction therefore uses stale history, not live form.
- For upcoming date-only requests, July 1 marks the new season. A 2026/27
  fixture uses completed 2025/26 possession. The required preceding season must
  have its complete 380-match history and 20-team possession table, with the
  frozen coverage threshold satisfied. Later unsupported seasons fail clearly.
- Missing prior EPL history or possession produces explicit warnings and uses
  frozen training medians. Older possession seasons and other leagues are never
  substituted. History counts remain available to the model.
- Exact canonical names are required; for example, use `Manchester United`,
  not `Man United`. Matching a historical EPL club does not verify its current
  league membership or whether the fixture is scheduled.
- `feature_as_of` is a match date, not a fabricated completion timestamp.

Phase 10 acceptance passed all 147 tests, including feature parity with the
existing pipeline, strict-date exclusion, deterministic normalized output,
unknown-team errors, cold starts, and frozen-artifact checks. Documentation
and reproducibility work beyond this command remains Phase 11.

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
