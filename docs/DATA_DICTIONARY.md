# Data dictionary

All football observations are EPL-only. Empty CSV cells mean missing, not zero.
Identity columns are strings (especially four-digit season codes), dates are
`YYYY-MM-DD`, percentages use 0–100, and returned model probabilities use 0–1.
CSV column order is a fixed contract. Data are never silently fuzzy-matched.

## Raw inputs and provenance

`data/raw/football_data/E0_{season}.csv`: immutable provider bytes, 380 fixtures
per season from 2010/11 through 2025/26. `Div=E0` identifies EPL. `Date`, optional
`Time`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `HS`, and `AS` feed cleaning.
Goals/results are required; missing shots remain missing. Betting columns may
exist in raw input but never enter the model.

The SofaScore raw CSV export has the ordered columns `season`, `season_id`,
`team`, `team_id`, `average_ball_possession`, `matches`, and `source_url`.
These retain the provider's season/team identities; the processed table below
adds canonical names, derived season fields and the explicit lag. The export
has 180 team-season rows from 2017/18–2025/26.
Provider statistics are percentages of possession, not minutes or individual
match observations. The two clubs' preceding-season averages need not sum to 100.

`data/raw/manifest.json` records source, source URL, retrieval time, local path,
SHA-256, and row count, plus applicable SofaScore endpoint/season/team metadata.
Checksums identify exact cached snapshots; live endpoints may subsequently change.

## Canonical matches — `data/processed/canonical_matches.csv`

One row per completed fixture, 6,080 rows, sorted by date and fixture identity.

| Column(s) | Type / meaning |
|---|---|
| `match_id` | Unique `{season}|{date}|{home_slug}|{away_slug}` identifier |
| `season` | Four-digit season code, e.g. `2526` for 2025/26 |
| `date` | Calendar match date; primary temporal boundary |
| `kickoff_time` | Optional `HH:MM`; not a known final-whistle time or fabricated UTC timestamp |
| `home_team`, `away_team` | Exact canonical club names |
| `home_team_slug`, `away_team_slug` | Stable canonical slug keys |
| `home_goals`, `away_goals` | Nonnegative full-time integer goals |
| `result_code` | `A`, `D`, or `H`, consistent with goals |
| `home_shots`, `away_shots` | Nullable nonnegative integer shots |
| `home_possession`, `away_possession` | Nullable legacy match percentages; unpopulated by the season-average source |
| `football_data_source_file` | Relative raw source path for that season |
| `api_fixture_id` | Nullable legacy API identity; not used by current possession join |

Legacy per-fixture possession requires its own API provenance if populated. It
is not a way to insert a season average into each fixture. Duplicate/conflicting
fixture IDs, unknown clubs, invalid dates or score/result disagreement fail.

## Team history — `data/processed/team_match_history.csv`

Two rows per completed fixture, 12,160 rows. These are observed history, not
already-shifted features.

| Column(s) | Type / meaning |
|---|---|
| `team_match_id` | Unique match ID plus `|home` or `|away` |
| `match_id`, `season`, `date` | Parent fixture identity and timing |
| `team`, `team_slug` | Club whose perspective the row represents |
| `opponent`, `opponent_slug` | Opposing club |
| `is_home` | Boolean venue role |
| `goals_for`, `goals_against` | Integer goals from this club's perspective |
| `shots` | Nullable integer shots by this club |
| `possession` | Nullable legacy match percentage, not the model's lagged season average |
| `points` | 3 for a win, 1 for a draw, 0 for a loss |

## Season possession — `data/processed/team_season_possession.csv`

180 rows, one per source season and canonical team. The valid source coverage is
100%; this does not imply 100% fixture-feature coverage for promoted clubs.

| Column | Type / meaning |
|---|---|
| `source_season` | Completed EPL season supplying the average |
| `target_season` | Immediately following season, the only permitted target |
| `source_season_start_year` | Full integer starting year |
| `sofascore_season_id` | Provider season identity |
| `team`, `team_slug` | Canonical club name and join key |
| `sofascore_team_name`, `sofascore_team_id` | Exact provider identity |
| `average_possession_pct` | Nullable float percentage in [0, 100] |
| `matches_recorded` | Provider match count; 38 for every row in this snapshot |
| `source_url` | Statistics endpoint provenance |

The lag is mandatory: source `2425` maps to target `2526`, never to `2425`.
Source `2526` supports future `2627` predictions. A club absent from the preceding
EPL season retains missing possession; an older season or Championship data is
never substituted. Missing established-team rows and inadequate source coverage
are integrity failures under the existing join/prediction checks.

## Model datasets and exact feature order

`model_dataset.csv` has 6,080 baseline fixture rows. `matched_model_dataset.csv`
has 3,040 rows from 2018/19–2025/26 (earliest available preceding possession
season is 2017/18). Both begin with these six identity/label columns:

```text
match_id, season, date, home_team, away_team, target
```

`target` is an integer: away win = 0, draw = 1, home win = 2. It is never an input
feature. All feature values are numeric; rolling values and possession may be
missing before imputation. Counts are always present, including zero.

Exact 25-feature order for Model B; the first 22 are the Model A/A-Matched order:

```text
home_goals_for_avg_5
home_goals_against_avg_5
home_shots_avg_5
home_form_points_5
home_overall_ppg_5
home_venue_ppg_5
home_history_matches
home_venue_history_matches
away_goals_for_avg_5
away_goals_against_avg_5
away_shots_avg_5
away_form_points_5
away_overall_ppg_5
away_venue_ppg_5
away_history_matches
away_venue_history_matches
goals_scored_edge
defensive_edge
shots_edge
form_edge
venue_edge
history_edge
home_previous_season_possession
away_previous_season_possession
possession_edge
```

| Feature suffix / edge | Definition and unit |
|---|---|
| `goals_for_avg_5` | Mean goals scored in the previous five EPL matches |
| `goals_against_avg_5` | Mean goals conceded in the previous five EPL matches |
| `shots_avg_5` | Mean shots in that window, requiring at least three nonmissing observations |
| `form_points_5` | Sum of points in that window (0–15 when five results exist) |
| `overall_ppg_5` | Mean points per match in that window (0–3) |
| `venue_ppg_5` | Mean points in the previous five matches at the same home/away role |
| `history_matches` | Count of all stored strictly prior EPL matches, not capped at five |
| `venue_history_matches` | Count of all strictly prior matches at that role |
| `goals_scored_edge` | Home goals-for mean minus away goals-for mean |
| `defensive_edge` | Away goals-against mean minus home goals-against mean |
| `shots_edge` | Home shots mean minus away shots mean |
| `form_edge` | Home form-points sum minus away form-points sum |
| `venue_edge` | Home-role PPG minus away-role PPG |
| `history_edge` | Home history count minus away history count |
| `previous_season_possession` | That team's immediately preceding EPL average (%) |
| `possession_edge` | Home minus away previous-season averages (percentage points) |

At least three observations are needed for rolling aggregates; missing shots
count separately from available results. Overall and venue windows are distinct.
All source dates must be strictly earlier than the fixture date; same-day matches
are contemporaneous. Windows carry across EPL seasons. Larger edge values follow
the home-positive convention, without claiming a causal advantage.

Edges are calculated before imputation; a missing edge uses its own frozen
training median, not a difference recomputed from already-imputed team values.
The saved preprocessing contains strategy `median`, `fitted_on_split=train`,
the ordered feature list and a value per feature. It is never fit on future rows.

## Splits, model artifacts and reports

`split_manifest.csv` and `matched_split_manifest.csv` both have `match_id`,
`season`, `date`, `split`. Split values are `train`, `validation`, `test`,
`holdout`, with disjoint IDs and chronological boundaries. Model A training is
2010/11–2022/23 (4,940 rows); matched training is 2018/19–2022/23 (1,900 rows).
Validation/test/holdout are 2023/24, 2024/25, 2025/26, each 380 rows.

The XGBoost JSONs save native model state. Metadata records model/feature-set
names, feature order, class labels, best iteration, parameters, split counts,
training time and checksums. `config/model_config.json` stores the authoritative
selected Model B fields under `frozen_candidate`; its top-level `feature_columns`
still represents the earlier 22-feature baseline contract.

`model_results.csv` and `final_holdout_results.csv` use the same fixed metric
columns: model name/family, feature set, split, row count, log loss, macro F1,
accuracy, best iteration and serialized parameters, followed by precision,
recall, F1 and support for each of `away_win`, `draw`, `home_win`, then the nine
`confusion_{actual}_pred_{predicted}` integer counts. Support and matrix totals
count fixtures, not mirrored team rows. Log loss is lower-is-better; macro F1
and accuracy are higher-is-better. Undefined per-class scores use zero.

`tuning_results.csv` records every candidate's validation metrics, selected
flag, best iteration and approved parameters. It does not use test/holdout
metrics for tuning. `possession_coverage.csv` records source/team coverage.

`final_holdout_predictions.json` stores each frozen holdout match ID, observed
target and three probabilities, with explicit probability column names.
The start record prevents repeat inference; the completion receipt records
completion and output checksums. These are historical audit files, not inputs
to feature construction.

## Prediction JSON

See [the exact sample](sample_prediction.json). Home/away/date identify the
user-supplied fixture. `model_name` and `model_version` identify the saved model
and original freeze checksum. `probabilities` maps away win/draw/home win to
finite numbers in [0,1] summing to one within 0.000001, mapped through estimator
classes. `predicted_outcome` is the largest probability. `feature_as_of` is the
latest stored completed match date; `warnings` is an ordered list of strings
covering missing data, staleness and unverified scheduling. Neither kickoff nor
completion times are fabricated, and no current-fixture result is required.
