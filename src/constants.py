"""Fixed labels, rolling settings, and Phase 3 output column contracts."""

ROLLING_WINDOW = 5
ROLLING_MIN_PERIODS = 3

TARGET_MAPPING = {"A": 0, "D": 1, "H": 2}

TEAM_HISTORY_COLUMNS = (
    "team_match_id",
    "match_id",
    "season",
    "date",
    "team",
    "team_slug",
    "opponent",
    "opponent_slug",
    "is_home",
    "goals_for",
    "goals_against",
    "shots",
    "possession",
    "points",
)

TEAM_ROLLING_FEATURE_COLUMNS = (
    "goals_for_avg_5",
    "goals_against_avg_5",
    "shots_avg_5",
    "form_points_5",
    "overall_ppg_5",
    "venue_ppg_5",
    "history_matches",
    "venue_history_matches",
)

HOME_FEATURE_COLUMNS = tuple(
    f"home_{column}" for column in TEAM_ROLLING_FEATURE_COLUMNS
)
AWAY_FEATURE_COLUMNS = tuple(
    f"away_{column}" for column in TEAM_ROLLING_FEATURE_COLUMNS
)
EDGE_FEATURE_COLUMNS = (
    "goals_scored_edge",
    "defensive_edge",
    "shots_edge",
    "form_edge",
    "venue_edge",
    "history_edge",
)

FEATURE_COLUMNS = HOME_FEATURE_COLUMNS + AWAY_FEATURE_COLUMNS + EDGE_FEATURE_COLUMNS

MODEL_IDENTITY_COLUMNS = (
    "match_id",
    "season",
    "date",
    "home_team",
    "away_team",
    "target",
)
MODEL_DATASET_COLUMNS = MODEL_IDENTITY_COLUMNS + FEATURE_COLUMNS

CURRENT_MATCH_STAT_COLUMNS = frozenset(
    {
        "home_goals",
        "away_goals",
        "result_code",
        "home_shots",
        "away_shots",
        "home_possession",
        "away_possession",
        "points",
    }
)
