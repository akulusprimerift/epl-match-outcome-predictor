"""Fixed labels, seasons, rolling settings, and output column contracts."""

RANDOM_SEED = 42
ROLLING_WINDOW = 5
ROLLING_MIN_PERIODS = 3

TARGET_MAPPING = {"A": 0, "D": 1, "H": 2}
CLASS_LABELS = (0, 1, 2)
CLASS_NAMES = ("away_win", "draw", "home_win")

TRAIN_SEASONS = (
    "1011",
    "1112",
    "1213",
    "1314",
    "1415",
    "1516",
    "1617",
    "1718",
    "1819",
    "1920",
    "2021",
    "2122",
    "2223",
)
VALIDATION_SEASONS = ("2324",)
TEST_SEASONS = ("2425",)
HOLDOUT_SEASONS = ("2526",)
SPLIT_ORDER = ("train", "validation", "test", "holdout")

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

SPLIT_MANIFEST_COLUMNS = ("match_id", "season", "date", "split")

MODEL_RESULT_COLUMNS = (
    "model_name",
    "model_family",
    "feature_set",
    "split",
    "row_count",
    "log_loss",
    "macro_f1",
    "accuracy",
    "best_iteration",
    "parameters",
    "precision_away_win",
    "recall_away_win",
    "f1_away_win",
    "support_away_win",
    "precision_draw",
    "recall_draw",
    "f1_draw",
    "support_draw",
    "precision_home_win",
    "recall_home_win",
    "f1_home_win",
    "support_home_win",
    "confusion_away_win_pred_away_win",
    "confusion_away_win_pred_draw",
    "confusion_away_win_pred_home_win",
    "confusion_draw_pred_away_win",
    "confusion_draw_pred_draw",
    "confusion_draw_pred_home_win",
    "confusion_home_win_pred_away_win",
    "confusion_home_win_pred_draw",
    "confusion_home_win_pred_home_win",
)

TUNING_RESULT_COLUMNS = (
    "candidate_id",
    "selected",
    "validation_log_loss",
    "validation_macro_f1",
    "validation_accuracy",
    "best_iteration",
    "objective",
    "num_class",
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "eval_metric",
    "early_stopping_rounds",
    "random_state",
)
