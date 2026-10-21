"""Small, reusable definitions shared by the modeling notebooks.

The notebooks keep their actual experiments visible. This module centralizes
only definitions that must agree across model selection, final testing, and
the exploratory trading backtest.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


FEATURE_COLUMNS = [
    "Daily_Return_Pct",
    "Momentum_5d_Pct",
    "Volatility_20d_Pct",
    "MA_Distance_20d_Pct",
]
TARGET_COLUMN = "Target_Next_Day_Return_Pct"

# Frozen after the validation comparison in notebook 03. Do not retune on test.
SELECTED_RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "min_samples_leaf": 10,
    "random_state": 42,
    "n_jobs": -1,
}


def load_model_data(project_dir: Path) -> pd.DataFrame:
    """Load the saved feature/target dataset and check basic integrity."""
    path = project_dir / "data" / "qqq_model_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run notebooks 01 and 02 in order first."
        )

    data = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Required model columns are missing: {sorted(missing)}")
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("Model dates must be unique and chronologically sorted.")
    if data[required].isna().any().any():
        raise ValueError("Model features and targets cannot contain missing values.")
    return data


def chronological_split(data: pd.DataFrame):
    """Use the original 70/15/15 positions with a one-row boundary gap.

    The target on row t is the return from close t to close t+1. Removing
    the final training and validation rows prevents their labels from
    reaching into the next data period.
    """
    n = len(data)
    if n < 10:
        raise ValueError("Too few observations for a three-way split.")
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = data.iloc[: train_end - 1].copy()
    validation = data.iloc[train_end : val_end - 1].copy()
    test = data.iloc[val_end:].copy()

    if not (train.index.max() < validation.index.min() < test.index.min()):
        raise ValueError("Data partitions overlap or are not chronological.")
    return train, validation, test


def selected_random_forest() -> RandomForestRegressor:
    """Return the frozen validation-selected model, not yet fitted."""
    return RandomForestRegressor(**SELECTED_RF_PARAMS)


def regression_metrics(actual, predicted) -> dict:
    """Errors are expressed in percentage points of daily return."""
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": root_mean_squared_error(actual, predicted),
    }
