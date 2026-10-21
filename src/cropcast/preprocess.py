"""Step 2: fill gaps, build helpful new features and cap extreme values.

Everything that is "learned" from data (medians, caps) is learned from the
TRAINING rows only and then applied unchanged to validation and test rows.
That stops information from the future leaking into the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import BASE_FEATURES, FLAG_COLUMNS, TARGET, TRAIN_END_YEAR, VALIDATION_YEAR

IDEAL_PH = 6.7


def split_by_year(df: pd.DataFrame, train_end: int = TRAIN_END_YEAR, val_year: int = VALIDATION_YEAR):
    """Split by calendar so the model is only ever tested on seasons after it was trained."""
    train = df[df["year"] <= train_end]
    val = df[df["year"] == val_year]
    test = df[df["year"] > val_year]
    return train, val, test


@dataclass
class Preprocessor:
    medians: pd.Series = field(default=None)
    lower: pd.Series = field(default=None)
    upper: pd.Series = field(default=None)
    columns: list = field(default=None)

    # ---- learning from training data ----
    def fit(self, train: pd.DataFrame) -> "Preprocessor":
        self.medians = train[BASE_FEATURES].median()
        X = self._build(train)
        X = X[train[TARGET].notna().values]
        self.lower, self.upper = X.quantile(0.01), X.quantile(0.99)
        self.columns = list(X.columns)
        return self

    # ---- applying what was learned ----
    def transform(self, df: pd.DataFrame, cap: bool = True) -> pd.DataFrame:
        X = self._build(df)
        return X.clip(self.lower, self.upper, axis=1) if cap else X

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[BASE_FEATURES].copy()
        for c in FLAG_COLUMNS:
            X[c + "_was_missing"] = X[c].isna().astype(int)
        X = X.fillna(self.medians)
        X["water_total_mm"] = X["rainfall_mm"] + X["irrigation_mm"]
        X["heat_x_temp"] = X["heat_days"] * X["avg_temp_c"]
        X["ph_dev_sq"] = (X["soil_ph"] - IDEAL_PH) ** 2
        return X
