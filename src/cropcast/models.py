"""Step 3: train the models and pick the best settings using the validation season."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RF_GRID = [(d, leaf) for d in (6, 10, 14) for leaf in (1, 3, 5)]
GB_GRID = [(lr, d) for lr in (0.03, 0.05, 0.1) for d in (2, 3, 4)]


def _rmse(y, p) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def make_linear():
    return make_pipeline(StandardScaler(), LinearRegression())


def tune_random_forest(Xtr, ytr, Xva, yva, seed: int = 0):
    best = None
    for depth, leaf in RF_GRID:
        m = RandomForestRegressor(n_estimators=200, max_depth=depth, min_samples_leaf=leaf,
                                  random_state=seed, n_jobs=-1).fit(Xtr, ytr)
        score = _rmse(yva, m.predict(Xva))
        if best is None or score < best["val_rmse"]:
            best = dict(val_rmse=score, max_depth=depth, min_samples_leaf=leaf)
    final = RandomForestRegressor(n_estimators=300, max_depth=best["max_depth"],
                                  min_samples_leaf=best["min_samples_leaf"],
                                  random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    return final, best


def make_gb(learning_rate: float, max_depth: int, seed: int = 0):
    return GradientBoostingRegressor(n_estimators=300, learning_rate=learning_rate, max_depth=max_depth,
                                     subsample=0.8, random_state=seed)


def tune_gradient_boosting(Xtr, ytr, Xva, yva, seed: int = 0):
    best = None
    for lr, depth in GB_GRID:
        m = make_gb(lr, depth, seed).fit(Xtr, ytr)
        score = _rmse(yva, m.predict(Xva))
        if best is None or score < best["val_rmse"]:
            best = dict(val_rmse=score, learning_rate=lr, max_depth=depth)
    final = make_gb(best["learning_rate"], best["max_depth"], seed).fit(Xtr, ytr)
    return final, best
