"""Put all the steps together: clean, split, train, evaluate, and make forecasts."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import evaluate as ev
from .config import BASE_FEATURES, RULES, TARGET
from .models import make_gb, make_linear, tune_gradient_boosting, tune_random_forest
from .preprocess import Preprocessor
from .validate import QualityReport, apply_rules

MIN_SEASONS = 6


def choose_split_years(df: pd.DataFrame):
    """Last two seasons = test, the one before = validation, everything earlier = training."""
    years = sorted(df["year"].dropna().unique())
    if len(years) < MIN_SEASONS:
        raise ValueError(f"Need at least {MIN_SEASONS} different seasons (years) of data; found {len(years)}.")
    return int(years[-3] - 1), int(years[-3])   # (train_end, validation_year)


@dataclass
class Results:
    quality: QualityReport
    clean: pd.DataFrame
    split_years: dict
    split_sizes: dict
    medians: dict
    metrics: dict                  # model name -> metric dict on the test set
    best_name: str
    tuning: dict
    interval: tuple                # (low, high) added to a forecast for a 90% range
    coverage_pct: float
    cv: list                       # time-series cross-validation rows
    by_year: dict
    importance: list               # [(feature, rmse increase)]
    card: list                     # pass-mark report card
    test_table: pd.DataFrame       # per-record actual vs predicted (best model)
    forecaster: "Forecaster" = field(repr=False, default=None)


class Forecaster:
    """Turns new farm inputs into a forecast with a 90% range."""

    def __init__(self, pre: Preprocessor, model, q_lo: float, q_hi: float):
        self.pre, self.model, self.q_lo, self.q_hi = pre, model, q_lo, q_hi

    def predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        df = inputs.copy()
        for col, (lo, hi) in RULES.items():          # impossible values become gaps, then get filled
            if col in df.columns:
                bad = (df[col] < lo) | (df[col] > hi)
                df.loc[bad, col] = np.nan
        X = self.pre.transform(df)
        p = self.model.predict(X[self.pre.columns])
        return pd.DataFrame({"forecast_t_ha": p, "low_t_ha": p + self.q_lo, "high_t_ha": p + self.q_hi}, index=inputs.index)

    def what_if(self, base: dict, feature: str, values) -> pd.DataFrame:
        rows = pd.DataFrame([{**base, feature: v} for v in values])
        out = self.predict(rows)
        out.insert(0, feature, list(values))
        return out


def run(df: pd.DataFrame, seed: int = 0, progress=None) -> Results:
    """Run the full pipeline on a table with the required columns."""
    def tick(msg):
        if progress:
            progress(msg)

    tick("Checking the data")
    clean, quality = apply_rules(df)
    clean = clean.reset_index(drop=True)
    clean = clean[clean[TARGET].notna()].reset_index(drop=True)

    train_end, val_year = choose_split_years(clean)
    tr, va, te = clean[clean.year <= train_end], clean[clean.year == val_year], clean[clean.year > val_year]
    if min(len(tr), len(va), len(te)) < 20:
        raise ValueError("Each of the training, validation and test sets needs at least 20 rows.")

    tick("Filling gaps and building features")
    pre = Preprocessor().fit(tr)
    Xtr, Xva, Xte = pre.transform(tr), pre.transform(va), pre.transform(te)
    ytr, yva, yte = tr[TARGET], va[TARGET], te[TARGET]

    preds = {}
    last_year = te["prev_yield"].fillna(pre.medians["prev_yield"]).values
    preds["Guess: same as last year"] = last_year
    preds["Guess: average of past yields"] = np.full(len(yte), ytr.mean())

    tick("Training linear regression")
    lin = make_linear().fit(Xtr, ytr)
    preds["Linear regression"] = lin.predict(Xte)

    tick("Tuning Random Forest")
    rf, rf_best = tune_random_forest(Xtr, ytr, Xva, yva, seed)
    preds["Random Forest"] = rf.predict(Xte)

    tick("Tuning Gradient Boosting")
    gb, gb_best = tune_gradient_boosting(Xtr, ytr, Xva, yva, seed)
    preds["Gradient Boosting"] = gb.predict(Xte)

    preds["Ensemble (Forest + Boosting)"] = (preds["Random Forest"] + preds["Gradient Boosting"]) / 2

    all_metrics = {k: ev.metrics(yte, v) for k, v in preds.items()}
    models_only = [k for k in all_metrics if not k.startswith("Guess")]
    best_name = min(models_only, key=lambda k: all_metrics[k]["RMSE"])
    # The app always forecasts with Gradient Boosting (used for the range and checks in the report)
    forecast_model, forecast_name = gb, "Gradient Boosting"
    p_gb = preds[forecast_name]

    tick("Checking reliability")
    q_lo, q_hi = ev.interval_from_residuals(yva, gb.predict(Xva))
    cover = ev.coverage(yte, p_gb, q_lo, q_hi)

    # time-series cross-validation over the four seasons ending at the validation year
    cv = []
    X_all = pre.transform(clean)
    for test_year in range(val_year - 3, val_year + 1):
        mtr, mte = clean.year < test_year, clean.year == test_year
        if mtr.sum() < 20 or mte.sum() < 5:
            continue
        mdl = make_gb(gb_best["learning_rate"], gb_best["max_depth"], seed).fit(X_all[mtr], clean.loc[mtr, TARGET])
        m = ev.metrics(clean.loc[mte, TARGET], mdl.predict(X_all[mte]))
        cv.append(dict(season=int(test_year), train_rows=int(mtr.sum()), test_rows=int(mte.sum()), **m))

    # drought-year stress test: driest test season vs the others
    by_year = {}
    test_years = sorted(te.year.unique())
    if len(test_years) >= 2:
        rain = te.groupby("year")["rainfall_mm"].median()
        dry = int(rain.idxmin())
        is_dry = (te.year == dry).values
        for name in ("Gradient Boosting", "Guess: same as last year"):
            by_year[name] = {
                "dry_year": dry,
                "dry": ev.metrics(yte[is_dry], preds[name][is_dry]),
                "other": ev.metrics(yte[~is_dry], preds[name][~is_dry]),
            }

    tick("Finding the most important inputs")
    pi = permutation_importance(gb, Xte, yte, n_repeats=10, random_state=0)
    importance = sorted(zip(Xte.columns, pi.importances_mean), key=lambda t: -t[1])

    best_guess_rmse = min(all_metrics["Guess: same as last year"]["RMSE"], all_metrics["Guess: average of past yields"]["RMSE"])
    gb_m = all_metrics[forecast_name]
    card = ev.report_card(
        gb_m, best_guess_rmse, float(yte.mean()), [c["RMSE"] for c in cv] or [gb_m["RMSE"]],
        by_year.get(forecast_name, {}).get("dry", {}).get("RMSE"),
        by_year.get(forecast_name, {}).get("other", {}).get("RMSE"), cover)

    test_table = te[["farm", "year"]].copy()
    test_table["actual_t_ha"] = yte.values
    test_table["forecast_t_ha"] = p_gb
    test_table["low_t_ha"] = p_gb + q_lo
    test_table["high_t_ha"] = p_gb + q_hi

    return Results(
        quality=quality, clean=clean,
        split_years=dict(train=(int(tr.year.min()), train_end), validation=val_year, test=(val_year + 1, int(te.year.max()))),
        split_sizes=dict(train=len(tr), validation=len(va), test=len(te)),
        medians={k: float(v) for k, v in pre.medians.items()},
        metrics=all_metrics, best_name=best_name,
        tuning=dict(random_forest=rf_best, gradient_boosting=gb_best),
        interval=(q_lo, q_hi), coverage_pct=cover, cv=cv, by_year=by_year,
        importance=importance, card=card, test_table=test_table.reset_index(drop=True),
        forecaster=Forecaster(pre, forecast_model, q_lo, q_hi),
    )
