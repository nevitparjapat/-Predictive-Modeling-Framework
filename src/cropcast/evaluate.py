"""Step 4: measure how good the forecasts are, in plain numbers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import PASS_MARKS


def metrics(y_true, y_pred) -> dict:
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    return dict(
        RMSE=float(np.sqrt(mean_squared_error(yt, yp))),
        MAE=float(mean_absolute_error(yt, yp)),
        MAPE=float(100 * np.mean(np.abs((yt - yp) / yt))),
        R2=float(r2_score(yt, yp)),
        Bias=float(np.mean(yp - yt)),
    )


def interval_from_residuals(y_val, p_val, lo_q: float = 0.05, hi_q: float = 0.95):
    """Approximate 90% range: 5th and 95th percentile of the validation errors."""
    res = np.asarray(y_val) - np.asarray(p_val)
    return float(np.quantile(res, lo_q)), float(np.quantile(res, hi_q))


def coverage(y_true, p_pred, q_lo, q_hi) -> float:
    yt, yp = np.asarray(y_true), np.asarray(p_pred)
    return float(100 * np.mean((yt >= yp + q_lo) & (yt <= yp + q_hi)))


def report_card(best: dict, best_baseline_rmse: float, mean_yield: float, cv_rmse: list,
                drought_rmse: float | None, normal_rmse: float | None, cover_pct: float) -> list:
    """Compare results with the pass marks. Returns a list of dicts for display."""
    pm = PASS_MARKS
    gain = 100 * (1 - best["RMSE"] / best_baseline_rmse)
    spread = 100 * (max(cv_rmse) - min(cv_rmse)) / (sum(cv_rmse) / len(cv_rmse))
    bias_pct = 100 * abs(best["Bias"]) / mean_yield
    rows = [
        ("Beats the best simple guess", f"at least {pm['rmse_gain_pct']:.0f}% lower error", f"{gain:.1f}% lower", gain >= pm["rmse_gain_pct"]),
        ("Average miss (MAPE)", f"{pm['mape_max']:.0f}% or less", f"{best['MAPE']:.1f}%", best["MAPE"] <= pm["mape_max"]),
        ("Share of ups and downs explained (R²)", f"{pm['r2_min']:.2f} or more", f"{best['R2']:.2f}", best["R2"] >= pm["r2_min"]),
        ("Steady from season to season", f"spread {pm['cv_spread_max_pct']:.0f}% or less", f"{spread:.0f}%", spread <= pm["cv_spread_max_pct"]),
    ]
    if drought_rmse and normal_rmse:
        ratio = drought_rmse / normal_rmse
        rows.append(("Holds up in a drought year", f"no more than {pm['drought_ratio_max']}x the normal-year error", f"{ratio:.2f}x", ratio <= pm["drought_ratio_max"]))
    rows.append(("The 90% range really covers about 90%", f"{pm['coverage_low']:.0f}% to {pm['coverage_high']:.0f}%", f"{cover_pct:.0f}%", pm["coverage_low"] <= cover_pct <= pm["coverage_high"]))
    rows.append(("Does not guess too low or too high", f"within {pm['bias_max_pct']:.0f}% of average yield", f"{bias_pct:.1f}%", bias_pct <= pm["bias_max_pct"]))
    return [dict(check=a, required=b, result=c, passed=bool(d)) for a, b, c, d in rows]
