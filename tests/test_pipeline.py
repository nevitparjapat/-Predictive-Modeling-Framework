import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cropcast import pipeline                                    # noqa: E402
from cropcast.config import BASE_FEATURES, RULES, TARGET          # noqa: E402
from cropcast.data import make_dataset, template_csv              # noqa: E402
from cropcast.evaluate import metrics                             # noqa: E402
from cropcast.preprocess import Preprocessor, split_by_year       # noqa: E402
from cropcast.validate import apply_rules, check_columns          # noqa: E402


@pytest.fixture(scope="module")
def raw():
    return make_dataset(seed=42)


@pytest.fixture(scope="module")
def results(raw):
    return pipeline.run(raw)


# ---------- data ----------
def test_dataset_shape_and_reproducible(raw):
    assert raw.shape[0] == 650 and raw.farm.nunique() == 50
    pd.testing.assert_frame_equal(raw, make_dataset(seed=42))


def test_template_has_required_columns():
    df = pd.read_csv(pd.io.common.StringIO(template_csv()))
    assert check_columns(df) == []


# ---------- validation ----------
def test_all_nine_planted_mistakes_are_caught(raw):
    clean, rep = apply_rules(raw)
    assert sum(rep.violations.values()) == 9
    for col, (lo, hi) in RULES.items():
        s = clean[col].dropna()
        assert s.between(lo, hi).all(), col


def test_apply_rules_does_not_change_input(raw):
    before = raw.copy()
    apply_rules(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_missing_columns_are_reported(raw):
    assert check_columns(raw.drop(columns=["soil_ph"])) == ["soil_ph"]


# ---------- preprocessing ----------
def test_medians_come_from_training_rows_only(raw):
    clean, _ = apply_rules(raw)
    train, val, test = split_by_year(clean)
    pre = Preprocessor().fit(train)
    assert pre.medians["rainfall_mm"] == pytest.approx(train["rainfall_mm"].median())
    assert pre.medians["rainfall_mm"] != pytest.approx(clean["rainfall_mm"].median())


def test_no_gaps_left_and_flags_added(raw):
    clean, _ = apply_rules(raw)
    train, _, test = split_by_year(clean)
    X = Preprocessor().fit(train).transform(test)
    assert not X.isna().any().any()
    assert "ndvi_peak_was_missing" in X.columns and "water_total_mm" in X.columns


def test_split_is_chronological(raw):
    clean, _ = apply_rules(raw)
    train, val, test = split_by_year(clean)
    assert train.year.max() < val.year.min() <= val.year.max() < test.year.min()


# ---------- metrics ----------
def test_metrics_match_hand_calculation():
    m = metrics([4.0, 3.0, 5.0], [3.5, 3.5, 4.0])
    assert m["MAE"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt((0.25 + 0.25 + 1.0) / 3))
    assert m["Bias"] == pytest.approx((-0.5 + 0.5 - 1.0) / 3)


# ---------- full pipeline (numbers quoted in the report) ----------
def test_reproduces_report_numbers(results):
    gb = results.metrics["Gradient Boosting"]
    assert gb["RMSE"] == pytest.approx(0.500, abs=0.002)
    assert gb["MAPE"] == pytest.approx(11.4, abs=0.1)
    assert gb["R2"] == pytest.approx(0.694, abs=0.003)
    assert results.split_sizes == dict(train=500, validation=50, test=100)


def test_models_beat_simple_guesses(results):
    m = results.metrics
    best_guess = min(m["Guess: same as last year"]["RMSE"], m["Guess: average of past yields"]["RMSE"])
    for name in ("Linear regression", "Random Forest", "Gradient Boosting"):
        assert m[name]["RMSE"] < 0.7 * best_guess


def test_report_card_flags_the_known_bias(results):
    failed = [r["check"] for r in results.card if not r["passed"]]
    assert failed == ["Does not guess too low or too high"]


# ---------- forecasting ----------
def test_forecast_has_sensible_range(results):
    fc = results.forecaster
    row = pd.DataFrame([results.medians])
    out = fc.predict(row).iloc[0]
    assert out.low_t_ha < out.forecast_t_ha < out.high_t_ha
    assert 1.0 < out.forecast_t_ha < 8.0


def test_forecast_survives_gaps_and_impossible_values(results):
    row = pd.DataFrame([results.medians])
    row.loc[0, "soil_ph"] = 66          # impossible -> treated as a gap
    row.loc[0, "ndvi_peak"] = np.nan    # real gap
    out = results.forecaster.predict(row)
    assert out.notna().all().all()


def test_drought_forecast_is_lower_than_typical(results):
    typical = dict(results.medians)
    drought = {**typical, "rainfall_mm": 330.0, "heat_days": 16.0, "irrigation_mm": 0.0}
    f = results.forecaster
    assert f.predict(pd.DataFrame([drought])).forecast_t_ha[0] < f.predict(pd.DataFrame([typical])).forecast_t_ha[0] - 0.5


def test_more_nitrogen_does_not_lower_yield_much(results):
    wi = results.forecaster.what_if(dict(results.medians), "nitrogen_kg_ha", [40, 80, 120, 160])
    assert wi.forecast_t_ha.iloc[-1] > wi.forecast_t_ha.iloc[0]


def test_too_few_seasons_gives_clear_error(raw):
    short = raw[raw.year >= 2019]
    with pytest.raises(ValueError, match="at least"):
        pipeline.run(short)
