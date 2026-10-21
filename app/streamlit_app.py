"""CropCast: a friendly web app for forecasting crop yield.

Run with:   streamlit run app/streamlit_app.py
"""
import io
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cropcast import pipeline                                    # noqa: E402
from cropcast.config import (BASE_FEATURES, LABELS, RULES, TARGET, pretty_name)  # noqa: E402
from cropcast.data import make_dataset, template_csv               # noqa: E402
from cropcast.validate import apply_rules, check_columns          # noqa: E402

GREEN, GREEN_LIGHT, AMBER, GREY = "#1F5F3A", "#7FB896", "#C9822B", "#8A9490"

st.set_page_config(page_title="CropCast | Crop yield forecasting", page_icon="\U0001F33E", layout="wide")

st.markdown(
    """
<style>
  .block-container {padding-top: 3rem; max-width: 1180px;}
  .hero {background: linear-gradient(135deg,#1F5F3A 0%,#2f7d4f 100%); color: #fff; padding: 1.6rem 1.8rem;
         border-radius: 16px; margin-bottom: 1.2rem;}
  .hero h1 {margin: 0 0 .25rem 0; font-size: 2rem; color:#fff;}
  .hero p {margin: 0; font-size: 1.05rem; opacity: .95;}
  .card {background:#F1F6F3; border:1px solid #DDE8E1; border-radius:14px; padding:1rem 1.2rem; height:100%;}
  .bigresult {background:#F1F6F3; border:2px solid #1F5F3A; border-radius:16px; padding:1.2rem 1.4rem; text-align:center;}
  .bigresult .num {font-size:3rem; font-weight:700; color:#1F5F3A; line-height:1.1;}
  .bigresult .unit {font-size:1.1rem; color:#46514b;}
  .bigresult .range {font-size:1.05rem; margin-top:.4rem;}
  .step {display:flex; gap:.7rem; align-items:flex-start; margin-bottom:.7rem;}
  .step .n {background:#1F5F3A; color:#fff; border-radius:50%; min-width:1.7rem; height:1.7rem; text-align:center;
            line-height:1.7rem; font-weight:700; font-size:.9rem;}
  .pill-pass {background:#E3F2E8; color:#17623a; padding:.15rem .6rem; border-radius:99px; font-weight:600; font-size:.85rem;}
  .pill-fail {background:#FBE9D6; color:#9a5300; padding:.15rem .6rem; border-radius:99px; font-weight:600; font-size:.85rem;}
  .small {font-size:.9rem; color:#55605a;}
  table.rc {width:100%; border-collapse:collapse; margin:.4rem 0 1rem 0;}
  table.rc td, table.rc th {padding:.65rem .5rem; border-bottom:1px solid #E3EAE6; text-align:left; vertical-align:middle;}
  table.rc th {font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; color:#66716b; font-weight:600;}
  table.rc td.res {font-weight:700; white-space:nowrap;}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False)
def sample_data() -> pd.DataFrame:
    return make_dataset(seed=42)


def data_signature(df: pd.DataFrame) -> str:
    return f"{len(df)}-{int(pd.util.hash_pandas_object(df, index=False).sum() % 10**9)}"


def load_data():
    """Return (dataframe, label) from the sidebar choice, or (None, message)."""
    if st.session_state.get("source") == "Upload my own CSV":
        up = st.session_state.get("upload")
        if up is None:
            return None, "Upload a CSV file in the sidebar to continue."
        try:
            df = pd.read_csv(io.BytesIO(up.getvalue()))     # getvalue() can be read again on every rerun
        except Exception:
            return None, "That file could not be read as a CSV. Please check the format."
        missing = check_columns(df)
        if missing:
            return None, "Your file is missing these columns: " + ", ".join(missing)
        return df, f"Your file: {up.name}"
    return sample_data(), "Sample data (simulated)"


def friendly_chart_theme(chart):
    return chart.configure_view(strokeWidth=0).configure_axis(labelFontSize=12, titleFontSize=12, gridColor="#E6ECE8")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### \U0001F33E CropCast")
    st.caption("Forecast crop yield with machine learning.")
    st.markdown("**1. Choose your data**")
    st.radio("Data source", ["Sample data", "Upload my own CSV"], key="source", label_visibility="collapsed")
    if st.session_state.get("source") == "Upload my own CSV":
        st.file_uploader("CSV file", type="csv", key="upload", label_visibility="collapsed")
        st.download_button("Download a CSV template", template_csv(), "cropcast_template.csv", "text/csv",
                           width="stretch")
        st.caption("Needs one row per farm per season, with columns: " + ", ".join(["year", "farm"] + BASE_FEATURES + [TARGET]) + ".")
    else:
        st.caption("A simulated set of 650 farm-seasons (50 farms, 2011 to 2023) with realistic gaps and mistakes added on purpose.")
    st.divider()
    st.markdown("**How it works**")
    for n, t in enumerate(["Check the data", "Train and compare models", "Forecast a season", "Read the report card"], 1):
        st.markdown(f'<div class="step"><div class="n">{n}</div><div>{t}</div></div>', unsafe_allow_html=True)
    st.divider()
    st.caption("Sample results show how the method behaves. They are not the accuracy you would get on a real farm.")

# ------------------------------------------------------------------ header
st.markdown(
    '<div class="hero"><h1>\U0001F33E CropCast</h1>'
    "<p>Know roughly how much your farm will harvest before the season ends, and see what changes it.</p></div>",
    unsafe_allow_html=True,
)

df_raw, source_label = load_data()
if df_raw is None:
    st.info(source_label, icon="\U0001F4C2")
    st.stop()

sig = data_signature(df_raw)
if st.session_state.get("sig") != sig:           # new data: forget old results
    st.session_state["sig"] = sig
    st.session_state.pop("results", None)

tab_data, tab_train, tab_fc, tab_card = st.tabs(
    ["① Your data", "② Train & compare", "③ Forecast", "④ Report card"])

# ================================================================== 1. DATA
with tab_data:
    st.subheader("Is the data healthy?")
    st.write("Before any model sees the data, CropCast checks every value against a sensible range. "
             "Impossible values (like a soil pH of 66) are turned into gaps, and gaps are filled in using typical values.")
    clean_preview, quality = apply_rules(df_raw)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{quality.n_rows:,}")
    c2.metric("Farms", f"{df_raw['farm'].nunique():,}")
    c3.metric("Seasons", f"{df_raw['year'].nunique():,}", f"{int(df_raw['year'].min())} to {int(df_raw['year'].max())}", delta_color="off", delta_arrow="off")
    c4.metric("Rows with a gap", f"{quality.rows_with_gaps:,}", f"{100 * quality.rows_with_gaps / quality.n_rows:.0f}% of rows", delta_color="off", delta_arrow="off")

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("**Gaps by column** (after impossible values were removed)")
        gaps = pd.DataFrame({
            "Column": [pretty_name(c) for c in quality.missing_after],
            "Gaps (%)": [100 * v / quality.n_rows for v in quality.missing_after.values()],
        })
        n_clean_cols = int((gaps["Gaps (%)"] == 0).sum())
        gaps = gaps[gaps["Gaps (%)"] > 0]
        chart = alt.Chart(gaps).mark_bar(color=GREEN, cornerRadiusEnd=4).encode(
            x=alt.X("Gaps (%):Q", title="Share of rows with a gap (%)"),
            y=alt.Y("Column:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            tooltip=["Column", alt.Tooltip("Gaps (%):Q", format=".1f")],
        ).properties(height=max(60, 44 * len(gaps)))
        if gaps.empty:
            st.success("No gaps in any column.", icon="\u2705")
        else:
            st.altair_chart(friendly_chart_theme(chart), width="stretch")
            st.caption(f"The other {n_clean_cols} columns have no gaps.")
    with right:
        st.markdown("**Impossible values caught**")
        viol = pd.DataFrame([
            {"Column": pretty_name(c), "Allowed": f"{RULES[c][0]} to {RULES[c][1]}", "Caught": n}
            for c, n in quality.violations.items() if n > 0
        ])
        if viol.empty:
            st.success("No impossible values found.", icon="✅")
        else:
            st.dataframe(viol, hide_index=True, width="stretch")
            st.caption(f"{int(viol['Caught'].sum())} values were outside their allowed range.")
        if quality.duplicates:
            st.warning(f"{quality.duplicates} duplicate farm/season rows found.")
        st.markdown('<div class="card small"><b>How gaps get filled</b><br>Under 5% missing: typical (median) value. '
                    "5% to 30%: typical value plus a note that it was missing. More than that needs a smarter method "
                    "and a careful look.</div>", unsafe_allow_html=True)

    with st.expander("Look at the raw table"):
        st.dataframe(df_raw.head(200), hide_index=True, width="stretch")
    st.download_button("Download the cleaned data (CSV)", clean_preview.to_csv(index=False), "cropcast_cleaned.csv", "text/csv")

# ================================================================== 2. TRAIN
with tab_train:
    st.subheader("Train the models and see which one forecasts best")
    st.write("CropCast trains a few models and tests them on seasons they have never seen. "
             "It also compares them with two simple guesses, so you can tell whether machine learning is actually worth it.")
    if "results" not in st.session_state:
        go = st.button("▶  Train models", type="primary")
        st.caption("Takes about 10 to 30 seconds.")
        if go:
            error_text = None
            with st.status("Training...", expanded=True) as status:
                try:
                    st.session_state["results"] = pipeline.run(df_raw, progress=lambda m: st.write("\u2022 " + m))
                    status.update(label="Done", state="complete", expanded=False)
                except ValueError as e:
                    error_text = str(e)
                except Exception as e:  # noqa: BLE001
                    error_text = f"Something unexpected went wrong while training: {e}"
                if error_text:
                    status.update(label="Could not train", state="error", expanded=False)
            if error_text:
                st.error(error_text, icon="\U0001F6AB")      # shown outside the box so it cannot be hidden
            else:
                st.rerun()
    res = st.session_state.get("results")
    if res is not None:
        m = res.metrics
        gb = m["Gradient Boosting"]
        base = m["Guess: same as last year"]
        sy = res.split_years
        st.success(f"Trained on seasons {sy['train'][0]} to {sy['train'][1]}, tuned on {sy['validation']}, "
                   f"and tested on {sy['test'][0]} to {sy['test'][1]} ({res.split_sizes['test']} farm-seasons).", icon="✅")
        k1, k2, k3 = st.columns(3)
        k1.metric("Typical miss", f"{gb['MAE']:.2f} t/ha", f"{100 * (1 - gb['RMSE'] / base['RMSE']):.0f}% better than a last-year guess", delta_color="normal")
        k2.metric("Average error", f"{gb['MAPE']:.1f}%", help="Mean absolute percentage error on the test seasons.")
        k3.metric("Ups and downs explained", f"{100 * max(gb['R2'], 0):.0f}%", help="R²: 0% means no better than guessing the average.")

        st.markdown("#### All models side by side")
        table = pd.DataFrame(m).T.reset_index().rename(columns={
            "index": "Model", "MAE": "Typical miss (t/ha)", "RMSE": "RMSE (t/ha)", "MAPE": "Error (%)", "R2": "R²", "Bias": "Bias (t/ha)"})
        table = table[["Model", "Typical miss (t/ha)", "RMSE (t/ha)", "Error (%)", "R²", "Bias (t/ha)"]]
        st.dataframe(
            table, hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.2f") for c in table.columns if c != "Model"})
        st.caption("Lower is better for everything except R². Bias below zero means the model usually guesses too low.")

        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("**Forecast vs what really happened**")
            tt = res.test_table
            lo, hi = float(min(tt.actual_t_ha.min(), tt.forecast_t_ha.min())), float(max(tt.actual_t_ha.max(), tt.forecast_t_ha.max()))
            pts = alt.Chart(tt).mark_circle(size=55, opacity=.6, color=GREEN).encode(
                x=alt.X("actual_t_ha:Q", title="Actual yield (t/ha)", scale=alt.Scale(domain=[lo - .2, hi + .2])),
                y=alt.Y("forecast_t_ha:Q", title="Forecast (t/ha)", scale=alt.Scale(domain=[lo - .2, hi + .2])),
                tooltip=["farm", "year", alt.Tooltip("actual_t_ha:Q", format=".2f"), alt.Tooltip("forecast_t_ha:Q", format=".2f")])
            line = alt.Chart(pd.DataFrame({"x": [lo - .2, hi + .2], "y": [lo - .2, hi + .2]})).mark_line(color=GREY, strokeDash=[5, 4]).encode(x="x:Q", y="y:Q")
            st.altair_chart(friendly_chart_theme((line + pts).properties(height=330)), width="stretch")
            st.caption("Dots close to the dashed line are good forecasts.")
        with b:
            st.markdown("**What drives yield the most**")
            imp = pd.DataFrame(res.importance[:7], columns=["feature", "value"])
            imp["Input"] = imp["feature"].map(pretty_name)
            ch = alt.Chart(imp).mark_bar(color=GREEN, cornerRadiusEnd=4).encode(
                x=alt.X("value:Q", title="How much the error grows if this input is scrambled (t/ha)"),
                y=alt.Y("Input:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
                tooltip=["Input", alt.Tooltip("value:Q", format=".3f")]).properties(height=330)
            st.altair_chart(friendly_chart_theme(ch), width="stretch")
            st.caption("Longer bar = the model relies on it more.")

        with st.expander("Technical details (settings chosen, season-by-season check)"):
            st.write("**Settings chosen on the validation season**")
            st.json(res.tuning)
            st.write("**Retrained on earlier seasons and tested on the next one**")
            cv = pd.DataFrame(res.cv)
            st.dataframe(cv.round(3), hide_index=True, width="stretch")
        if st.button("Train again"):
            st.session_state.pop("results")
            st.rerun()

# ================================================================== 3. FORECAST
with tab_fc:
    st.subheader("Forecast a season")
    res = st.session_state.get("results")
    if res is None:
        st.info("Train the models first (step ②) and this page will come alive.", icon="☝️")
    else:
        fc = res.forecaster
        clean = res.clean
        # slider ranges come from the data itself
        bounds = {}
        for c in BASE_FEATURES:
            s = clean[c].dropna()
            lo_, hi_ = float(s.quantile(0.01)), float(s.quantile(0.99))
            pad = (hi_ - lo_) * 0.1
            rlo, rhi = RULES.get(c, (-1e9, 1e9))
            bounds[c] = (max(rlo, lo_ - pad), min(rhi, hi_ + pad), float(s.median()))
        medians = {c: bounds[c][2] for c in BASE_FEATURES}

        presets = {
            "Typical season": medians,
            "Drought season": {**medians, "rainfall_mm": bounds["rainfall_mm"][0] + 0.15 * (bounds["rainfall_mm"][2] - bounds["rainfall_mm"][0]),
                               "heat_days": min(bounds["heat_days"][1], medians["heat_days"] * 2.2 + 4), "irrigation_mm": 0.0},
            "Wet, mild season": {**medians, "rainfall_mm": min(bounds["rainfall_mm"][1], medians["rainfall_mm"] * 1.25),
                                 "heat_days": max(bounds["heat_days"][0], medians["heat_days"] * 0.4)},
        }

        def apply_preset():
            for k, v in presets[st.session_state["preset"]].items():
                lo_, hi_, _ = bounds[k]
                st.session_state["in_" + k] = float(min(max(v, lo_), hi_))

        for c in BASE_FEATURES:
            st.session_state.setdefault("in_" + c, float(bounds[c][2]))

        form_col, out_col = st.columns([2, 3], gap="large")
        with form_col:
            st.selectbox("Start from", list(presets), key="preset", on_change=apply_preset,
                         help="Pick a starting point, then fine-tune with the sliders.")

            def slider(c, step):
                name, unit, hint = LABELS[c]
                lo_, hi_, _ = bounds[c]
                lab = f"{name} ({unit})" if unit else name
                fmt = "%.0f" if step >= 1 else ("%.2f" if step < 0.1 else "%.1f")
                st.slider(lab, float(round(lo_, 2)), float(round(hi_, 2)), key="in_" + c, step=step, help=hint, format=fmt)

            st.markdown("**☁️ Weather**")
            slider("rainfall_mm", 10.0); slider("avg_temp_c", 0.1); slider("heat_days", 1.0)
            st.markdown("**\U0001F331 Soil and crop**")
            slider("soil_ph", 0.1); slider("soil_oc_pct", 0.05); slider("ndvi_peak", 0.01)
            st.markdown("**\U0001F69C What you control**")
            slider("nitrogen_kg_ha", 5.0); slider("irrigation_mm", 10.0)
            st.markdown("**\U0001F4C5 History**")
            slider("prev_yield", 0.1)

        current = {c: float(st.session_state["in_" + c]) for c in BASE_FEATURES}
        out = fc.predict(pd.DataFrame([current])).iloc[0]
        avg_hist = float(clean[TARGET].mean())
        diff = out.forecast_t_ha - avg_hist

        with out_col:
            st.markdown(
                f'<div class="bigresult"><div class="unit">Expected yield</div>'
                f'<div class="num">{out.forecast_t_ha:.1f}<span class="unit"> t/ha</span></div>'
                f'<div class="range">Most likely between <b>{out.low_t_ha:.1f}</b> and <b>{out.high_t_ha:.1f}</b> t/ha</div></div>',
                unsafe_allow_html=True)
            st.write("")
            r1, r2 = st.columns(2)
            r1.metric("Versus the average season", f"{diff:+.2f} t/ha", f"{100 * diff / avg_hist:+.0f}%")
            r2.metric("Spread of the range", f"±{(out.high_t_ha - out.low_t_ha) / 2:.2f} t/ha",
                      help="The range is about 90% likely to contain the real yield, based on past forecast errors.")

            st.markdown("#### What if I change one thing?")
            what = st.selectbox("Change", ["nitrogen_kg_ha", "irrigation_mm", "rainfall_mm", "heat_days"],
                                format_func=lambda c: pretty_name(c), label_visibility="collapsed")
            lo_, hi_, _ = bounds[what]
            grid = np.linspace(lo_, hi_, 25)
            wi = fc.what_if(current, what, grid)
            wi_long = wi.rename(columns={what: "x"})
            band = alt.Chart(wi_long).mark_area(opacity=.18, color=GREEN).encode(x="x:Q", y="low_t_ha:Q", y2="high_t_ha:Q")
            ln = alt.Chart(wi_long).mark_line(color=GREEN, strokeWidth=3).encode(
                x=alt.X("x:Q", title=f"{pretty_name(what)} ({LABELS[what][1]})"),
                y=alt.Y("forecast_t_ha:Q", title="Forecast yield (t/ha)", scale=alt.Scale(zero=False)))
            now = alt.Chart(pd.DataFrame({"x": [current[what]], "y": [out.forecast_t_ha]})).mark_point(
                size=170, filled=True, color=AMBER).encode(x="x:Q", y="y:Q", tooltip=[alt.Tooltip("x:Q", title="Your setting"), alt.Tooltip("y:Q", title="Forecast", format=".2f")])
            st.altair_chart(friendly_chart_theme((band + ln + now).properties(height=270)), width="stretch")
            if what == "nitrogen_kg_ha":
                step = 20.0
                hi_val = min(current[what] + step, hi_)
                up = float(fc.what_if(current, what, [hi_val])["forecast_t_ha"].iloc[0] - out.forecast_t_ha)
                st.caption(f"Adding {hi_val - current[what]:.0f} kg/ha of nitrogen changes the forecast by about {up:+.2f} t/ha. "
                           "The curve flattens as you add more, so there is a point where extra fertiliser stops paying for itself. "
                           "The orange dot is your current setting and the shaded band is the likely range.")
            else:
                st.caption("The orange dot is your current setting and the shaded band is the likely range.")
            st.caption("Small steps in the line are normal: the model is built from decision trees, which change in steps rather than smooth curves.")

        with st.expander("Forecast many farms at once (upload a CSV)"):
            st.write("Upload a CSV with the input columns (see the template in the sidebar). The target column is not needed.")
            batch = st.file_uploader("Inputs CSV", type="csv", key="batch", label_visibility="collapsed")
            if batch is not None:
                try:
                    bdf = pd.read_csv(io.BytesIO(batch.getvalue()))
                    miss = [c for c in BASE_FEATURES if c not in bdf.columns]
                    if miss:
                        st.error("Missing columns: " + ", ".join(miss))
                    else:
                        pred = fc.predict(bdf)
                        outdf = pd.concat([bdf.reset_index(drop=True), pred.reset_index(drop=True).round(2)], axis=1)
                        st.dataframe(outdf, hide_index=True, width="stretch")
                        st.download_button("Download forecasts (CSV)", outdf.to_csv(index=False), "cropcast_forecasts.csv", "text/csv")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not process that file: {e}")

# ================================================================== 4. REPORT CARD
with tab_card:
    st.subheader("Report card: is this model good enough to trust?")
    res = st.session_state.get("results")
    if res is None:
        st.info("Train the models first (step ②) to see the report card.", icon="☝️")
    else:
        passed = sum(r["passed"] for r in res.card)
        total = len(res.card)
        st.write("The pass marks were decided **before** looking at the test results, so the model cannot be marked on a curve.")
        st.progress(passed / total, text=f"{passed} of {total} checks passed")
        rows_html = "".join(
            f"<tr><td><b>{r['check']}</b></td><td class='small'>needs {r['required']}</td><td class='res'>{r['result']}</td>"
            f"<td><span class='pill-{'pass' if r['passed'] else 'fail'}'>{'\u2705 Pass' if r['passed'] else '\u26A0\uFE0F Fix'}</span></td></tr>"
            for r in res.card)
        st.markdown("<table class='rc'><tr><th>Check</th><th>Pass mark</th><th>Result</th><th>Status</th></tr>" + rows_html + "</table>",
                    unsafe_allow_html=True)
        failed = [r for r in res.card if not r["passed"]]
        if failed:
            st.warning("**What to do about the checks marked Fix:** " + "; ".join(r["check"].lower() for r in failed) +
                       ". For a model that guesses too low, the usual cause is a rising yield trend that older training seasons cannot "
                       "anticipate. Retraining every season, or predicting each farm's change from its own average, tends to help.", icon="\U0001F6E0️")
        else:
            st.success("All checks passed. Keep watching accuracy after every harvest.", icon="\U0001F389")
        if res.by_year:
            d = res.by_year["Gradient Boosting"]
            g = res.by_year["Guess: same as last year"]
            st.markdown(f"#### Drought test ({d['dry_year']})")
            st.write(f"In the driest test season the model's error was **{d['dry']['RMSE']:.2f} t/ha** versus "
                     f"**{d['other']['RMSE']:.2f} t/ha** in the other season. Guessing last year's yield gave "
                     f"{g['dry']['RMSE']:.2f} and {g['other']['RMSE']:.2f} t/ha.")
        st.download_button("Download test-set forecasts (CSV)", res.test_table.round(3).to_csv(index=False),
                           "cropcast_test_forecasts.csv", "text/csv")
