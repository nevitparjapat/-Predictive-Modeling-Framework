"""Create the illustrative (simulated) farm dataset used in the report.

The data is generated so that it follows simple agronomic rules:
more water and nitrogen raise yield (with diminishing returns), heat stress and
unsuitable soil pH lower it, richer soil helps, and technology adds a little each year.
Realistic problems (gaps and typing mistakes) can be added on purpose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BASE_FEATURES, TARGET


def make_dataset(seed: int = 42, inject_problems: bool = True) -> pd.DataFrame:
    """Return the simulated dataset (650 rows with the default settings)."""
    rng = np.random.default_rng(seed)
    years = list(range(2010, 2024))
    farms = 50

    farm_base = rng.normal(0, 0.35, farms)          # how good each farm is overall
    farm_soc = rng.uniform(0.6, 1.8, farms)
    farm_ph = rng.normal(6.7, 0.5, farms).clip(5.2, 8.0)
    farm_region = rng.integers(0, 3, farms)

    rows = []
    for y in years:
        trend = 0.05 * (y - 2010)                   # technology gain per year
        for f in range(farms):
            rain = rng.normal(620 + 40 * farm_region[f], 130)
            if y in (2015, 2022):                   # two drought years
                rain *= 0.68
            temp = rng.normal(24.5, 1.1)
            heat = max(0, rng.normal(6, 4) + (temp - 24.5) * 3)
            n = rng.choice([60, 80, 100, 120, 140]) + rng.normal(0, 5)
            irr = rng.choice([0, 50, 100, 150])
            ndvi = (0.35 + 0.00045 * (rain + irr) - 0.004 * heat + 0.03 * farm_soc[f]
                    + rng.normal(0, 0.03)).clip(0.2, 0.92)
            water = rain + irr
            w_eff = 1 - np.exp(-water / 450)
            n_eff = 1 - np.exp(-n / 70)
            heat_pen = 0.09 * heat + 0.004 * heat ** 2
            yld = (1.2 + 3.6 * w_eff * n_eff + 0.55 * farm_soc[f] + farm_base[f]
                   + trend - heat_pen - 0.35 * (farm_ph[f] - 6.7) ** 2
                   + 0.9 * (ndvi - 0.6) + rng.normal(0, 0.28))
            rows.append(dict(
                year=y, farm=f, region=int(farm_region[f]), rainfall_mm=rain, avg_temp_c=temp,
                heat_days=heat, soil_ph=farm_ph[f], soil_oc_pct=farm_soc[f], nitrogen_kg_ha=n,
                irrigation_mm=irr, ndvi_peak=ndvi, yield_t_ha=yld))

    df = pd.DataFrame(rows).sort_values(["farm", "year"]).reset_index(drop=True)
    df["prev_yield"] = df.groupby("farm")[TARGET].shift(1)
    df = df.dropna(subset=["prev_yield"]).reset_index(drop=True)   # first season has no "last year"

    if inject_problems:
        n_rows = len(df)

        def inject_missing(col: str, frac: float) -> None:
            idx = rng.choice(n_rows, int(frac * n_rows), replace=False)
            df.loc[idx, col] = np.nan

        inject_missing("ndvi_peak", 0.12)       # cloud cover
        inject_missing("soil_ph", 0.08)         # lab test skipped
        inject_missing("rainfall_mm", 0.05)     # rain gauge failure
        inject_missing("nitrogen_kg_ha", 0.04)

        bad = rng.choice(n_rows, 9, replace=False)   # typing / sensor mistakes
        df.loc[bad[:3], "soil_ph"] = [14.2, 0.0, 66.0]
        df.loc[bad[3:6], "rainfall_mm"] = [-35.0, 4800.0, 9999.0]
        df.loc[bad[6:9], "ndvi_peak"] = [1.4, 1.7, -0.2]

    return df


def template_csv() -> str:
    """A tiny example file users can download and fill in with their own records."""
    cols = ["year", "farm"] + BASE_FEATURES + [TARGET]
    example = [
        [2021, 1, 640, 24.3, 5, 6.6, 1.2, 100, 50, 0.68, 3.4, 3.7],
        [2022, 1, 410, 23.8, 0, 6.5, 1.5, 140, 0, 0.58, 3.5, 4.6],
    ]
    return pd.DataFrame(example, columns=cols).to_csv(index=False)
