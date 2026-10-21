"""Step 1: check the data against sensible physical ranges.

A value outside its range is not deleted. It is turned into a gap (NaN) and then
filled in later, the same way as any other missing value.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import RULES, REQUIRED_COLUMNS, BASE_FEATURES, TARGET


@dataclass
class QualityReport:
    n_rows: int
    missing_before: dict      # column -> number of gaps in the raw file
    violations: dict          # column -> number of impossible values found
    missing_after: dict       # column -> gaps after impossible values became gaps
    rows_with_gaps: int
    duplicates: int
    missing_columns: list


def check_columns(df: pd.DataFrame) -> list:
    """Return the list of required columns that are missing from the file."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def apply_rules(df: pd.DataFrame):
    """Return (cleaned copy, QualityReport). The input is never modified."""
    out = df.copy()
    cols = [c for c in BASE_FEATURES + [TARGET] if c in out.columns]
    missing_before = {c: int(out[c].isna().sum()) for c in cols}

    violations = {}
    for col, (lo, hi) in RULES.items():
        if col not in out.columns:
            continue
        bad = (out[col] < lo) | (out[col] > hi)
        violations[col] = int(bad.sum())
        out.loc[bad, col] = float("nan")

    report = QualityReport(
        n_rows=len(out),
        missing_before=missing_before,
        violations=violations,
        missing_after={c: int(out[c].isna().sum()) for c in cols},
        rows_with_gaps=int(out[BASE_FEATURES].isna().any(axis=1).sum()),
        duplicates=int(out.duplicated(subset=["farm", "year"]).sum()) if {"farm", "year"} <= set(out.columns) else 0,
        missing_columns=check_columns(df),
    )
    return out, report


def explain_violation(col: str) -> str:
    lo, hi = RULES[col]
    return f"{col} should be between {lo} and {hi}"
