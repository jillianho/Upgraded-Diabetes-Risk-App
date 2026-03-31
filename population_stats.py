"""Population percentile comparisons using NHANES data."""

import pandas as pd
import numpy as np
from pathlib import Path

_NHANES_PATH = Path(__file__).parent / "NHANES.csv"

# Age-group bins: (label, min_age, max_age)
_AGE_BINS = [
    ("18–29", 18, 29),
    ("30–39", 30, 39),
    ("40–49", 40, 49),
    ("50–59", 50, 59),
    ("60–69", 60, 69),
    ("70+",   70, 200),
]


def _load_nhanes() -> pd.DataFrame:
    df = pd.read_csv(_NHANES_PATH)
    return df[df["Age"] >= 18].copy()


def _age_group_label(age: int) -> str:
    for label, lo, hi in _AGE_BINS:
        if lo <= age <= hi:
            return label
    return "70+"


def compute_percentiles(age: int, bmi: float, bp_dia: float, waist_cm: float | None = None) -> dict:
    """Return percentile information for the user's age group.

    Returns a dict with keys like:
        {
            "age_group": "40–49",
            "n_in_group": 1234,
            "bmi":  {"value": 28.5, "percentile": 72, "median": 27.1},
            "bp":   {"value": 85,   "percentile": 60, "median": 78},
            "waist": {"value": 96, "percentile": 55, "median": 94} | None,
        }
    """
    df = _load_nhanes()
    label = _age_group_label(age)

    for lbl, lo, hi in _AGE_BINS:
        if lbl == label:
            subset = df[(df["Age"] >= lo) & (df["Age"] <= hi)]
            break
    else:
        subset = df[df["Age"] >= 70]

    result: dict = {"age_group": label, "n_in_group": len(subset)}

    # BMI
    bmi_vals = subset["BMI"].dropna()
    if len(bmi_vals) > 10:
        pct = float((bmi_vals < bmi).mean() * 100)
        result["bmi"] = {
            "value": round(bmi, 1),
            "percentile": int(round(pct)),
            "median": round(float(bmi_vals.median()), 1),
        }
    else:
        result["bmi"] = None

    # Blood pressure (diastolic)
    bp_vals = subset["BPDiaAve"].dropna()
    if len(bp_vals) > 10:
        pct = float((bp_vals < bp_dia).mean() * 100)
        result["bp"] = {
            "value": round(bp_dia, 1),
            "percentile": int(round(pct)),
            "median": round(float(bp_vals.median()), 1),
        }
    else:
        result["bp"] = None

    # Waist – NHANES doesn't have waist in this extract, so we approximate
    # from BMI distribution as a fallback.  If waist is provided we still
    # report vs BMI-derived proxy distribution.
    if waist_cm is not None:
        # Use weight as rough proxy – not perfect but gives directional sense
        weight_vals = subset["Weight"].dropna()
        if len(weight_vals) > 10:
            # Map waist to a pseudo-percentile using the weight distribution
            # (waist correlates ~0.85 with weight in NHANES literature)
            # Normalise: (waist - 60) / 90 ≈ (weight - min) / range
            w_min, w_max = float(weight_vals.quantile(0.02)), float(weight_vals.quantile(0.98))
            waist_norm = (waist_cm - 60) / 90  # rough 0-1 scale
            synth_weight = w_min + waist_norm * (w_max - w_min)
            pct = float((weight_vals < synth_weight).mean() * 100)
            result["waist"] = {
                "value": round(waist_cm, 1),
                "percentile": int(round(min(max(pct, 1), 99))),
            }
        else:
            result["waist"] = None
    else:
        result["waist"] = None

    return result
