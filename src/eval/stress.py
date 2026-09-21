"""Calm vs. stress regime split, and per-regime metric recomputation.

Stress days are defined as the top decile (90th percentile and above) of
realized volatility. A model that looks good on average but falls apart
exactly on the days markets are turbulent is the known failure mode this
phase exists to catch -- the average alone hides it.
"""

from __future__ import annotations

import pandas as pd

from src.eval.metrics import mae, mean_qlike

STRESS_PERCENTILE = 0.90


def stress_threshold(realized_vol: pd.Series, percentile: float = STRESS_PERCENTILE) -> float:
    return float(realized_vol.quantile(percentile))


def split_calm_stress(
    realized_vol: pd.Series, percentile: float = STRESS_PERCENTILE
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Returns (calm_dates, stress_dates) -- stress is the top decile of RV."""
    threshold = stress_threshold(realized_vol, percentile)
    stress_dates = realized_vol.index[realized_vol >= threshold]
    calm_dates = realized_vol.index[realized_vol < threshold]
    return calm_dates, stress_dates


def score_on_dates(
    realized_vol: pd.Series, forecast_vol: pd.Series, dates: pd.DatetimeIndex
) -> dict:
    rv_subset = realized_vol.loc[realized_vol.index.isin(dates)]
    forecast_subset = forecast_vol.loc[forecast_vol.index.isin(dates)]
    rv_aligned, forecast_aligned = rv_subset.align(forecast_subset, join="inner")

    if len(rv_aligned) == 0:
        return {"n_obs": 0, "qlike": None, "mae": None}

    rv_var = rv_aligned**2
    forecast_var = forecast_aligned**2

    return {
        "n_obs": int(len(rv_aligned)),
        "qlike": mean_qlike(rv_var, forecast_var),
        "mae": mae(rv_aligned, forecast_aligned),
    }


def calm_stress_report(realized_vol: pd.Series, forecast_vol: pd.Series) -> dict:
    calm_dates, stress_dates = split_calm_stress(realized_vol)
    return {
        "stress_threshold": stress_threshold(realized_vol),
        "calm": score_on_dates(realized_vol, forecast_vol, calm_dates),
        "stress": score_on_dates(realized_vol, forecast_vol, stress_dates),
    }
