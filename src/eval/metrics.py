"""QLIKE and MAE for volatility forecast evaluation, implemented by hand.

QLIKE is the loss function the volatility-forecasting literature actually
uses, not plain MSE. MSE penalizes a large-magnitude forecast error the same
whether volatility is over- or under-predicted. QLIKE is asymmetric: it
punishes UNDER-prediction of variance far more severely than over-prediction.

For a risk system, under-predicting risk is the dangerous failure mode (you
size a position assuming calm markets and get blown up by a spike), so QLIKE
is the metric that actually reflects what the system needs to get right.

QLIKE = sigma^2 / h - ln(sigma^2 / h) - 1

where sigma^2 is realized variance (the "truth") and h is the forecast
variance. As h -> 0 while sigma^2 stays fixed (severe under-prediction),
sigma^2/h -> infinity and -ln(sigma^2/h) -> -infinity, but the ratio term
dominates, so QLIKE -> infinity. As h -> infinity (severe over-prediction),
sigma^2/h -> 0 and -ln(sigma^2/h) -> infinity, but grows only logarithmically
-- much more slowly. That asymmetry is the whole point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def qlike(realized_variance: pd.Series, forecast_variance: pd.Series) -> pd.Series:
    """Per-observation QLIKE loss. Both inputs are VARIANCE, not volatility
    (i.e. already squared). Aligns on index and drops non-overlapping rows.
    """
    aligned_rv, aligned_h = realized_variance.align(forecast_variance, join="inner")
    ratio = aligned_rv / aligned_h
    return ratio - np.log(ratio) - 1


def mean_qlike(realized_variance: pd.Series, forecast_variance: pd.Series) -> float:
    return float(qlike(realized_variance, forecast_variance).mean())


def mae(realized_vol: pd.Series, forecast_vol: pd.Series) -> float:
    """Mean absolute error on VOLATILITY (not variance)."""
    aligned_realized, aligned_forecast = realized_vol.align(forecast_vol, join="inner")
    return float((aligned_realized - aligned_forecast).abs().mean())
