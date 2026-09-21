"""Naive volatility baselines: persistence and rolling mean.

Costs almost nothing to compute, but anchors the whole project: any model
that can't beat "tomorrow's volatility = today's volatility" has learned
nothing. Both baselines are one-line forecasters -- deliberately dumb.
"""

from __future__ import annotations

import pandas as pd


def persistence_forecast(realized_vol: pd.Series) -> pd.Series:
    """Forecast for day t = realized volatility observed on day t-1."""
    return realized_vol.shift(1)


def rolling_mean_forecast(realized_vol: pd.Series, window: int = 7) -> pd.Series:
    """Forecast for day t = mean realized volatility over the trailing `window` days,
    using only data strictly before day t (shifted by 1 to avoid lookahead).
    """
    return realized_vol.shift(1).rolling(window=window, min_periods=window).mean()
