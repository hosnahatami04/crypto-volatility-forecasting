"""Next-24h realized volatility target, built from hourly log returns.

RV_d = sqrt(sum(r_h^2)) over the 24 hourly returns in day d. This is the
ground truth every forecaster is scored against. Building it from hourly
returns (rather than a single daily close-to-close return) is precisely why
hourly data was worth downloading: it captures intraday movement that a daily
close-to-close return would average away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(hourly_log_returns: pd.Series) -> pd.Series:
    """Aggregate hourly log returns into daily realized volatility.

    RV_d = sqrt(sum_{h in d} r_h^2)

    Days with fewer than 24 hourly observations (data gaps, exchange
    outages) are dropped rather than silently scaled -- an incomplete day's
    RV is not comparable to a full day's RV.
    """
    squared = hourly_log_returns**2
    daily = squared.groupby(hourly_log_returns.index.date)

    counts = daily.transform("count")
    complete = hourly_log_returns[counts == 24]

    daily_sum_sq = complete.pow(2).groupby(complete.index.date).sum()
    rv = np.sqrt(daily_sum_sq)
    rv.index = pd.to_datetime(rv.index).tz_localize(hourly_log_returns.index.tz)
    rv.name = "realized_vol"
    return rv
