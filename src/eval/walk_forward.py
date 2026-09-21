"""Rolling-origin walk-forward splitter -- the spine of the repo.

Simulates how a forecaster would actually run in production: fit on a
12-month trailing window, forecast the next day, slide the origin forward by
one day, repeat. The model never sees a day's data until every prior day has
already been "lived through." This is the opposite of a random train/test
split, which would leak future information into training for time series.

Refit cadence is configurable per model because refitting a neural network
daily on CPU is not realistic. That tradeoff is made explicit here rather
than hidden inside a model file.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    origin: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # inclusive; origin's forecast may only use data up to here
    forecast_date: pd.Timestamp
    should_refit: bool


def walk_forward_splits(
    dates: pd.DatetimeIndex,
    window_days: int = 365,
    refit_every_n_days: int = 1,
) -> Iterator[WalkForwardSplit]:
    """Generate rolling-origin splits over a sorted DatetimeIndex of daily dates.

    For each origin day i (starting once a full window is available):
      - train window = [dates[i - window_days], dates[i - 1]]
      - forecast target = dates[i]
    `should_refit` is True every `refit_every_n_days` steps (0-indexed from
    the first split), so a caller can skip refitting an expensive model on
    days when should_refit is False while still forecasting every day.
    """
    sorted_dates = dates.sort_values().unique()

    if len(sorted_dates) <= window_days:
        return

    for step, i in enumerate(range(window_days, len(sorted_dates))):
        train_start = sorted_dates[i - window_days]
        train_end = sorted_dates[i - 1]
        forecast_date = sorted_dates[i]

        yield WalkForwardSplit(
            origin=forecast_date,
            train_start=pd.Timestamp(train_start),
            train_end=pd.Timestamp(train_end),
            forecast_date=pd.Timestamp(forecast_date),
            should_refit=(step % refit_every_n_days == 0),
        )


def train_slice(series: pd.Series, split: WalkForwardSplit) -> pd.Series:
    """Return the portion of `series` visible to a model at this split's origin.

    Strictly excludes `forecast_date` and anything after it -- this is the
    no-lookahead boundary every model must respect.
    """
    return series.loc[split.train_start : split.train_end]
