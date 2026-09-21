"""Hybrid model: GARCH's forecast fed into the LSTM as an extra input feature.

Statistics and deep learning cooperating instead of competing. GARCH's
one-step-ahead variance forecast for day t is appended as a constant extra
channel across all 168 hours of that day's input window. The network can
then learn residual structure GARCH misses (if any exists) rather than
rediscovering volatility clustering from scratch -- Phase 4 already showed
plain LSTM struggles to do that from raw returns alone.

Alignment discipline: the GARCH feature for forecast day t must itself have
been produced using data through t-1 only. It is read directly from Phase 3's
stored walk-forward output (results/garch_forecast_{symbol}.csv), never
recomputed with full-sample luxury -- recomputing it here would risk silently
leaking a GARCH fit that saw more data than the walk-forward protocol allows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.lstm import N_CHANNELS as LSTM_N_CHANNELS

HYBRID_N_CHANNELS = LSTM_N_CHANNELS + 1  # + GARCH forecast channel


def load_garch_forecast_series(path) -> pd.Series:
    series = pd.read_csv(path, index_col="date", parse_dates=["date"])["garch_forecast_vol"]
    series.index = pd.to_datetime(series.index, utc=True)
    return series


def append_garch_channel(
    lstm_channels_window: np.ndarray, garch_forecast_vol: float
) -> np.ndarray:
    """lstm_channels_window: shape (WINDOW_HOURS, LSTM_N_CHANNELS).
    Appends a constant channel equal to garch_forecast_vol across every hour
    of the window, producing shape (WINDOW_HOURS, LSTM_N_CHANNELS + 1).
    """
    window_hours = lstm_channels_window.shape[0]
    garch_channel = np.full((window_hours, 1), garch_forecast_vol, dtype=lstm_channels_window.dtype)
    return np.concatenate([lstm_channels_window, garch_channel], axis=-1)
