"""LSTM volatility forecaster -- the deep-learning contender.

Deliberately small: this is a small-data regime (a few hundred training
sequences per walk-forward fold), so a bigger network just memorizes noise.
A window of the last 168 hourly returns (one week) is fed through 1-2 LSTM
layers with hidden size 64, plus a linear head.

Two channels beyond the raw hourly log return are included -- |r| and r^2 --
since volatility lives in magnitudes, not signed direction, and giving the
network that transformation directly (rather than making it rediscover
"square this") is a reasonable inductive bias for a small model.

The network's linear head has no positivity constraint, so it must not
predict volatility directly -- an unconstrained regression target that has
to stay positive routinely produces small negative outputs, which is
catastrophic under QLIKE (a forecast variance near zero sends sigma^2/h
toward infinity). Instead the head predicts log(variance); exponentiating
guarantees a strictly positive volatility forecast by construction. See
log_variance_target / variance_from_log_prediction below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

WINDOW_HOURS = 168
N_CHANNELS = 3  # raw return, |r|, r^2
HIDDEN_SIZE = 64
N_LAYERS = 1
SEED = 42


@dataclass(frozen=True)
class WindowScaler:
    """A per-channel standardizer fit ONLY on a training window.

    Fitting a scaler on the full dataset (including validation/test) is the
    most common silent leakage bug in time-series deep learning: the scaler's
    mean/std would then encode information about future volatility regimes
    the model should never have seen at forecast time. This class is fit
    once per walk-forward fold, strictly on that fold's training slice.
    """

    mean: np.ndarray  # shape (N_CHANNELS,)
    std: np.ndarray  # shape (N_CHANNELS,)

    @classmethod
    def fit(cls, windows: np.ndarray) -> "WindowScaler":
        """windows: shape (n_samples, WINDOW_HOURS, N_CHANNELS)."""
        mean = windows.reshape(-1, windows.shape[-1]).mean(axis=0)
        std = windows.reshape(-1, windows.shape[-1]).std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)  # avoid divide-by-zero on constant channels
        return cls(mean=mean, std=std)

    def transform(self, windows: np.ndarray) -> np.ndarray:
        return (windows - self.mean) / self.std


def build_channels(hourly_log_returns: np.ndarray) -> np.ndarray:
    """Expand a 1D array of hourly log returns into (n, N_CHANNELS): [r, |r|, r^2]."""
    r = hourly_log_returns
    return np.stack([r, np.abs(r), r**2], axis=-1)


def make_windows(
    channels: np.ndarray, targets: np.ndarray, window_hours: int = WINDOW_HOURS
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding windows. `channels` is (n_hours, N_CHANNELS) hourly features
    aligned to hourly timestamps; `targets` is (n_days,) daily RV targets whose
    index i's window uses channels strictly before that day's first hour.

    Returns (X, y) where X has shape (n_samples, window_hours, N_CHANNELS) and
    y has shape (n_samples,). A window for target index i uses
    channels[i*24 - window_hours : i*24], i.e. strictly the 168 hours
    immediately preceding day i's start -- never crossing into day i itself.
    """
    n_days = len(targets)
    hours_per_day = 24
    X, y = [], []

    for day_idx in range(n_days):
        end = day_idx * hours_per_day
        start = end - window_hours
        if start < 0:
            continue
        X.append(channels[start:end])
        y.append(targets[day_idx])

    return np.array(X), np.array(y)


class VolLSTM(nn.Module):
    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        hidden_size: int = HIDDEN_SIZE,
        n_layers: int = N_LAYERS,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        return self.head(last_hidden).squeeze(-1)


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


LOG_VARIANCE_EPS = 1e-12  # numerical floor so log(0) never occurs on a zero-RV day


def log_variance_target(realized_vol: np.ndarray) -> np.ndarray:
    """Training target: log(realized_vol^2), i.e. log-variance. Training the
    network on this instead of raw volatility means its unconstrained linear
    output can never map back to a negative or zero volatility forecast.
    """
    return np.log(realized_vol**2 + LOG_VARIANCE_EPS)


def volatility_from_log_variance(log_variance: np.ndarray) -> np.ndarray:
    """Inverse of log_variance_target: recovers a strictly positive
    volatility forecast from the network's raw (unconstrained) output.
    """
    return np.sqrt(np.exp(log_variance))
