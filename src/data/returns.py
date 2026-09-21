"""Log returns and stationarity checks.

Why log returns, not raw prices: prices are a random walk with drift -- their
mean and variance change over time, so a model fit on prices "learns" a moving
target. Log returns r_t = ln(P_t / P_{t-1}) are approximately stationary
(constant mean/variance over time), which is what GARCH and the LSTM windows
both assume. The ADF test below is the measured evidence for that claim on our
actual data, rather than a textbook assertion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def log_returns(prices: pd.Series) -> pd.Series:
    """r_t = ln(P_t / P_{t-1}). First value is dropped (no prior price)."""
    return np.log(prices / prices.shift(1)).dropna()


@dataclass
class ADFResult:
    statistic: float
    p_value: float
    n_lags: int
    n_obs: int
    critical_values: dict[str, float]

    def is_stationary(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def to_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n_lags": self.n_lags,
            "n_obs": self.n_obs,
            "critical_values": self.critical_values,
            "is_stationary_at_5pct": self.p_value < 0.05,
        }


def adf_test(series: pd.Series) -> ADFResult:
    """Augmented Dickey-Fuller test. Null hypothesis: series has a unit root
    (is non-stationary). A low p-value rejects the null -> series is stationary.
    """
    clean = series.dropna()
    statistic, p_value, n_lags, n_obs, critical_values, _ = adfuller(clean, autolag="AIC")
    return ADFResult(
        statistic=float(statistic),
        p_value=float(p_value),
        n_lags=int(n_lags),
        n_obs=int(n_obs),
        critical_values={k: float(v) for k, v in critical_values.items()},
    )
