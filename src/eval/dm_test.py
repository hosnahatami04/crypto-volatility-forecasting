"""Diebold-Mariano test for equal predictive accuracy, with the
small-sample (Harvey-Leybourne-Newbold) correction.

Given two forecasters scored by the same loss function on the same dates,
the DM test asks: is the average difference in their daily losses
statistically distinguishable from zero, or could it be sampling noise?

    d_t = loss_1(t) - loss_2(t)        (daily loss differential)
    DM  = mean(d) / sqrt(Var(mean(d)))

The variance of the mean uses a Newey-West / HAC long-run variance estimate
(loss differentials for overlapping-horizon forecasts are typically
autocorrelated, and ignoring that understates the true variance and
overstates significance). The Harvey-Leybourne-Newbold (1997) correction
rescales the raw DM statistic to better match a Student's t distribution in
small samples, which daily QLIKE differentials over a ~365-day walk-forward
test period firmly are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DMTestResult:
    dm_statistic: float
    p_value: float
    n_obs: int
    mean_loss_diff: float  # negative means model 1 has lower (better) average loss

    def to_dict(self) -> dict:
        return {
            "dm_statistic": self.dm_statistic,
            "p_value": self.p_value,
            "n_obs": self.n_obs,
            "mean_loss_diff": self.mean_loss_diff,
            "significant_at_5pct": self.p_value < 0.05,
        }


def _newey_west_long_run_variance(d: np.ndarray, max_lag: int) -> float:
    """Newey-West HAC estimator of the long-run variance of the mean of `d`."""
    n = len(d)
    d_centered = d - d.mean()
    gamma_0 = np.dot(d_centered, d_centered) / n
    long_run_var = gamma_0

    for lag in range(1, max_lag + 1):
        gamma_lag = np.dot(d_centered[lag:], d_centered[:-lag]) / n
        weight = 1 - lag / (max_lag + 1)
        long_run_var += 2 * weight * gamma_lag

    return long_run_var


def diebold_mariano_test(
    loss_1: pd.Series, loss_2: pd.Series, max_lag: int | None = None
) -> DMTestResult:
    """Test whether loss_1 and loss_2 (e.g. daily QLIKE for two forecasters,
    aligned on the same dates) have equal expected loss.

    A negative dm_statistic with a small p-value means loss_1 is
    significantly LOWER (better) than loss_2 on average.
    """
    aligned_1, aligned_2 = loss_1.align(loss_2, join="inner")
    d = (aligned_1 - aligned_2).values
    n = len(d)

    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))  # standard HAC lag rule

    mean_d = d.mean()
    long_run_var = _newey_west_long_run_variance(d, max_lag)
    dm_raw = mean_d / np.sqrt(long_run_var / n)

    # Harvey-Leybourne-Newbold (1997) small-sample correction.
    h = 1  # one-step-ahead forecast horizon
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_corrected = dm_raw * correction

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_corrected), df=n - 1))

    return DMTestResult(
        dm_statistic=float(dm_corrected),
        p_value=float(p_value),
        n_obs=n,
        mean_loss_diff=float(mean_d),
    )
