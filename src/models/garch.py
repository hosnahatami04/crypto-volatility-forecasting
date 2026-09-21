"""GARCH(1,1) volatility forecaster -- the classical statistical baseline.

GARCH(1,1) models tomorrow's conditional variance as a weighted combination
of a long-run baseline, yesterday's squared shock, and yesterday's variance:

    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

- omega: the baseline variance the process reverts to absent any shocks.
- alpha: how much a single day's shock (squared return) feeds into tomorrow's
  variance -- the "reaction" parameter.
- beta: how much yesterday's variance itself persists into today -- the
  "memory" parameter.

alpha + beta close to 1 is the mathematical signature of volatility
clustering: shocks decay slowly, so a turbulent day is followed by more
turbulent days. This is measured on our own data, not assumed.

Daily log returns are used (not hourly) because GARCH is a daily-frequency
model in the literature it comes from, and daily log returns are additive
sums of the hourly log returns already computed in src/data/returns.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from arch import arch_model


@dataclass(frozen=True)
class GarchFitResult:
    omega: float
    alpha: float
    beta: float
    next_day_variance: float  # in (log-return-percent)^2 units

    @property
    def persistence(self) -> float:
        """alpha + beta. Close to 1 indicates strong volatility clustering."""
        return self.alpha + self.beta

    def to_dict(self) -> dict:
        return {
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "persistence": self.persistence,
            "next_day_variance": self.next_day_variance,
        }


def daily_log_returns_pct(hourly_log_returns: pd.Series) -> pd.Series:
    """Aggregate hourly log returns into daily log returns, in percent units
    (arch's optimizer is numerically better-behaved on percent-scale returns).
    """
    daily = hourly_log_returns.resample("D").sum() * 100
    return daily.dropna()


def fit_garch_11(train_returns_pct: pd.Series) -> GarchFitResult:
    """Fit GARCH(1,1) on a training window of daily percent log returns and
    produce the one-step-ahead conditional variance forecast.
    """
    model = arch_model(train_returns_pct, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
    fit = model.fit(disp="off")

    omega = float(fit.params["omega"])
    alpha = float(fit.params["alpha[1]"])
    beta = float(fit.params["beta[1]"])

    forecast = fit.forecast(horizon=1, reindex=False)
    next_day_variance = float(forecast.variance.iloc[0, 0])

    return GarchFitResult(omega=omega, alpha=alpha, beta=beta, next_day_variance=next_day_variance)
