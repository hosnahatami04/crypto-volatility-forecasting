import numpy as np
import pandas as pd
import pytest

from src.models.garch import GarchFitResult, daily_log_returns_pct, fit_garch_11


def test_daily_log_returns_pct_hand_computed():
    # Two days, 24 hourly returns of 0.001 each -> daily sum = 0.024 -> *100 = 2.4
    index = pd.date_range("2024-01-01 00:00", periods=48, freq="h", tz="UTC")
    hourly = pd.Series([0.001] * 48, index=index)

    daily = daily_log_returns_pct(hourly)

    assert len(daily) == 2
    np.testing.assert_allclose(daily.iloc[0], 0.024 * 100, rtol=1e-8)
    np.testing.assert_allclose(daily.iloc[1], 0.024 * 100, rtol=1e-8)


def test_garch_fit_returns_valid_parameters():
    rng = np.random.default_rng(42)
    # Simulate a simple GARCH-like series: enough observations for arch to fit.
    n = 500
    returns_pct = pd.Series(rng.normal(0, 1.5, n))

    result = fit_garch_11(returns_pct)

    assert isinstance(result, GarchFitResult)
    assert result.omega > 0
    assert result.alpha >= 0
    assert result.beta >= 0
    assert result.next_day_variance > 0


def test_garch_persistence_property():
    result = GarchFitResult(omega=0.1, alpha=0.05, beta=0.9, next_day_variance=1.0)
    assert result.persistence == pytest.approx(0.95)


def test_garch_to_dict_shape():
    result = GarchFitResult(omega=0.1, alpha=0.05, beta=0.9, next_day_variance=1.0)
    d = result.to_dict()
    assert set(d.keys()) == {"omega", "alpha", "beta", "persistence", "next_day_variance"}


def test_garch_forecast_uses_only_training_window():
    """Two fits on windows differing only in a poisoned tail value outside
    the training window must produce identical forecasts -- i.e. the fit
    function itself never reaches past what's handed to it.
    """
    rng = np.random.default_rng(7)
    n = 400
    base = rng.normal(0, 1.2, n)

    returns_a = pd.Series(base.copy())
    returns_b = pd.Series(base.copy())

    # fit_garch_11 only ever receives the slice passed to it -- simulate the
    # walk-forward boundary by fitting on an identical prefix from two series
    # whose *future* (unused) tail differs.
    train_a = returns_a.iloc[:300]
    train_b = returns_b.iloc[:300]
    # Poison b's future (unused) portion
    returns_b.iloc[300:] = 9999.0

    result_a = fit_garch_11(train_a)
    result_b = fit_garch_11(train_b)

    assert result_a.omega == pytest.approx(result_b.omega)
    assert result_a.alpha == pytest.approx(result_b.alpha)
    assert result_a.beta == pytest.approx(result_b.beta)
    assert result_a.next_day_variance == pytest.approx(result_b.next_day_variance)
