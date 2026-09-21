import numpy as np
import pandas as pd
import pytest

from src.eval.stress import (
    calm_stress_report,
    score_on_dates,
    split_calm_stress,
    stress_threshold,
)


def test_stress_threshold_hand_computed():
    rv = pd.Series(np.arange(1, 11, dtype=float))  # 1..10
    threshold = stress_threshold(rv, percentile=0.90)
    # 90th percentile of 1..10 (linear interpolation) = 9.1
    assert threshold == pytest.approx(9.1)


def test_split_calm_stress_top_decile():
    dates = pd.date_range("2024-01-01", periods=10)
    rv = pd.Series(np.arange(1, 11, dtype=float), index=dates)

    calm_dates, stress_dates = split_calm_stress(rv, percentile=0.90)

    # threshold = 9.1 -> only value 10 (index 9) qualifies as stress
    assert len(stress_dates) == 1
    assert stress_dates[0] == dates[9]
    assert len(calm_dates) == 9


def test_score_on_dates_hand_computed():
    dates = pd.date_range("2024-01-01", periods=3)
    rv = pd.Series([2.0, 2.0, 2.0], index=dates)
    forecast = pd.Series([2.0, 2.0, 2.0], index=dates)

    result = score_on_dates(rv, forecast, dates)
    assert result["n_obs"] == 3
    assert result["qlike"] == pytest.approx(0.0, abs=1e-10)
    assert result["mae"] == pytest.approx(0.0, abs=1e-10)


def test_score_on_dates_empty_subset():
    dates = pd.date_range("2024-01-01", periods=3)
    rv = pd.Series([1.0, 2.0, 3.0], index=dates)
    forecast = pd.Series([1.0, 2.0, 3.0], index=dates)

    empty_dates = pd.DatetimeIndex([])
    result = score_on_dates(rv, forecast, empty_dates)
    assert result["n_obs"] == 0
    assert result["qlike"] is None


def test_calm_stress_report_shape():
    dates = pd.date_range("2024-01-01", periods=20)
    rng = np.random.default_rng(0)
    rv = pd.Series(rng.uniform(0.01, 0.05, 20), index=dates)
    forecast = pd.Series(rng.uniform(0.01, 0.05, 20), index=dates)

    report = calm_stress_report(rv, forecast)
    assert set(report.keys()) == {"stress_threshold", "calm", "stress"}
    assert report["calm"]["n_obs"] + report["stress"]["n_obs"] == 20


def test_stress_days_are_the_most_volatile():
    dates = pd.date_range("2024-01-01", periods=100)
    rng = np.random.default_rng(1)
    rv = pd.Series(rng.uniform(0.01, 0.10, 100), index=dates)

    calm_dates, stress_dates = split_calm_stress(rv)

    max_calm_rv = rv.loc[calm_dates].max()
    min_stress_rv = rv.loc[stress_dates].min()
    assert min_stress_rv >= max_calm_rv
