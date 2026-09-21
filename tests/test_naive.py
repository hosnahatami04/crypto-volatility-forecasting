import numpy as np
import pandas as pd

from src.models.naive import persistence_forecast, rolling_mean_forecast


def test_persistence_forecast_shifts_by_one():
    rv = pd.Series([0.1, 0.2, 0.3, 0.4])
    forecast = persistence_forecast(rv)
    assert np.isnan(forecast.iloc[0])
    assert forecast.iloc[1] == 0.1
    assert forecast.iloc[2] == 0.2
    assert forecast.iloc[3] == 0.3


def test_rolling_mean_forecast_hand_computed():
    rv = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    forecast = rolling_mean_forecast(rv, window=3)

    # Forecast for index 3 (value 4.0) uses shift(1) -> values [1,2,3,4,5,6,7]
    # rolling(3) at position 3 in shifted series = mean of positions 1,2,3 = mean(1,2,3) = 2.0
    assert forecast.iloc[3] == 2.0
    # Forecast for index 4 uses shifted values [1,2,3,4], rolling window of last 3 = mean(2,3,4)
    assert forecast.iloc[4] == 3.0


def test_rolling_mean_forecast_no_lookahead():
    # Plant a poisoned future value; forecast at early indices must not change.
    rv_clean = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    rv_poisoned = pd.Series([1.0, 2.0, 3.0, 4.0, 9999.0])

    forecast_clean = rolling_mean_forecast(rv_clean, window=3)
    forecast_poisoned = rolling_mean_forecast(rv_poisoned, window=3)

    # Forecasts for indices 0-3 only depend on values before index 4, so must be identical.
    pd.testing.assert_series_equal(
        forecast_clean.iloc[:4], forecast_poisoned.iloc[:4]
    )


def test_rolling_mean_insufficient_window_is_nan():
    rv = pd.Series([1.0, 2.0])
    forecast = rolling_mean_forecast(rv, window=7)
    assert forecast.isna().all()
