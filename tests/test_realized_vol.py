import numpy as np
import pandas as pd

from src.target.realized_vol import realized_volatility


def test_realized_vol_hand_computed_single_day():
    # 24 hourly returns, all equal to 0.01 -> RV = sqrt(24 * 0.01^2) = 0.01 * sqrt(24)
    index = pd.date_range("2024-01-01 00:00", periods=24, freq="h", tz="UTC")
    returns = pd.Series([0.01] * 24, index=index)

    rv = realized_volatility(returns)

    assert len(rv) == 1
    expected = 0.01 * np.sqrt(24)
    np.testing.assert_allclose(rv.iloc[0], expected, rtol=1e-10)


def test_incomplete_day_dropped():
    # Only 23 hours on day 1 (incomplete), full 24 on day 2.
    index = pd.date_range("2024-01-01 01:00", periods=23, freq="h", tz="UTC").append(
        pd.date_range("2024-01-02 00:00", periods=24, freq="h", tz="UTC")
    )
    returns = pd.Series([0.01] * len(index), index=index)

    rv = realized_volatility(returns)

    assert len(rv) == 1
    assert pd.Timestamp(rv.index[0]).date() == pd.Timestamp("2024-01-02").date()


def test_realized_vol_two_full_days():
    index = pd.date_range("2024-01-01 00:00", periods=48, freq="h", tz="UTC")
    values = [0.01] * 24 + [0.02] * 24
    returns = pd.Series(values, index=index)

    rv = realized_volatility(returns)

    assert len(rv) == 2
    np.testing.assert_allclose(rv.iloc[0], 0.01 * np.sqrt(24), rtol=1e-10)
    np.testing.assert_allclose(rv.iloc[1], 0.02 * np.sqrt(24), rtol=1e-10)
