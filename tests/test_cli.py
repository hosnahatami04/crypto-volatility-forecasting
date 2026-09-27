import pandas as pd
import pytest

from src.cli import PAIR_TO_SYMBOL, _ordinal, regime_percentile


def test_pair_to_symbol_mapping():
    assert PAIR_TO_SYMBOL["BTC"] == "BTCUSDT"
    assert PAIR_TO_SYMBOL["ETH"] == "ETHUSDT"


def test_ordinal_basic_suffixes():
    assert _ordinal(1) == "1st"
    assert _ordinal(2) == "2nd"
    assert _ordinal(3) == "3rd"
    assert _ordinal(4) == "4th"
    assert _ordinal(62) == "62nd"  # the case from the reported CLI bug
    assert _ordinal(21) == "21st"


def test_ordinal_teens_are_all_th():
    # 11th, 12th, 13th are special -- not 11st/12nd/13rd.
    assert _ordinal(11) == "11th"
    assert _ordinal(12) == "12th"
    assert _ordinal(13) == "13th"
    assert _ordinal(111) == "111th"


def test_regime_percentile_hand_computed():
    dates = pd.date_range("2024-01-01", periods=10)
    historical_rv = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=dates)

    # A forecast of 5.5 is greater than exactly 5 of the 10 historical values -> 50th percentile
    result = regime_percentile(5.5, historical_rv)
    assert result == pytest.approx(50.0)


def test_regime_percentile_below_everything():
    dates = pd.date_range("2024-01-01", periods=5)
    historical_rv = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates)
    result = regime_percentile(0.5, historical_rv)
    assert result == pytest.approx(0.0)


def test_regime_percentile_above_everything():
    dates = pd.date_range("2024-01-01", periods=5)
    historical_rv = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates)
    result = regime_percentile(10.0, historical_rv)
    assert result == pytest.approx(100.0)


def test_regime_percentile_only_uses_trailing_year():
    """A value from more than 365 days ago must not affect the percentile."""
    dates = pd.date_range("2020-01-01", periods=400)
    # Ancient huge values, then a full recent year of small ones.
    values = [1000.0] * 35 + [1.0] * 365
    historical_rv = pd.Series(values, index=dates)

    # 500 is far above the recent-year values (all 1.0) but far below the
    # ancient ones (all 1000.0) -- if the ancient values leaked into the
    # trailing window, this would NOT be the 100th percentile.
    result_mid = regime_percentile(500.0, historical_rv)
    assert result_mid == pytest.approx(100.0)
