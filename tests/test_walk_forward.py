import numpy as np
import pandas as pd

from src.eval.walk_forward import train_slice, walk_forward_splits


def _make_daily_index(n_days: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n_days, freq="D")


def test_no_splits_when_insufficient_data():
    dates = _make_daily_index(10)
    splits = list(walk_forward_splits(dates, window_days=365))
    assert splits == []


def test_split_count_matches_available_days():
    dates = _make_daily_index(370)
    splits = list(walk_forward_splits(dates, window_days=365))
    assert len(splits) == 370 - 365


def test_train_window_never_includes_forecast_date():
    dates = _make_daily_index(400)
    for split in walk_forward_splits(dates, window_days=365):
        assert split.train_end < split.forecast_date
        assert split.train_start <= split.train_end


def test_window_slides_forward_by_one_day():
    dates = _make_daily_index(370)
    splits = list(walk_forward_splits(dates, window_days=365))
    for a, b in zip(splits, splits[1:]):
        assert (b.forecast_date - a.forecast_date).days == 1
        assert (b.train_start - a.train_start).days == 1


def test_refit_cadence_flags_correct_steps():
    dates = _make_daily_index(370)
    splits = list(walk_forward_splits(dates, window_days=365, refit_every_n_days=7))
    refit_flags = [s.should_refit for s in splits]
    # Refits on step 0, 7, 14, ... (0-indexed)
    expected = [i % 7 == 0 for i in range(len(splits))]
    assert refit_flags == expected


def test_no_lookahead_property_planted_poison():
    """The credibility test: plant a poisoned future value far in the series
    and assert that no train_slice for an earlier origin can ever see it.
    """
    n_days = 400
    dates = _make_daily_index(n_days)
    values = np.arange(n_days, dtype=float)

    poison_index = n_days - 1
    poisoned_value = 999999.0
    values[poison_index] = poisoned_value
    series = pd.Series(values, index=dates)

    splits = list(walk_forward_splits(dates, window_days=365))

    # Every split except the ones whose forecast_date lands on/after the
    # poisoned day must have a train window that never contains the poison.
    for split in splits:
        if split.forecast_date >= dates[poison_index]:
            continue
        window = train_slice(series, split)
        assert poisoned_value not in window.values, (
            f"Lookahead leak: split forecasting {split.forecast_date} "
            f"saw the poisoned future value"
        )


def test_train_slice_returns_exact_window_bounds():
    dates = _make_daily_index(400)
    series = pd.Series(np.arange(400, dtype=float), index=dates)
    splits = list(walk_forward_splits(dates, window_days=365))

    first_split = splits[0]
    window = train_slice(series, first_split)

    assert window.index.min() == first_split.train_start
    assert window.index.max() == first_split.train_end
    assert len(window) == 365
