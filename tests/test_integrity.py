import pandas as pd

from src.data.integrity import (
    check_integrity,
    enforce_monotonic,
    find_duplicate_timestamps,
    find_gaps,
    is_monotonic_increasing,
    reject_duplicates,
)


def _make_hourly_df(timestamps: list[str]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(t, tz="UTC") for t in timestamps])
    return pd.DataFrame({"close": range(len(timestamps))}, index=index)


def test_find_gaps_detects_planted_gap():
    # Hourly series with a 3-hour gap planted between 02:00 and 06:00.
    timestamps = [
        "2024-01-01 00:00",
        "2024-01-01 01:00",
        "2024-01-01 02:00",
        "2024-01-01 06:00",
        "2024-01-01 07:00",
    ]
    df = _make_hourly_df(timestamps)
    gaps = find_gaps(df.index, expected_freq="1h")

    assert len(gaps) == 1
    after, before, n_missing = gaps[0]
    assert n_missing == 3  # 03:00, 04:00, 05:00 missing
    assert str(after) == "2024-01-01 02:00:00+00:00"
    assert str(before) == "2024-01-01 06:00:00+00:00"


def test_find_gaps_no_gap_in_contiguous_series():
    timestamps = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame({"close": range(10)}, index=timestamps)
    gaps = find_gaps(df.index, expected_freq="1h")
    assert gaps == []


def test_duplicate_timestamps_detected_and_rejected():
    timestamps = [
        "2024-01-01 00:00",
        "2024-01-01 01:00",
        "2024-01-01 01:00",  # duplicate
        "2024-01-01 02:00",
    ]
    df = _make_hourly_df(timestamps)

    assert find_duplicate_timestamps(df.index) == 1

    cleaned = reject_duplicates(df)
    assert len(cleaned) == 3
    assert find_duplicate_timestamps(cleaned.index) == 0


def test_monotonicity_enforced():
    timestamps = [
        "2024-01-01 02:00",
        "2024-01-01 00:00",
        "2024-01-01 01:00",
    ]
    df = _make_hourly_df(timestamps)
    assert not is_monotonic_increasing(df.index)

    fixed = enforce_monotonic(df)
    assert is_monotonic_increasing(fixed.index)


def test_check_integrity_full_report():
    timestamps = [
        "2024-01-01 00:00",
        "2024-01-01 01:00",
        "2024-01-01 01:00",  # duplicate
        "2024-01-01 03:00",  # gap: missing 02:00
    ]
    df = _make_hourly_df(timestamps)
    report = check_integrity(df, symbol="BTCUSDT", interval="1h", expected_freq="1h")

    assert report.n_duplicates == 1
    assert report.n_missing_hours == 1
    assert len(report.gaps) == 1

    d = report.to_dict()
    assert d["symbol"] == "BTCUSDT"
    assert d["n_missing_hours"] == 1
