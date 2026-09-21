"""Integrity checks on downloaded OHLCV data: gaps, duplicates, monotonicity.

Real exchange data has holes (outages, maintenance windows). We list gaps
explicitly rather than silently interpolating over them, so the gap report can
be committed and reviewed as part of the project's honesty about data quality.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class IntegrityReport:
    symbol: str
    interval: str
    n_rows: int
    n_duplicates: int
    is_monotonic: bool
    gaps: list[tuple[pd.Timestamp, pd.Timestamp, int]]  # (after, before, missing_count)

    @property
    def n_missing_hours(self) -> int:
        return sum(count for _, _, count in self.gaps)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "n_rows": self.n_rows,
            "n_duplicates": self.n_duplicates,
            "is_monotonic": self.is_monotonic,
            "n_gaps": len(self.gaps),
            "n_missing_hours": self.n_missing_hours,
            "gaps": [
                {"after": str(after), "before": str(before), "missing_count": count}
                for after, before, count in self.gaps
            ],
        }


def find_duplicate_timestamps(index: pd.DatetimeIndex) -> int:
    return int(index.duplicated().sum())


def is_monotonic_increasing(index: pd.DatetimeIndex) -> bool:
    return bool(index.is_monotonic_increasing)


def find_gaps(
    index: pd.DatetimeIndex, expected_freq: str
) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Find missing timestamps in a supposedly-regular index.

    Returns a list of (timestamp_before_gap, timestamp_after_gap, n_missing_periods).
    """
    if len(index) < 2:
        return []

    sorted_index = index.sort_values().unique()
    expected_delta = pd.Timedelta(expected_freq)
    gaps = []

    for i in range(1, len(sorted_index)):
        prev_ts = sorted_index[i - 1]
        curr_ts = sorted_index[i]
        delta = curr_ts - prev_ts
        if delta > expected_delta:
            n_missing = int(delta / expected_delta) - 1
            gaps.append((prev_ts, curr_ts, n_missing))

    return gaps


def check_integrity(
    df: pd.DataFrame, symbol: str, interval: str, expected_freq: str
) -> IntegrityReport:
    """Run the full integrity check suite on a downloaded OHLCV dataframe."""
    n_duplicates = find_duplicate_timestamps(df.index)
    monotonic = is_monotonic_increasing(df.index)
    gaps = find_gaps(df.index, expected_freq)

    return IntegrityReport(
        symbol=symbol,
        interval=interval,
        n_rows=len(df),
        n_duplicates=n_duplicates,
        is_monotonic=monotonic,
        gaps=gaps,
    )


def reject_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate-timestamp rows, keeping the first occurrence."""
    return df[~df.index.duplicated(keep="first")]


def enforce_monotonic(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by index to guarantee monotonicity."""
    return df.sort_index()
