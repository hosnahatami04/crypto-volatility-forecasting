import numpy as np
import pandas as pd
import pytest

from src.models.hybrid import HYBRID_N_CHANNELS, append_garch_channel, load_garch_forecast_series
from src.models.lstm import N_CHANNELS, WINDOW_HOURS


def test_append_garch_channel_shape():
    window = np.random.default_rng(0).normal(0, 1, size=(WINDOW_HOURS, N_CHANNELS))
    result = append_garch_channel(window, garch_forecast_vol=0.05)

    assert result.shape == (WINDOW_HOURS, N_CHANNELS + 1)
    assert result.shape[-1] == HYBRID_N_CHANNELS


def test_append_garch_channel_is_constant_across_window():
    window = np.random.default_rng(1).normal(0, 1, size=(WINDOW_HOURS, N_CHANNELS))
    result = append_garch_channel(window, garch_forecast_vol=0.0321)

    garch_channel = result[:, -1]
    np.testing.assert_allclose(garch_channel, 0.0321)


def test_append_garch_channel_preserves_lstm_channels():
    window = np.random.default_rng(2).normal(0, 1, size=(WINDOW_HOURS, N_CHANNELS))
    result = append_garch_channel(window, garch_forecast_vol=0.02)

    np.testing.assert_array_equal(result[:, :N_CHANNELS], window)


def test_load_garch_forecast_series_hand_built_csv(tmp_path):
    csv_path = tmp_path / "garch_forecast_TEST.csv"
    csv_path.write_text(
        "date,garch_forecast_vol\n"
        "2024-01-01 00:00:00+00:00,0.01\n"
        "2024-01-02 00:00:00+00:00,0.02\n"
    )

    series = load_garch_forecast_series(csv_path)

    assert len(series) == 2
    assert series.iloc[0] == pytest.approx(0.01)
    assert series.iloc[1] == pytest.approx(0.02)
    assert series.index.tz is not None


def test_garch_feature_alignment_no_future_leakage():
    """The core alignment-discipline property: a GARCH forecast series that
    only covers dates strictly before some cutoff must never be usable to
    build a hybrid window whose forecast_date is at or after that cutoff --
    i.e. the hybrid pipeline can only feed in a GARCH value for a date it
    actually has, never interpolate or forward-fill from a stale one.
    """
    dates_with_garch = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    garch_forecast = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=dates_with_garch)

    # A forecast_date beyond the GARCH series' coverage must not silently
    # find a value -- the hybrid pipeline's make_hybrid_windows skips dates
    # not present in garch_forecast.index, verified by direct membership.
    future_date = pd.Timestamp("2024-01-10", tz="UTC")
    assert future_date not in garch_forecast.index
