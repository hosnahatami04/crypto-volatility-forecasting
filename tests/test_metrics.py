import numpy as np
import pandas as pd
import pytest

from src.eval.metrics import mae, mean_qlike, qlike


def test_qlike_hand_computed():
    # sigma^2 = 4, h = 2 -> ratio = 2 -> QLIKE = 2 - ln(2) - 1 = 1 - ln(2)
    rv = pd.Series([4.0])
    forecast = pd.Series([2.0])
    result = qlike(rv, forecast)
    expected = 1 - np.log(2)
    np.testing.assert_allclose(result.iloc[0], expected, rtol=1e-10)


def test_qlike_zero_at_perfect_forecast():
    # ratio = 1 -> QLIKE = 1 - ln(1) - 1 = 0
    rv = pd.Series([9.0, 16.0])
    forecast = pd.Series([9.0, 16.0])
    result = qlike(rv, forecast)
    np.testing.assert_allclose(result.values, [0.0, 0.0], atol=1e-12)


def test_qlike_penalizes_underprediction_more_than_overprediction():
    # Same realized variance, forecasts equidistant in ratio-space (0.5x vs 2x).
    underpredict = pd.Series([2.0])  # h = rv/2 -> under-predicts risk
    overpredict = pd.Series([8.0])  # h = rv*2 -> over-predicts risk

    under_loss = mean_qlike(pd.Series([4.0]), underpredict)
    over_loss = mean_qlike(pd.Series([4.0]), overpredict)

    # ratio=2 (underpredict): 2 - ln(2) - 1 = 1 - ln(2) ~ 0.307
    # ratio=0.5 (overpredict): 0.5 - ln(0.5) - 1 = -0.5 + ln(2) ~ 0.193
    assert under_loss > over_loss


def test_mean_qlike_averages_correctly():
    rv = pd.Series([4.0, 9.0])
    forecast = pd.Series([4.0, 9.0])
    assert mean_qlike(rv, forecast) == pytest.approx(0.0, abs=1e-12)


def test_mae_hand_computed():
    realized = pd.Series([0.10, 0.20, 0.15])
    forecast = pd.Series([0.12, 0.18, 0.15])
    result = mae(realized, forecast)
    expected = (0.02 + 0.02 + 0.0) / 3
    assert result == pytest.approx(expected, rel=1e-10)


def test_metrics_align_on_index():
    rv = pd.Series([4.0, 9.0, 16.0], index=[0, 1, 2])
    forecast = pd.Series([4.0, 9.0], index=[1, 2])  # missing index 0
    result = qlike(rv, forecast)
    assert len(result) == 2
