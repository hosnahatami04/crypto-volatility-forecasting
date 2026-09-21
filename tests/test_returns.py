import numpy as np
import pandas as pd

from src.data.returns import adf_test, log_returns


def test_log_returns_hand_computed():
    prices = pd.Series([100.0, 110.0, 99.0], index=pd.date_range("2024-01-01", periods=3, freq="h"))
    result = log_returns(prices)

    expected = pd.Series(
        [np.log(110.0 / 100.0), np.log(99.0 / 110.0)],
        index=prices.index[1:],
    )

    assert len(result) == 2
    np.testing.assert_allclose(result.values, expected.values, rtol=1e-10)


def test_log_returns_drops_first_nan():
    prices = pd.Series([50.0, 55.0, 60.0])
    result = log_returns(prices)
    assert len(result) == 2
    assert not result.isna().any()


def test_adf_on_random_walk_vs_stationary_noise():
    rng = np.random.default_rng(42)

    # A random walk (cumulative sum of noise) should NOT be stationary.
    random_walk = pd.Series(np.cumsum(rng.normal(0, 1, 500)))
    walk_result = adf_test(random_walk)

    # White noise IS stationary by construction.
    white_noise = pd.Series(rng.normal(0, 1, 500))
    noise_result = adf_test(white_noise)

    assert not walk_result.is_stationary(alpha=0.05)
    assert noise_result.is_stationary(alpha=0.05)


def test_adf_result_to_dict_shape():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(0, 1, 200))
    result = adf_test(series)
    d = result.to_dict()
    assert "p_value" in d
    assert "critical_values" in d
    assert set(d["critical_values"].keys()) == {"1%", "5%", "10%"}
