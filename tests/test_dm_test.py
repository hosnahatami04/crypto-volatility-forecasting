import numpy as np
import pandas as pd
from scipy import stats

from src.eval.dm_test import diebold_mariano_test


def test_dm_detects_clearly_better_model():
    rng = np.random.default_rng(0)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n)

    loss_1 = pd.Series(rng.normal(0.3, 0.1, n), index=dates)
    loss_2 = pd.Series(rng.normal(0.5, 0.1, n), index=dates)

    result = diebold_mariano_test(loss_1, loss_2)

    assert result.dm_statistic < 0  # model 1 has lower loss
    assert result.p_value < 0.05
    assert result.mean_loss_diff < 0


def test_dm_no_significance_when_no_real_difference():
    rng = np.random.default_rng(1)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n)

    loss_1 = pd.Series(rng.normal(0.4, 0.1, n), index=dates)
    loss_2 = pd.Series(rng.normal(0.4, 0.1, n), index=dates)

    result = diebold_mariano_test(loss_1, loss_2)

    assert result.p_value > 0.05


def test_dm_symmetric_under_swap():
    rng = np.random.default_rng(2)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n)
    loss_1 = pd.Series(rng.normal(0.3, 0.1, n), index=dates)
    loss_2 = pd.Series(rng.normal(0.5, 0.1, n), index=dates)

    result_forward = diebold_mariano_test(loss_1, loss_2)
    result_swapped = diebold_mariano_test(loss_2, loss_1)

    assert result_forward.dm_statistic == -result_swapped.dm_statistic
    assert result_forward.p_value == result_swapped.p_value


def test_dm_aligns_on_index():
    dates_1 = pd.date_range("2024-01-01", periods=10)
    dates_2 = pd.date_range("2024-01-05", periods=10)
    loss_1 = pd.Series(np.ones(10) * 0.3, index=dates_1)
    loss_2 = pd.Series(np.ones(10) * 0.5, index=dates_2)

    result = diebold_mariano_test(loss_1, loss_2)
    # Overlap is 2024-01-05 through 2024-01-10 -> 6 days
    assert result.n_obs == 6


def test_dm_matches_plain_ttest_when_no_autocorrelation():
    """Verification against a reference implementation: for i.i.d. (zero
    autocorrelation) loss differentials, the DM statistic before the small-
    sample correction should closely match a standard paired t-test
    statistic, since the Newey-West long-run variance collapses toward the
    simple sample variance when there's no serial correlation to correct for.
    """
    rng = np.random.default_rng(3)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n)
    loss_1 = pd.Series(rng.normal(0.35, 0.15, n), index=dates)
    loss_2 = pd.Series(rng.normal(0.40, 0.15, n), index=dates)

    dm_result = diebold_mariano_test(loss_1, loss_2, max_lag=0)

    d = (loss_1 - loss_2).values
    ttest_result = stats.ttest_1samp(d, popmean=0)

    # With max_lag=0 (no HAC correction) and a large n (correction factor
    # close to 1), the DM statistic should be very close to the plain
    # t-statistic.
    assert abs(dm_result.dm_statistic - ttest_result.statistic) < 0.05
    assert abs(dm_result.p_value - ttest_result.pvalue) < 0.01


def test_dm_result_to_dict_shape():
    rng = np.random.default_rng(4)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n)
    loss_1 = pd.Series(rng.normal(0.3, 0.1, n), index=dates)
    loss_2 = pd.Series(rng.normal(0.3, 0.1, n), index=dates)

    result = diebold_mariano_test(loss_1, loss_2)
    d = result.to_dict()
    assert set(d.keys()) == {
        "dm_statistic",
        "p_value",
        "n_obs",
        "mean_loss_diff",
        "significant_at_5pct",
    }
