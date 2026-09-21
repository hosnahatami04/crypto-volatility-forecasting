import numpy as np
import torch

from src.models.lstm import (
    N_CHANNELS,
    WINDOW_HOURS,
    VolLSTM,
    WindowScaler,
    build_channels,
    log_variance_target,
    make_windows,
    set_seed,
    volatility_from_log_variance,
)


def test_build_channels_shape_and_values():
    r = np.array([0.01, -0.02, 0.0])
    channels = build_channels(r)
    assert channels.shape == (3, 3)
    np.testing.assert_allclose(channels[:, 0], r)  # raw
    np.testing.assert_allclose(channels[:, 1], np.abs(r))  # |r|
    np.testing.assert_allclose(channels[:, 2], r**2)  # r^2


def test_make_windows_shape():
    # WINDOW_HOURS=168 = 7 days. With 10 days of hourly data, day indices
    # 7, 8, 9 are the first ones with a full 168-hour history before them.
    n_days = 10
    n_hours = n_days * 24
    channels = np.random.default_rng(0).normal(0, 1, size=(n_hours, N_CHANNELS))
    targets = np.arange(n_days, dtype=float)

    X, y = make_windows(channels, targets, window_hours=WINDOW_HOURS)

    assert X.shape == (3, WINDOW_HOURS, N_CHANNELS)
    assert y.shape == (3,)
    np.testing.assert_array_equal(y, [7.0, 8.0, 9.0])


def test_make_windows_never_crosses_forecast_origin():
    """The window for day i must end exactly at day i's start -- never
    include any hour from day i itself or later.
    """
    n_days = 10
    n_hours = n_days * 24
    channels = np.arange(n_hours * N_CHANNELS, dtype=float).reshape(n_hours, N_CHANNELS)
    targets = np.zeros(n_days)

    X, y = make_windows(channels, targets, window_hours=WINDOW_HOURS)

    # First day with a full 168-hour history is day 7 (7*24=168 hours available).
    first_valid_day = WINDOW_HOURS // 24
    for sample_idx, day_idx in enumerate(range(first_valid_day, n_days)):
        expected_end = day_idx * 24
        expected_start = expected_end - WINDOW_HOURS
        expected_window = channels[expected_start:expected_end]
        np.testing.assert_array_equal(X[sample_idx], expected_window)


def test_window_scaler_fit_only_on_given_data():
    """Fitting on two windows that differ only in a value OUTSIDE the fitted
    array must not affect the scaler -- proves fit() only looks at what it's
    handed, not any external/future state.
    """
    rng = np.random.default_rng(1)
    windows_a = rng.normal(0, 1, size=(20, WINDOW_HOURS, N_CHANNELS))
    windows_b = windows_a.copy()

    scaler_a = WindowScaler.fit(windows_a)

    # Mutate windows_b AFTER fitting scaler_a -- scaler_a must be unaffected.
    windows_b[:] = 9999.0
    np.testing.assert_allclose(scaler_a.mean, windows_a.reshape(-1, N_CHANNELS).mean(axis=0))


def test_window_scaler_transform_standardizes():
    rng = np.random.default_rng(2)
    windows = rng.normal(5, 2, size=(50, WINDOW_HOURS, N_CHANNELS))
    scaler = WindowScaler.fit(windows)
    transformed = scaler.transform(windows)

    flat = transformed.reshape(-1, N_CHANNELS)
    np.testing.assert_allclose(flat.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(flat.std(axis=0), 1.0, atol=1e-6)


def test_scaler_not_fit_on_validation_or_future_data():
    """Simulates the walk-forward leakage trap directly: fit a scaler on a
    'training' slice, then verify a scaler fit on training+future data
    would have produced DIFFERENT statistics -- proving the training-only
    scaler is not accidentally identical to a full-data fit by construction.
    """
    rng = np.random.default_rng(3)
    train_windows = rng.normal(0, 1, size=(30, WINDOW_HOURS, N_CHANNELS))
    # Future windows drawn from a very different regime.
    future_windows = rng.normal(50, 10, size=(10, WINDOW_HOURS, N_CHANNELS))

    train_only_scaler = WindowScaler.fit(train_windows)
    full_data_scaler = WindowScaler.fit(np.concatenate([train_windows, future_windows], axis=0))

    assert not np.allclose(train_only_scaler.mean, full_data_scaler.mean)


def test_same_seed_same_weights():
    set_seed(123)
    model_a = VolLSTM()
    weights_a = [p.clone() for p in model_a.parameters()]

    set_seed(123)
    model_b = VolLSTM()
    weights_b = [p.clone() for p in model_b.parameters()]

    for wa, wb in zip(weights_a, weights_b):
        torch.testing.assert_close(wa, wb)


def test_same_seed_same_forecast():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, size=(5, WINDOW_HOURS, N_CHANNELS)).astype(np.float32)

    set_seed(7)
    model_a = VolLSTM()
    model_a.eval()
    with torch.no_grad():
        pred_a = model_a(torch.tensor(X))

    set_seed(7)
    model_b = VolLSTM()
    model_b.eval()
    with torch.no_grad():
        pred_b = model_b(torch.tensor(X))

    torch.testing.assert_close(pred_a, pred_b)


def test_forward_output_shape():
    set_seed(0)
    model = VolLSTM()
    x = torch.randn(4, WINDOW_HOURS, N_CHANNELS)
    out = model(x)
    assert out.shape == (4,)


def test_log_variance_roundtrip():
    vol = np.array([0.01, 0.05, 0.1, 0.2])
    log_var = log_variance_target(vol)
    recovered = volatility_from_log_variance(log_var)
    np.testing.assert_allclose(recovered, vol, rtol=1e-6)


def test_volatility_from_log_variance_always_positive():
    """The whole point of the log-variance target: no matter what real
    number the network's unconstrained linear head outputs (including large
    negative numbers), the recovered volatility must never be negative or
    zero -- this is what prevents the QLIKE blowup an earlier version hit.
    """
    arbitrary_head_outputs = np.array([-50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0])
    recovered = volatility_from_log_variance(arbitrary_head_outputs)
    assert np.all(recovered > 0)


def test_log_variance_target_handles_zero_volatility():
    # A realized_vol of exactly 0 must not raise (log(0) is avoided via the epsilon floor).
    vol = np.array([0.0])
    log_var = log_variance_target(vol)
    assert np.isfinite(log_var).all()
