"""Run the LSTM through walk-forward validation (monthly refit) on real
BTC/ETH data, score with QLIKE/MAE.

CPU budget: refitting an LSTM daily is not realistic on CPU, so refit
cadence is monthly (~30 days) -- an honest, documented tradeoff. Total
training time is recorded in the output so the cost is visible, not hidden.

The network is trained on log-variance (see src.models.lstm), not raw
volatility: an early version trained directly on volatility produced a
handful of negative forecasts, which QLIKE punishes catastrophically (a
near-zero forecast variance sends sigma^2/h toward infinity). Training on
log-variance and exponentiating back guarantees a positive forecast by
construction, without needing to clip or otherwise post-hoc patch the output.

Run as: python -m src.eval.run_lstm
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.returns import log_returns
from src.eval.metrics import mae, mean_qlike
from src.eval.walk_forward import walk_forward_splits
from src.models.lstm import (
    WINDOW_HOURS,
    WindowScaler,
    build_channels,
    log_variance_target,
    make_windows,
    volatility_from_log_variance,
)
from src.models.lstm_train import predict, train_lstm
from src.target.realized_vol import realized_volatility

SYMBOLS = ("BTCUSDT", "ETHUSDT")
WINDOW_DAYS = 365
REFIT_EVERY_N_DAYS = 30  # monthly refit -- documented CPU-budget tradeoff
SEED = 42

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"
RESULTS_DIR = ROOT / "results"


def load_data(symbol: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_parquet(CACHE_DIR / f"{symbol}_1h.parquet")
    hourly_returns = log_returns(df["close"])
    rv = realized_volatility(hourly_returns)
    return hourly_returns, rv


def run_lstm_walk_forward(hourly_returns: pd.Series, rv: pd.Series) -> dict:
    channels_full = build_channels(hourly_returns.values)
    hourly_index = hourly_returns.index

    splits = list(
        walk_forward_splits(
            rv.index,
            window_days=WINDOW_DAYS,
            refit_every_n_days=REFIT_EVERY_N_DAYS,
        )
    )

    forecasts: dict[pd.Timestamp, float] = {}
    trained_model = None
    trained_scaler = None
    total_train_seconds = 0.0
    n_refits = 0

    for split in splits:
        train_rv = rv.loc[split.train_start : split.train_end]
        train_hour_mask = (hourly_index >= split.train_start) & (
            hourly_index < split.train_end + pd.Timedelta(days=1)
        )
        train_channels = channels_full[train_hour_mask]

        X_train, y_train_rv = make_windows(
            train_channels, train_rv.values, window_hours=WINDOW_HOURS
        )
        y_train = log_variance_target(y_train_rv)

        if split.should_refit or trained_model is None:
            if len(X_train) < 20:
                # Not enough windows yet to train meaningfully; skip this origin.
                continue
            scaler = WindowScaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)

            t0 = time.time()
            result = train_lstm(X_train_scaled, y_train, scaler.mean, scaler.std, seed=SEED)
            total_train_seconds += time.time() - t0
            n_refits += 1

            trained_model = result.model
            trained_scaler = scaler

        # Build the single forecast window for this origin: the 168 hours
        # immediately preceding the forecast_date, using only data through
        # train_end (no lookahead).
        forecast_end_hour = hourly_index.searchsorted(split.forecast_date)
        forecast_start_hour = forecast_end_hour - WINDOW_HOURS
        if forecast_start_hour < 0:
            continue

        window = channels_full[forecast_start_hour:forecast_end_hour]
        if len(window) != WINDOW_HOURS:
            continue

        window_scaled = trained_scaler.transform(window[np.newaxis, ...])
        log_var_pred = predict(trained_model, window_scaled)
        vol_pred = volatility_from_log_variance(log_var_pred)
        forecasts[split.forecast_date] = float(vol_pred[0])

    forecast_series = pd.Series(forecasts).sort_index()
    rv_aligned, forecast_aligned = rv.align(forecast_series, join="inner")

    rv_var = rv_aligned**2
    forecast_var = forecast_aligned**2

    score = {
        "n_obs": int(len(rv_aligned)),
        "qlike": mean_qlike(rv_var, forecast_var),
        "mae": mae(rv_aligned, forecast_aligned),
    }

    return {
        "score": score,
        "n_refits": n_refits,
        "total_train_seconds": round(total_train_seconds, 2),
    }


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for symbol in SYMBOLS:
        print(f"[{symbol}] loading data and training LSTM through walk-forward...")
        hourly_returns, rv = load_data(symbol)

        t_total = time.time()
        result = run_lstm_walk_forward(hourly_returns, rv)
        wall_time = time.time() - t_total

        results[symbol] = result
        score = result["score"]
        print(
            f"[{symbol}] QLIKE={score['qlike']:.4f} MAE={score['mae']:.6f} (n={score['n_obs']}) "
            f"| {result['n_refits']} refits, {result['total_train_seconds']:.1f}s training, "
            f"{wall_time:.1f}s wall total"
        )

    out_path = RESULTS_DIR / "lstm.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
