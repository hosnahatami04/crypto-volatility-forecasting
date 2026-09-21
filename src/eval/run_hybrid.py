"""Run the hybrid model (GARCH forecast + LSTM) through walk-forward
validation on real BTC/ETH data, score with QLIKE/MAE.

Requires results/garch_forecast_{symbol}.csv to already exist (produced by
src/eval/run_garch.py) -- the GARCH feature is read from that file, never
recomputed here, per the alignment discipline documented in src/models/hybrid.py.

Run as: python -m src.eval.run_hybrid
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
from src.models.hybrid import append_garch_channel, load_garch_forecast_series
from src.models.lstm import (
    WINDOW_HOURS,
    WindowScaler,
    build_channels,
    log_variance_target,
    volatility_from_log_variance,
)
from src.models.lstm_train import predict, train_lstm
from src.target.realized_vol import realized_volatility

SYMBOLS = ("BTCUSDT", "ETHUSDT")
WINDOW_DAYS = 365
REFIT_EVERY_N_DAYS = 30  # same monthly cadence as the plain LSTM, for a fair comparison
SEED = 42

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"
RESULTS_DIR = ROOT / "results"


def load_data(symbol: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    df = pd.read_parquet(CACHE_DIR / f"{symbol}_1h.parquet")
    hourly_returns = log_returns(df["close"])
    rv = realized_volatility(hourly_returns)
    garch_forecast = load_garch_forecast_series(RESULTS_DIR / f"garch_forecast_{symbol}.csv")
    return hourly_returns, rv, garch_forecast


def make_hybrid_windows(
    lstm_channels: np.ndarray,
    hourly_index: pd.DatetimeIndex,
    rv: pd.Series,
    garch_forecast: pd.Series,
    window_hours: int = WINDOW_HOURS,
    require_garch: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build hybrid windows: for each day with an RV target, append that
    day's GARCH forecast as a constant extra channel to the 168-hour LSTM
    window preceding it.

    When require_garch is True (the default), days without a GARCH forecast
    are skipped entirely. When False, they are still included with a NaN
    GARCH channel -- this lets the walk-forward splitter see the full date
    range needed to form a 365-day training window, while the caller is
    responsible for excluding NaN-channel rows from any training batch or
    forecast origin (see run_hybrid_walk_forward's has_garch mask).
    """
    X, y, dates = [], [], []

    for date, rv_value in rv.items():
        has_garch_value = date in garch_forecast.index
        if require_garch and not has_garch_value:
            continue

        end_hour = hourly_index.searchsorted(date)
        start_hour = end_hour - window_hours
        if start_hour < 0:
            continue
        lstm_window = lstm_channels[start_hour:end_hour]
        if len(lstm_window) != window_hours:
            continue

        garch_value = garch_forecast.loc[date] if has_garch_value else np.nan
        hybrid_window = append_garch_channel(lstm_window, garch_value)
        X.append(hybrid_window)
        y.append(rv_value)
        dates.append(date)

    return np.array(X), np.array(y), pd.DatetimeIndex(dates)


def run_hybrid_walk_forward(
    hourly_returns: pd.Series, rv: pd.Series, garch_forecast: pd.Series
) -> dict:
    lstm_channels = build_channels(hourly_returns.values)
    hourly_index = hourly_returns.index

    # Build hybrid windows over rv's FULL history (not just the dates GARCH
    # has a forecast for) -- the walk-forward splitter needs a long enough
    # date range to form 365-day training windows. Days without a GARCH
    # forecast get a NaN hybrid channel and are filtered out of every
    # training window and skipped as forecast origins below.
    X_all, y_all_rv, dates_all = make_hybrid_windows(
        lstm_channels, hourly_index, rv, garch_forecast, require_garch=False
    )
    y_all = log_variance_target(y_all_rv)
    has_garch = np.array([d in garch_forecast.index for d in dates_all])

    splits = list(
        walk_forward_splits(
            dates_all,
            window_days=WINDOW_DAYS,
            refit_every_n_days=REFIT_EVERY_N_DAYS,
        )
    )

    forecasts: dict[pd.Timestamp, float] = {}
    trained_model = None
    trained_scaler = None
    total_train_seconds = 0.0
    n_refits = 0

    date_to_idx = {d: i for i, d in enumerate(dates_all)}

    for split in splits:
        if not has_garch[date_to_idx[split.forecast_date]]:
            continue

        train_mask = (
            (dates_all >= split.train_start) & (dates_all <= split.train_end) & has_garch
        )
        X_train, y_train = X_all[train_mask], y_all[train_mask]

        if split.should_refit or trained_model is None:
            if len(X_train) < 20:
                continue
            scaler = WindowScaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)

            t0 = time.time()
            result = train_lstm(X_train_scaled, y_train, scaler.mean, scaler.std, seed=SEED)
            total_train_seconds += time.time() - t0
            n_refits += 1

            trained_model = result.model
            trained_scaler = scaler

        if trained_model is None:
            continue

        idx = date_to_idx[split.forecast_date]
        window = X_all[idx : idx + 1]
        window_scaled = trained_scaler.transform(window)
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
        print(f"[{symbol}] loading data and training hybrid model through walk-forward...")
        hourly_returns, rv, garch_forecast = load_data(symbol)

        t_total = time.time()
        result = run_hybrid_walk_forward(hourly_returns, rv, garch_forecast)
        wall_time = time.time() - t_total

        results[symbol] = result
        score = result["score"]
        print(
            f"[{symbol}] QLIKE={score['qlike']:.4f} MAE={score['mae']:.6f} (n={score['n_obs']}) "
            f"| {result['n_refits']} refits, {result['total_train_seconds']:.1f}s training, "
            f"{wall_time:.1f}s wall total"
        )

    out_path = RESULTS_DIR / "hybrid.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
