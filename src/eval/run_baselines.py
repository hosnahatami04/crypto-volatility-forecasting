"""Run naive baselines (persistence, 7-day rolling mean) through the full
walk-forward protocol on real cached BTC/ETH data, scored with QLIKE and MAE.

This is the floor everything else in the project must clear. Run as:
    python -m src.eval.run_baselines
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.returns import log_returns
from src.eval.metrics import mae, mean_qlike
from src.eval.walk_forward import walk_forward_splits
from src.models.naive import persistence_forecast, rolling_mean_forecast
from src.target.realized_vol import realized_volatility

SYMBOLS = ("BTCUSDT", "ETHUSDT")
WINDOW_DAYS = 365

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"
RESULTS_DIR = ROOT / "results"


def load_realized_vol(symbol: str) -> pd.Series:
    df = pd.read_parquet(CACHE_DIR / f"{symbol}_1h.parquet")
    returns = log_returns(df["close"])
    return realized_volatility(returns)


def score_baseline(rv: pd.Series, forecast: pd.Series, window_days: int) -> dict:
    """Score a forecast series only over the walk-forward test period
    (i.e. only origins that had a full window_days of history available).
    """
    splits = list(walk_forward_splits(rv.index, window_days=window_days))
    test_dates = pd.DatetimeIndex([s.forecast_date for s in splits])

    rv_test = rv.loc[rv.index.isin(test_dates)]
    forecast_test = forecast.loc[forecast.index.isin(test_dates)]

    rv_test, forecast_test = rv_test.align(forecast_test, join="inner")
    valid = forecast_test.notna() & rv_test.notna()
    rv_test = rv_test[valid]
    forecast_test = forecast_test[valid]

    rv_var = rv_test**2
    forecast_var = forecast_test**2

    return {
        "n_obs": int(len(rv_test)),
        "qlike": mean_qlike(rv_var, forecast_var),
        "mae": mae(rv_test, forecast_test),
    }


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for symbol in SYMBOLS:
        rv = load_realized_vol(symbol)
        print(f"[{symbol}] {len(rv)} days of realized volatility")

        persistence = persistence_forecast(rv)
        rolling_mean = rolling_mean_forecast(rv, window=7)

        persistence_score = score_baseline(rv, persistence, WINDOW_DAYS)
        rolling_mean_score = score_baseline(rv, rolling_mean, WINDOW_DAYS)

        results[symbol] = {
            "persistence": persistence_score,
            "rolling_mean_7d": rolling_mean_score,
        }

        print(
            f"[{symbol}] persistence: QLIKE={persistence_score['qlike']:.4f} "
            f"MAE={persistence_score['mae']:.6f} (n={persistence_score['n_obs']})"
        )
        print(
            f"[{symbol}] rolling_mean_7d: QLIKE={rolling_mean_score['qlike']:.4f} "
            f"MAE={rolling_mean_score['mae']:.6f} (n={rolling_mean_score['n_obs']})"
        )

    out_path = RESULTS_DIR / "baselines.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
