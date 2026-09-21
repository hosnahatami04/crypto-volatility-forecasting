"""Run GARCH(1,1) through walk-forward validation (weekly refit) on real
BTC/ETH data, score with QLIKE/MAE, and record parameter interpretation.

Run as: python -m src.eval.run_garch
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.returns import log_returns
from src.eval.metrics import mae, mean_qlike
from src.eval.walk_forward import walk_forward_splits
from src.models.garch import daily_log_returns_pct, fit_garch_11
from src.target.realized_vol import realized_volatility

SYMBOLS = ("BTCUSDT", "ETHUSDT")
WINDOW_DAYS = 365
REFIT_EVERY_N_DAYS = 7  # weekly refit, per the documented CPU-budget tradeoff

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"
RESULTS_DIR = ROOT / "results"


def load_returns_and_rv(symbol: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_parquet(CACHE_DIR / f"{symbol}_1h.parquet")
    hourly_returns = log_returns(df["close"])
    rv = realized_volatility(hourly_returns)
    returns_pct = daily_log_returns_pct(hourly_returns)
    return returns_pct, rv


def run_garch_walk_forward(returns_pct: pd.Series, rv: pd.Series) -> dict:
    splits = list(
        walk_forward_splits(
            returns_pct.index,
            window_days=WINDOW_DAYS,
            refit_every_n_days=REFIT_EVERY_N_DAYS,
        )
    )

    forecast_variance = {}
    last_params: dict | None = None

    for split in splits:
        if split.should_refit or last_params is None:
            train_window = returns_pct.loc[split.train_start : split.train_end]
            result = fit_garch_11(train_window)
            last_params = result.to_dict()
            forecast_variance[split.forecast_date] = result.next_day_variance
        else:
            # Between refits, re-forecast one step ahead using the last-fit
            # parameters applied to the latest observed return (still uses
            # only data through train_end -- no lookahead).
            prev_return_pct = returns_pct.loc[split.train_end]
            omega, alpha, beta = last_params["omega"], last_params["alpha"], last_params["beta"]
            prev_variance = last_params["next_day_variance"]
            next_variance = omega + alpha * prev_return_pct**2 + beta * prev_variance
            last_params = {**last_params, "next_day_variance": next_variance}
            forecast_variance[split.forecast_date] = next_variance

    forecast_series_pct2 = pd.Series(forecast_variance).sort_index()
    # Convert forecast variance from (percent log return)^2 to (log return)^2 units
    # to match realized volatility's units: percent^2 -> raw^2 is divide by 100^2.
    forecast_variance_raw = forecast_series_pct2 / (100**2)
    forecast_vol_raw = np.sqrt(forecast_variance_raw)

    rv_aligned, forecast_vol_aligned = rv.align(forecast_vol_raw, join="inner")
    rv_var = rv_aligned**2
    forecast_var = forecast_vol_aligned**2

    score = {
        "n_obs": int(len(rv_aligned)),
        "qlike": mean_qlike(rv_var, forecast_var),
        "mae": mae(rv_aligned, forecast_vol_aligned),
    }

    # Fit one final time on the full available history for parameter interpretation.
    final_fit = fit_garch_11(returns_pct)

    return {
        "score": score,
        "parameter_interpretation": final_fit.to_dict(),
    }


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for symbol in SYMBOLS:
        print(f"[{symbol}] loading data and fitting GARCH(1,1) through walk-forward...")
        returns_pct, rv = load_returns_and_rv(symbol)
        result = run_garch_walk_forward(returns_pct, rv)
        results[symbol] = result

        score = result["score"]
        params = result["parameter_interpretation"]
        print(
            f"[{symbol}] QLIKE={score['qlike']:.4f} MAE={score['mae']:.6f} (n={score['n_obs']})"
        )
        print(
            f"[{symbol}] omega={params['omega']:.4f} alpha={params['alpha']:.4f} "
            f"beta={params['beta']:.4f} persistence(a+b)={params['persistence']:.4f}"
        )

    out_path = RESULTS_DIR / "garch.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
