"""Phase 6: the full five-way comparison, Diebold-Mariano significance
testing, and calm/stress regime breakdown.

Reads each model's stored walk-forward forecast series (never recomputes
any model) and produces:
  - results/final_report.json: the five-way QLIKE/MAE table for BTC and ETH,
    a DM test of the best model vs. GARCH, and per-regime (calm/stress)
    metrics for every model.

Run as: python -m src.eval.run_final_report
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.returns import log_returns
from src.eval.dm_test import diebold_mariano_test
from src.eval.metrics import mae, mean_qlike, qlike
from src.eval.stress import calm_stress_report
from src.target.realized_vol import realized_volatility

SYMBOLS = ("BTCUSDT", "ETHUSDT")
MODELS = ("persistence", "rolling_mean", "garch", "lstm", "hybrid")

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"
RESULTS_DIR = ROOT / "results"

FORECAST_FILES = {
    "persistence": "persistence_forecast_{symbol}.csv",
    "rolling_mean": "rolling_mean_forecast_{symbol}.csv",
    "garch": "garch_forecast_{symbol}.csv",
    "lstm": "lstm_forecast_{symbol}.csv",
    "hybrid": "hybrid_forecast_{symbol}.csv",
}
FORECAST_COLUMNS = {
    "persistence": "persistence_forecast_vol",
    "rolling_mean": "rolling_mean_forecast_vol",
    "garch": "garch_forecast_vol",
    "lstm": "lstm_forecast_vol",
    "hybrid": "hybrid_forecast_vol",
}


def load_forecast_series(model: str, symbol: str) -> pd.Series:
    path = RESULTS_DIR / FORECAST_FILES[model].format(symbol=symbol)
    series = pd.read_csv(path, index_col="date", parse_dates=["date"])[FORECAST_COLUMNS[model]]
    series.index = pd.to_datetime(series.index, utc=True)
    return series


def load_realized_vol(symbol: str) -> pd.Series:
    df = pd.read_parquet(CACHE_DIR / f"{symbol}_1h.parquet")
    returns = log_returns(df["close"])
    return realized_volatility(returns)


def score(rv: pd.Series, forecast: pd.Series) -> dict:
    rv_aligned, forecast_aligned = rv.align(forecast, join="inner")
    rv_var = rv_aligned**2
    forecast_var = forecast_aligned**2
    return {
        "n_obs": int(len(rv_aligned)),
        "qlike": mean_qlike(rv_var, forecast_var),
        "mae": mae(rv_aligned, forecast_aligned),
    }


def daily_qlike_series(rv: pd.Series, forecast: pd.Series) -> pd.Series:
    rv_aligned, forecast_aligned = rv.align(forecast, join="inner")
    return qlike(rv_aligned**2, forecast_aligned**2)


def run() -> None:
    report: dict[str, dict] = {}

    for symbol in SYMBOLS:
        print(f"=== {symbol} ===")
        rv = load_realized_vol(symbol)
        forecasts = {model: load_forecast_series(model, symbol) for model in MODELS}

        five_way_table = {model: score(rv, forecasts[model]) for model in MODELS}
        for model, result in five_way_table.items():
            print(f"  {model}: QLIKE={result['qlike']:.4f} MAE={result['mae']:.6f}")

        best_model = min(five_way_table, key=lambda m: five_way_table[m]["qlike"])
        print(f"  -> best model: {best_model}")

        dm_results = {}
        if best_model != "garch":
            best_daily_qlike = daily_qlike_series(rv, forecasts[best_model])
            garch_daily_qlike = daily_qlike_series(rv, forecasts["garch"])
            dm_result = diebold_mariano_test(best_daily_qlike, garch_daily_qlike)
            dm_results[f"{best_model}_vs_garch"] = dm_result.to_dict()
            print(
                f"  DM test ({best_model} vs garch): stat={dm_result.dm_statistic:.3f} "
                f"p={dm_result.p_value:.4f} "
                f"significant={'yes' if dm_result.p_value < 0.05 else 'no'}"
            )
        else:
            # GARCH is already the best model; test it against the next-best
            # non-naive contender (hybrid) to see if ITS lead is significant.
            challenger = min(
                (m for m in MODELS if m not in ("garch", "persistence", "rolling_mean")),
                key=lambda m: five_way_table[m]["qlike"],
            )
            garch_daily_qlike = daily_qlike_series(rv, forecasts["garch"])
            challenger_daily_qlike = daily_qlike_series(rv, forecasts[challenger])
            dm_result = diebold_mariano_test(garch_daily_qlike, challenger_daily_qlike)
            dm_results[f"garch_vs_{challenger}"] = dm_result.to_dict()
            print(
                f"  DM test (garch vs {challenger}): stat={dm_result.dm_statistic:.3f} "
                f"p={dm_result.p_value:.4f} "
                f"significant={'yes' if dm_result.p_value < 0.05 else 'no'}"
            )

        stress_reports = {model: calm_stress_report(rv, forecasts[model]) for model in MODELS}
        for model, sr in stress_reports.items():
            calm_q = sr["calm"]["qlike"]
            stress_q = sr["stress"]["qlike"]
            print(
                f"  {model} regime split: calm QLIKE={calm_q:.4f}, "
                f"stress QLIKE={stress_q:.4f}"
            )

        report[symbol] = {
            "five_way_table": five_way_table,
            "best_model": best_model,
            "dm_tests": dm_results,
            "regime_split": stress_reports,
        }
        print()

    out_path = RESULTS_DIR / "final_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    run()
