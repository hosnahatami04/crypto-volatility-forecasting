"""Generate the three committed report plots from stored results:

1. forecast vs. realized volatility over the whole test year, with a
   zoomed panel on the single most violent week
2. cumulative QLIKE-difference curve (best model vs. GARCH)
3. calm vs. stress QLIKE bar comparison per model

Reads only committed results/ files and cached data -- never retrains any
model. Run as: python -m src.eval.report
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.returns import log_returns
from src.eval.metrics import qlike
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


def plot_forecast_vs_realized(symbol: str, rv: pd.Series, garch_forecast: pd.Series) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[2, 1])

    rv_aligned, forecast_aligned = rv.align(garch_forecast, join="inner")

    axes[0].plot(rv_aligned.index, rv_aligned.values, label="Realized volatility", color="#333333")
    axes[0].plot(
        forecast_aligned.index,
        forecast_aligned.values,
        label="GARCH(1,1) forecast",
        color="#c0392b",
        alpha=0.85,
    )
    axes[0].set_title(f"{symbol}: forecast vs. realized volatility (full test year)")
    axes[0].set_ylabel("Volatility")
    axes[0].legend()

    worst_week_end = rv_aligned.idxmax()
    zoom_start = worst_week_end - pd.Timedelta(days=5)
    zoom_end = worst_week_end + pd.Timedelta(days=2)
    zoom_rv = rv_aligned.loc[zoom_start:zoom_end]
    zoom_forecast = forecast_aligned.loc[zoom_start:zoom_end]

    axes[1].plot(zoom_rv.index, zoom_rv.values, label="Realized", color="#333333", marker="o")
    axes[1].plot(
        zoom_forecast.index,
        zoom_forecast.values,
        label="GARCH forecast",
        color="#c0392b",
        marker="o",
    )
    axes[1].set_title("Zoom: the single most violent week")
    axes[1].set_ylabel("Volatility")
    axes[1].legend()

    fig.tight_layout()
    out_path = RESULTS_DIR / f"plot_forecast_vs_realized_{symbol}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_cumulative_qlike_diff(symbol: str, rv: pd.Series, best_model: str) -> None:
    garch_forecast = load_forecast_series("garch", symbol)
    best_forecast = load_forecast_series(best_model, symbol)

    rv_g, garch_f = rv.align(garch_forecast, join="inner")
    rv_b, best_f = rv.align(best_forecast, join="inner")

    garch_qlike = qlike(rv_g**2, garch_f**2)
    best_qlike = qlike(rv_b**2, best_f**2)

    garch_qlike, best_qlike = garch_qlike.align(best_qlike, join="inner")
    diff = garch_qlike - best_qlike  # positive means best_model beat GARCH that day
    cumulative = diff.cumsum()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(cumulative.index, cumulative.values, color="#2874a6")
    ax.axhline(0, color="#888888", linewidth=1, linestyle="--")
    ax.set_title(f"{symbol}: cumulative QLIKE difference, {best_model} vs. GARCH")
    ax.set_ylabel("Cumulative (GARCH QLIKE - model QLIKE)\nabove 0 = model beating GARCH")
    ax.set_xlabel("Date")

    fig.tight_layout()
    out_path = RESULTS_DIR / f"plot_cumulative_qlike_{symbol}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_calm_vs_stress(symbol: str, rv: pd.Series) -> None:
    calm_scores, stress_scores = [], []
    for model in MODELS:
        forecast = load_forecast_series(model, symbol)
        report = calm_stress_report(rv, forecast)
        calm_scores.append(report["calm"]["qlike"])
        stress_scores.append(report["stress"]["qlike"])

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(MODELS))
    width = 0.35

    ax.bar([i - width / 2 for i in x], calm_scores, width, label="Calm", color="#5499c7")
    ax.bar([i + width / 2 for i in x], stress_scores, width, label="Stress", color="#c0392b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(MODELS, rotation=20)
    ax.set_ylabel("QLIKE")
    ax.set_title(f"{symbol}: calm vs. stress regime QLIKE by model")
    ax.legend()

    fig.tight_layout()
    out_path = RESULTS_DIR / f"plot_calm_vs_stress_{symbol}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")


def run() -> None:
    import json

    final_report = json.loads((RESULTS_DIR / "final_report.json").read_text())

    for symbol in SYMBOLS:
        print(f"=== {symbol} ===")
        rv = load_realized_vol(symbol)
        garch_forecast = load_forecast_series("garch", symbol)

        plot_forecast_vs_realized(symbol, rv, garch_forecast)

        best_model = final_report[symbol]["best_model"]
        non_garch_best = best_model if best_model != "garch" else "hybrid"
        plot_cumulative_qlike_diff(symbol, rv, non_garch_best)

        plot_calm_vs_stress(symbol, rv)


if __name__ == "__main__":
    run()
