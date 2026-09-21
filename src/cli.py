"""Forecast CLI: prints tomorrow's volatility estimate from cached data.

Usage:
    python -m src.cli forecast --pair BTC --horizon 24h

Uses only the committed data cache and the final full-history GARCH fit --
no network access required. This is the honest, offline demo surface for
the project.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.returns import log_returns
from src.models.garch import daily_log_returns_pct, fit_garch_11
from src.target.realized_vol import realized_volatility

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"

PAIR_TO_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}


def load_cached_close(symbol: str) -> pd.Series:
    cache_path = CACHE_DIR / f"{symbol}_1h.parquet"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached data for {symbol} at {cache_path}. "
            "Run `python -m src.data.pipeline` to download it first."
        )
    df = pd.read_parquet(cache_path)
    return df["close"]


def forecast_next_day_vol(symbol: str) -> tuple[float, pd.Series]:
    """Fit GARCH(1,1) on the full cached history and forecast tomorrow's
    volatility. Returns (forecast_vol, historical_realized_vol) so the
    caller can compute percentile context.
    """
    close = load_cached_close(symbol)
    hourly_returns = log_returns(close)
    rv = realized_volatility(hourly_returns)
    returns_pct = daily_log_returns_pct(hourly_returns)

    fit = fit_garch_11(returns_pct)
    forecast_variance_raw = fit.next_day_variance / (100**2)
    forecast_vol = forecast_variance_raw**0.5

    return forecast_vol, rv


def regime_percentile(forecast_vol: float, historical_rv: pd.Series) -> float:
    """What percentile of the past year's realized volatility does the
    forecast fall at? Gives calm-vs-stressed context to the raw number.
    """
    last_year = historical_rv.tail(365)
    return float((last_year < forecast_vol).mean() * 100)


def run_forecast(pair: str, horizon: str) -> None:
    if horizon != "24h":
        raise ValueError(f"Only --horizon 24h is currently supported, got: {horizon}")

    symbol = PAIR_TO_SYMBOL.get(pair.upper())
    if symbol is None:
        raise ValueError(f"Unknown pair '{pair}'. Supported: {list(PAIR_TO_SYMBOL)}")

    forecast_vol, historical_rv = forecast_next_day_vol(symbol)
    percentile = regime_percentile(forecast_vol, historical_rv)

    if percentile >= 90:
        regime = "STRESSED"
    elif percentile >= 60:
        regime = "elevated"
    else:
        regime = "calm"

    print(f"{pair.upper()}/USDT -- next-24h volatility forecast (GARCH(1,1))")
    print(f"  Forecast: {forecast_vol:.4f}")
    print(
        f"  Regime: {regime} (today's forecast sits at the {percentile:.0f}th "
        "percentile of the past year)"
    )
    print("  Note: this is not investment advice. Forecasts volatility, not price direction.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    forecast_parser = subparsers.add_parser("forecast", help="Forecast next-period volatility")
    forecast_parser.add_argument("--pair", required=True, help="BTC or ETH")
    forecast_parser.add_argument(
        "--horizon", default="24h", help="Forecast horizon (only 24h supported)"
    )

    args = parser.parse_args()

    if args.command == "forecast":
        run_forecast(args.pair, args.horizon)


if __name__ == "__main__":
    main()
