"""Download hourly OHLCV klines and cache to parquet.

Data source decision (Phase 1 connectivity probe, 2026-09-21): Binance's public
REST API (api.binance.com) is reachable from the development network with no
geo-blocking, returns ~0.7s latency, and serves full 2-year hourly history.
CoinGecko's free tier only returns hourly granularity for the trailing ~90 days
(older history collapses to daily candles), so it cannot satisfy this project's
2-year hourly requirement and is not wired up as a live fallback. Any future
fallback source must implement the same (symbol, interval, start_ms, end_ms) ->
list[Kline] shape as `fetch_klines` below.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_KLINES_PER_REQUEST = 1000
RATE_LIMIT_SLEEP_SECONDS = 0.3

COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


@dataclass(frozen=True)
class KlineRequest:
    symbol: str
    interval: str
    start_ms: int
    end_ms: int


def fetch_klines(
    request: KlineRequest,
    session: requests.Session | None = None,
    sleep_seconds: float = RATE_LIMIT_SLEEP_SECONDS,
) -> list[list]:
    """Fetch all klines for the requested window, paginating past the 1000-row cap."""
    session = session or requests.Session()
    all_rows: list[list] = []
    cursor = request.start_ms

    while cursor < request.end_ms:
        params = {
            "symbol": request.symbol,
            "interval": request.interval,
            "startTime": cursor,
            "endTime": request.end_ms,
            "limit": MAX_KLINES_PER_REQUEST,
        }
        response = session.get(BINANCE_KLINES_URL, params=params, timeout=15)
        response.raise_for_status()
        rows = response.json()
        if not rows:
            break

        all_rows.extend(rows)
        last_open_time = rows[-1][0]
        cursor = last_open_time + 1

        if len(rows) < MAX_KLINES_PER_REQUEST:
            break

        time.sleep(sleep_seconds)

    return all_rows


def klines_to_dataframe(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.drop(columns=["ignore"])
    return df.set_index("open_time").sort_index()


def download_symbol(
    symbol: str,
    interval: str,
    years: float,
    cache_dir: Path,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download `years` of history for `symbol`/`interval` and cache to parquet."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(years * 365 * 24 * 60 * 60 * 1000)

    request = KlineRequest(symbol=symbol, interval=interval, start_ms=start_ms, end_ms=end_ms)
    rows = fetch_klines(request, session=session)
    df = klines_to_dataframe(rows)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol}_{interval}.parquet"
    df.to_parquet(cache_path, compression="brotli")
    return df


def load_cached(symbol: str, interval: str, cache_dir: Path) -> pd.DataFrame:
    cache_path = cache_dir / f"{symbol}_{interval}.parquet"
    return pd.read_parquet(cache_path)


if __name__ == "__main__":
    cache_dir = Path(__file__).resolve().parents[2] / "data" / "cache"
    for symbol in ("BTCUSDT", "ETHUSDT"):
        print(f"Downloading {symbol}...")
        df = download_symbol(symbol, "1h", years=2.0, cache_dir=cache_dir)
        print(f"  {len(df)} rows, {df.index.min()} -> {df.index.max()}")
