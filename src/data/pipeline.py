"""End-to-end Phase 1 pipeline: download -> integrity check -> log returns -> ADF.

Run as `python -m src.data.pipeline`. Produces:
  - data/cache/{SYMBOL}_1h.parquet  (raw OHLCV, committed)
  - results/integrity_report.json   (gap report, committed -- part of the honesty)
  - results/adf_report.json         (stationarity evidence for prices vs. returns)
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.download import download_symbol
from src.data.integrity import check_integrity, enforce_monotonic, reject_duplicates
from src.data.returns import adf_test, log_returns

SYMBOLS = ("BTCUSDT", "ETHUSDT")
INTERVAL = "1h"
YEARS = 2.0

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"
RESULTS_DIR = ROOT / "results"


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    integrity_reports = {}
    adf_reports = {}

    for symbol in SYMBOLS:
        print(f"[{symbol}] downloading {YEARS} years of {INTERVAL} klines...")
        df = download_symbol(symbol, INTERVAL, years=YEARS, cache_dir=CACHE_DIR)
        print(f"[{symbol}] {len(df)} rows: {df.index.min()} -> {df.index.max()}")

        report = check_integrity(df, symbol=symbol, interval=INTERVAL, expected_freq="1h")
        integrity_reports[symbol] = report.to_dict()
        print(
            f"[{symbol}] integrity: {report.n_duplicates} duplicates, "
            f"{len(report.gaps)} gaps, {report.n_missing_hours} missing hours"
        )

        df = reject_duplicates(df)
        df = enforce_monotonic(df)
        df.to_parquet(CACHE_DIR / f"{symbol}_{INTERVAL}.parquet", compression="brotli")

        returns = log_returns(df["close"])
        price_adf = adf_test(df["close"])
        returns_adf = adf_test(returns)
        adf_reports[symbol] = {
            "price": price_adf.to_dict(),
            "log_returns": returns_adf.to_dict(),
        }
        print(
            f"[{symbol}] ADF p-value: price={price_adf.p_value:.4f} "
            f"(stationary={price_adf.is_stationary()}), "
            f"returns={returns_adf.p_value:.4f} (stationary={returns_adf.is_stationary()})"
        )

    (RESULTS_DIR / "integrity_report.json").write_text(json.dumps(integrity_reports, indent=2))
    (RESULTS_DIR / "adf_report.json").write_text(json.dumps(adf_reports, indent=2))
    print(f"\nWrote {RESULTS_DIR / 'integrity_report.json'}")
    print(f"Wrote {RESULTS_DIR / 'adf_report.json'}")


if __name__ == "__main__":
    run()
