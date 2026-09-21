"""Fast CI regression gate: re-fits GARCH(1,1) walk-forward on the committed
data cache and fails if QLIKE regresses beyond tolerance versus the committed
results/garch.json. Deliberately does NOT retrain the LSTM/hybrid models --
those take minutes, which is not viable on every CI run. GARCH is fast
(seconds) and is the project's best model, so a regression there is the
highest-value thing to catch automatically.

Run as: python -m src.eval.ci_regression_gate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.eval.run_garch import load_returns_and_rv, run_garch_walk_forward

SYMBOLS = ("BTCUSDT", "ETHUSDT")
QLIKE_TOLERANCE = 0.05  # allow up to 5% relative regression before failing

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"


def run() -> int:
    committed = json.loads((RESULTS_DIR / "garch.json").read_text())
    failures = []

    for symbol in SYMBOLS:
        committed_qlike = committed[symbol]["score"]["qlike"]

        returns_pct, rv = load_returns_and_rv(symbol)
        result = run_garch_walk_forward(returns_pct, rv)
        fresh_qlike = result["score"]["qlike"]

        relative_change = (fresh_qlike - committed_qlike) / committed_qlike
        status = "OK"
        if relative_change > QLIKE_TOLERANCE:
            status = "REGRESSION"
            failures.append(symbol)

        print(
            f"[{symbol}] committed QLIKE={committed_qlike:.4f} fresh QLIKE={fresh_qlike:.4f} "
            f"({relative_change:+.1%}) -- {status}"
        )

    if failures:
        print(f"\nFAILED: QLIKE regressed beyond {QLIKE_TOLERANCE:.0%} for: {failures}")
        return 1

    print(f"\nPASSED: all symbols within {QLIKE_TOLERANCE:.0%} tolerance of committed QLIKE")
    return 0


if __name__ == "__main__":
    sys.exit(run())
