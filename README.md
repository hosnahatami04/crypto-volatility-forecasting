# Crypto Volatility Forecasting

Forecasts next-24h volatility for Bitcoin and Ethereum -- **not price direction**.
Price is essentially unpredictable; volatility (how violently price moves) is
partially predictable, thanks to volatility clustering. This repo pits a
classical statistical model (GARCH) against deep learning (LSTM) and a
hybrid of the two, under strict walk-forward validation, and checks whether
any improvement over the classical baseline is statistically significant.

> Personal project. Public market data. Not investment advice.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Table of contents

- [Why volatility, not price](#why-volatility-not-price)
- [Pipeline](#pipeline)
- [Results](#results)
- [Installation -- users](#installation--users)
- [Installation -- developers](#installation--developers)
- [Contributor expectations](#contributor-expectations)
- [Known issues](#known-issues)

## Why volatility, not price

Nobody can reliably predict whether Bitcoin's price will go up or down
tomorrow -- if they could, they wouldn't be selling that prediction as a blog
post. But *how violently* the price moves is a different question, and it
has real structure: turbulent days cluster together, and calm days cluster
together. That's volatility clustering, and it's the entire premise this
repo is built on.

This matters because volatility is what risk management is actually built
on -- position sizing, stop-loss placement, and options pricing all depend
on an estimate of expected volatility, not a guess about direction.

The choice of *log returns* over raw prices is not a textbook default here --
it's measured on this project's own data. An Augmented Dickey-Fuller test on
BTC's price series gives p=0.396 (cannot reject non-stationarity), while the
same test on log returns gives p<0.0001 (clearly stationary). Every model in
this repo works on log returns because of that measured result, not because
a textbook said so. Full ADF results: [`results/adf_report.json`](results/adf_report.json).

## Pipeline

```mermaid
flowchart LR
    A[Binance API] -->|download.py| B[Hourly OHLCV cache]
    B -->|returns.py| C[Log returns]
    C -->|realized_vol.py| D[Daily realized volatility]
    C --> E[Naive baselines]
    C --> F[GARCH 1,1]
    C --> G[LSTM]
    F -->|forecast feature| G2[Hybrid: GARCH + LSTM]
    G --> G2
    E --> H[Walk-forward evaluation]
    F --> H
    G --> H
    G2 --> H
    H -->|QLIKE, MAE| I[Diebold-Mariano test]
    H --> J[Calm / stress regime split]
    I --> K[Final report]
    J --> K
```

Every model is scored through the same rolling-origin walk-forward protocol:
fit on a 12-month trailing window, forecast the next day, slide forward one
day, repeat across the full test year. No model ever sees a day's data
before every prior day has been "lived through" -- verified by a test that
plants a poisoned future value and asserts no training window can see it
([`tests/test_walk_forward.py`](tests/test_walk_forward.py)).

## Results

Five-way comparison. QLIKE is the project's primary metric (lower is
better) -- it punishes under-prediction of risk far more severely than
over-prediction, which is what a risk system actually needs. MAPE (also
lower is better) is a plain percentage-error figure included for
intuition, but it's *not* the primary metric: MAPE treats over- and
under-prediction symmetrically, so it can and does disagree with QLIKE (see
note below the table).

| Model | BTC QLIKE | BTC MAE | BTC MAPE | ETH QLIKE | ETH MAE | ETH MAPE |
|---|---|---|---|---|---|---|
| Persistence | 0.909 | 0.00833 | 52.3% | 0.901 | 0.01152 | 49.9% |
| Rolling mean (7d) | 0.492 | 0.00718 | 45.1% | 0.572 | 0.00993 | 43.8% |
| **GARCH(1,1)** | **0.406** | 0.00766 | 52.9% | **0.458** | 0.01243 | 69.2% |
| LSTM | 0.572 | 0.00763 | 50.8% | 0.489 | 0.01059 | 52.1% |
| Hybrid (GARCH + LSTM) | 0.512 | 0.01078 | 64.5% | 0.488 | 0.01070 | 53.2% |

**GARCH(1,1) is the best model on both coins by QLIKE.** Its edge over the
hybrid on BTC is statistically significant but only just (Diebold-Mariano
p=0.049); its edge over the hybrid on ETH is not (p=0.507) -- reported
plainly either way, per this project's rule against massaging results.

Notice GARCH has the worst or near-worst MAPE despite the best QLIKE --
that's not a contradiction, it's exactly what QLIKE is designed to reward
that MAPE isn't: GARCH's errors evidently skew toward over-predicting risk,
which QLIKE treats as a much smaller sin than under-predicting it. A model
optimized for MAPE alone would not necessarily be the model you want for
risk management.

The result that matters most for a risk instrument: **GARCH degrades far
less than the LSTM or hybrid on the top-decile most volatile days.** On BTC,
GARCH's stress-day QLIKE (1.29) is well under half the LSTM's (3.05). The
LSTM and hybrid looked reasonably competitive on calm-day averages but both
broke down specifically on the days a risk model exists to get right --
this gap did not close even after fixing the LSTM/hybrid refit cadence (see
below), so it appears to be a structural difference between the models.

**Refit cadence was tested and fixed.** The LSTM and hybrid originally
refit monthly against GARCH's weekly refit, as a CPU-budget tradeoff.
Switching them to weekly refit (matching GARCH) was tested directly and
produced a real, reproducible improvement on every combination -- 2.9% to
4.6% lower QLIKE -- and is now the permanent configuration. It narrowed the
gap to GARCH but did not close it. Full before/after numbers in
[`results/analysis.md`](results/analysis.md).

Full analysis, every number sourced from committed `results/` files:
[`results/analysis.md`](results/analysis.md) · GARCH parameter interpretation:
[`results/garch_interpretation.md`](results/garch_interpretation.md) · LSTM
and hybrid honest write-ups:
[`results/lstm_interpretation.md`](results/lstm_interpretation.md),
[`results/hybrid_interpretation.md`](results/hybrid_interpretation.md).

Plots (regenerated by `python -m src.eval.report`):

- `results/plot_forecast_vs_realized_{SYMBOL}.png` -- forecast vs. realized
  volatility over the test year, with a zoomed panel on the most violent week
- `results/plot_cumulative_qlike_{SYMBOL}.png` -- cumulative QLIKE
  difference, best contender vs. GARCH
- `results/plot_calm_vs_stress_{SYMBOL}.png` -- calm vs. stress QLIKE by model

### Demo video

*(Link to be added -- recorded by the repo owner. See the CLI demo below for
the same command shown in the video.)*

## Installation -- users

Reproduces every number in this README from the committed data cache,
offline, no API access required:

```bash
docker build -t crypto-vol-forecast .
docker run crypto-vol-forecast
```

This re-runs the full evaluation pipeline (`src/eval/run_final_report.py`)
against the committed `data/cache/` parquet files and prints the same
five-way table shown above.

To run the forecast CLI directly instead:

```bash
docker run crypto-vol-forecast python -m src.cli forecast --pair BTC --horizon 24h
```

## Installation -- developers

```bash
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run the test suite and lint
pytest -v
ruff check src tests

# Re-download the raw data (optional -- the cache is already committed)
python -m src.data.pipeline

# Re-run any stage of the pipeline
python -m src.eval.run_baselines
python -m src.eval.run_garch
python -m src.eval.run_lstm      # several minutes on CPU
python -m src.eval.run_hybrid    # several minutes on CPU
python -m src.eval.run_final_report
python -m src.eval.report        # regenerate plots
```

The forecast CLI:

```bash
python -m src.cli forecast --pair BTC --horizon 24h
```

## Contributor expectations

- Branch per change, PR into `main` -- no direct pushes to `main`.
- `pytest -v` and `ruff check src tests` must both pass before a PR is opened.
- CI runs the full test suite, lint, and a QLIKE regression gate
  (`src/eval/ci_regression_gate.py`) that re-fits GARCH on the committed
  cache and fails the build if QLIKE regresses by more than 5% -- a fast
  check that doesn't require retraining the LSTM/hybrid on every push.

## Known issues

- **Refit cadence: fixed, not just documented.** The LSTM and hybrid
  originally refit monthly against GARCH's weekly refit, as a CPU-budget
  tradeoff. This was tested directly and found to cost real accuracy (2.9%
  to 4.6% higher QLIKE across all four LSTM/hybrid x BTC/ETH combinations),
  so both models now refit weekly, matching GARCH. Daily refit for the
  neural models remains impractical on CPU and was not attempted at scale.
  The stress-regime gap to GARCH did *not* close with this fix, suggesting
  it's a structural difference between the models rather than purely a
  training-frequency artifact -- see `results/analysis.md`.
- **~2 years of data is a small-data regime for a neural network.** The
  LSTM and hybrid models learned real structure (both beat the naive
  baselines) but did not have enough independent training signal to match a
  well-specified 3-parameter GARCH model on this dataset size, even after
  the refit-cadence fix above.
- **Data gaps**: the download pipeline detects and reports gaps explicitly
  (`results/integrity_report.json`) rather than silently interpolating over
  them; the committed cache for this run had zero gaps and zero duplicate
  timestamps on both symbols.
- **Stress-period weakness is real and reported, not hidden**: every model
  in this repo, including GARCH, gets meaningfully worse on the most
  volatile 10% of days. GARCH degrades the least, but "degrades the least"
  is not the same as "handles stress days well in absolute terms" -- see
  `results/analysis.md` for the actual numbers.
- **Not investment advice.** This forecasts volatility, not price direction,
  and is built on public market data for a portfolio project, not a
  production risk system.
