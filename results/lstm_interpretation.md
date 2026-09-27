# LSTM: honest result

The plan flagged this as a real possibility before any code was written: a
plain LSTM often fails to beat GARCH on volatility forecasting. That is
exactly what happened here.

## QLIKE comparison (lower is better)

| Model | BTC QLIKE | ETH QLIKE |
|---|---|---|
| Persistence | 0.909 | 0.901 |
| Rolling mean (7d) | 0.492 | 0.572 |
| GARCH(1,1) | 0.406 | 0.458 |
| **LSTM** | **0.572** | **0.489** |

GARCH wins on both coins. The LSTM is 40.9% worse than GARCH on BTC and
6.8% worse on ETH (QLIKE, higher = worse) -- both narrower gaps than the
monthly-refit numbers reported earlier in this project (47.7% and 10.1%
respectively), after switching the LSTM to weekly refit (see below).

The LSTM beats plain persistence on both coins, and beats the 7-day rolling
mean on ETH (0.489 vs 0.572) -- but NOT on BTC, where the rolling mean
(0.492) is actually better than the LSTM (0.572). So it learned something
real from the return windows (it is not broken or random), but on BTC a
trivial moving average still beats it. It does not outperform a
well-specified 3-parameter statistical model on a dataset this size (roughly
2 years, one observation per day for training purposes).

## Why GARCH likely has the edge here

- **Data size**: walk-forward training folds give GARCH a full year of daily
  returns to fit 3 parameters. The LSTM has to learn general patterns from
  roughly the same number of overlapping 168-hour windows, which is a much
  higher-parameter model (thousands of weights) fit to comparatively little
  independent information.
- **Inductive bias**: GARCH's functional form (today's variance depends on
  yesterday's shock and yesterday's variance) is close to the true generating
  mechanism of financial volatility clustering. The LSTM has to discover
  something equivalent from raw return sequences with no such prior.
- **Refit cadence**: originally GARCH refit weekly while the LSTM refit
  monthly (a CPU-budget tradeoff). This was tested directly and fixed: the
  LSTM now refits weekly too (`REFIT_EVERY_N_DAYS=7`), matching GARCH's
  cadence, which measurably improved QLIKE by 4.6% (BTC) and 2.9% (ETH) --
  see `results/analysis.md` for the full before/after comparison. The
  remaining gap to GARCH is not explained by refit cadence alone.
- **The LSTM may be undertrained, not just data-starved.** The training
  loop (`src/models/lstm_train.py`) is full-batch: one Adam step per epoch,
  capped at 100 epochs with early stopping (patience 8). That is a small
  number of gradient updates, and combined with an MSE loss on log-variance
  (which, being symmetric in log space, mildly biases point forecasts
  downward relative to the QLIKE-optimal forecast), it is a plausible second
  contributor to the LSTM's tendency to under-predict and to its poor
  stress-day performance -- not just "too little data." This is an honest
  limitation of the current setup rather than a demonstrated fact; a
  mini-batch loop with more update steps, or a QLIKE-aware loss, would be the
  natural next experiment. It was left out of scope here to keep the
  committed results reproducible under a single documented protocol.

## A bug found and fixed along the way

An earlier version of the LSTM predicted raw volatility directly through an
unconstrained linear output head. That produced a small number of negative
volatility forecasts (6 out of 364 test days for ETH), which is nonsensical
for a variance-based quantity and catastrophic under QLIKE specifically: as
a forecast variance approaches zero, QLIKE's `realized/forecast` ratio
diverges toward infinity. That bug alone inflated ETH's QLIKE to 1042.7 while
MAE looked unremarkable (0.0134) -- a reminder that QLIKE and MAE can disagree
sharply, and QLIKE's asymmetry is not just a theoretical property but something
that can wreck a metric on a handful of bad forecasts.

Fixed by training the network on log-variance instead of raw volatility and
exponentiating the output back -- this guarantees a strictly positive
forecast by construction, with no clipping or other post-hoc patch. Full
walk-forward numbers above are from the fixed model.

## What this means for the hybrid model (Phase 5)

Since GARCH's own forecast already captures the dominant signal here, the
hybrid model's job in Phase 5 is to test whether the LSTM can add residual
structure GARCH misses -- not to replace GARCH. If the hybrid doesn't beat
plain GARCH either, that is also a valid, reportable finding.
