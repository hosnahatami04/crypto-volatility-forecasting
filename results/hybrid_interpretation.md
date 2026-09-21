# Hybrid model: honest result

The hybrid feeds GARCH's walk-forward variance forecast into the LSTM as an
extra input channel. The question this phase answers: does the LSTM find
residual structure GARCH misses, when it's handed GARCH's own forecast as a
head start?

## QLIKE comparison (lower is better)

| Model | BTC QLIKE | ETH QLIKE |
|---|---|---|
| Persistence | 0.909 | 0.901 |
| Rolling mean (7d) | 0.492 | 0.572 |
| GARCH(1,1) | 0.406 | 0.458 |
| LSTM (Phase 4) | 0.600 | 0.504 |
| **Hybrid** | **0.529** | **0.507** |

GARCH still wins on both coins. The hybrid does not beat plain GARCH.

## What the GARCH feature actually bought

Comparing hybrid to plain LSTM in isolation (holding architecture, refit
cadence, and training protocol fixed) shows the GARCH feature had different
effects on the two coins:

- **BTC**: hybrid QLIKE 0.529 vs plain LSTM 0.600 -- an 11.8% improvement.
  The GARCH feature gave the network real, useful information here.
- **ETH**: hybrid QLIKE 0.507 vs plain LSTM 0.504 -- essentially unchanged
  (0.6% worse). The GARCH feature made no meaningful difference on this coin.

So the hybrid partially delivered on its premise (it helped meaningfully on
BTC) but did not close the gap to plain GARCH on either coin, and had no
effect at all on ETH. This is reported as the finding, not massaged: a
single constant-per-window GARCH feature is not enough to let the LSTM
match, let alone exceed, a well-specified GARCH(1,1) fit directly on the
same data.

## Why hybrid likely couldn't close the gap to GARCH

- GARCH's forecast is available to the LSTM only as one constant number
  broadcast across a 168-hour window -- a small amount of information
  compared to the 168 raw hourly observations the network also has to
  process and can get lost in.
- The LSTM still has to learn everything else about volatility clustering
  from scratch on top of that one feature; Phase 4 already showed it
  struggles to do that reliably from raw returns given this data size.
- GARCH's own forecast is already close to what the walk-forward protocol
  can extract from this data with a correctly-specified 3-parameter model --
  there may simply not be much exploitable residual structure left for an
  LSTM to find in a dataset this size.

## Two real bugs found and fixed while building this pipeline

1. **Walk-forward split count bug**: the splitter was initially called on
   only the 365 dates that had a GARCH forecast available. Since
   `window_days=365`, the splitter's `len(dates) <= window_days` guard
   rejected every date, producing zero splits and an all-NaN result. Fixed
   by building hybrid windows over the realized-volatility series' full
   history and filtering training/forecast eligibility with a `has_garch`
   mask, rather than restricting the date range passed to the splitter.

2. **Hardcoded channel count**: `train_lstm` always constructed `VolLSTM()`
   with its default channel count (3), which crashed on the hybrid's
   4-channel input with a tensor shape mismatch. Fixed by reading the
   channel count from the actual training data's shape instead of hardcoding
   it -- the same training function now serves both the plain LSTM and the
   hybrid without needing to know in advance which one it's training.

## What this means going into Phase 6

GARCH(1,1) remains the best model on both coins across this entire project so
far. Phase 6's Diebold-Mariano test will determine whether GARCH's lead over
the naive baselines is statistically significant -- and, separately, whether
the small hybrid-vs-plain-LSTM improvement on BTC is real signal or noise.
