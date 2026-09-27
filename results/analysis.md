# Final analysis: what won, by how much, with what significance

## The five-way table (QLIKE, lower is better)

| Model | BTC QLIKE | BTC MAE | BTC MAPE | ETH QLIKE | ETH MAE | ETH MAPE |
|---|---|---|---|---|---|---|
| Persistence | 0.909 | 0.00833 | 52.3% | 0.901 | 0.01152 | 49.9% |
| Rolling mean (7d) | 0.492 | 0.00718 | 45.1% | 0.572 | 0.00993 | 43.8% |
| GARCH(1,1) | **0.406** | 0.00766 | 52.9% | **0.458** | 0.01243 | 69.2% |
| LSTM | 0.572 | 0.00763 | 50.8% | 0.489 | 0.01059 | 52.1% |
| Hybrid | 0.512 | 0.01078 | 64.5% | 0.488 | 0.01070 | 53.2% |

**GARCH(1,1) is still the best model on both coins by QLIKE**, the
project's primary metric. But note the MAPE column tells a different story:
GARCH's MAPE is high on both coins despite its best-in-class QLIKE (52.9% on
BTC, 69.2% on ETH -- the worst MAPE of any model on ETH, and mid-pack on
BTC, where the hybrid's 64.5% is actually worse). This is not a
contradiction -- it's the point of using QLIKE instead of a symmetric
percentage-error metric. QLIKE penalizes under-predicting risk far more
than over-predicting it; MAPE treats both directions the same. GARCH's
forecasts are evidently biased toward slight over-prediction relative to
the other models, which MAPE punishes but QLIKE does not punish nearly as
hard -- and over-predicting risk is the safer failure mode for an actual
risk system. See `results/lstm_interpretation.md` and
`results/hybrid_interpretation.md` for the full LSTM/hybrid write-ups.

### Where each model ranks against the naive floor (per coin)

Whether the neural models beat the naive baselines depends on the coin --
this is not uniform, and an earlier version of this document overstated it:

- **BTC**: the 7-day rolling mean (QLIKE 0.492) actually beats both the LSTM
  (0.572) and the hybrid (0.512). Only GARCH (0.406) beats the rolling mean
  here. So on BTC, the neural models clear *persistence* but NOT the
  rolling-mean baseline.
- **ETH**: both the LSTM (0.489) and hybrid (0.488) beat the rolling mean
  (0.572), and all three trail GARCH (0.458).

Every model beats plain persistence on both coins. But "the LSTM and hybrid
beat both naive baselines" is only true on ETH, not BTC.

## Refit cadence: a real improvement, applied

An earlier version of this project ran the LSTM and hybrid with monthly
refit (vs. GARCH's weekly refit) as a documented CPU-budget tradeoff. That
tradeoff was tested directly: refitting the LSTM and hybrid weekly instead
of monthly, matching GARCH's cadence, produced a real, reproducible
improvement on all four combinations:

| Model | Coin | Monthly refit QLIKE | Weekly refit QLIKE | Improvement |
|---|---|---|---|---|
| LSTM | BTC | 0.600 | 0.572 | 4.6% |
| LSTM | ETH | 0.504 | 0.489 | 2.9% |
| Hybrid | BTC | 0.529 | 0.512 | 3.3% |
| Hybrid | ETH | 0.507 | 0.488 | 3.8% |

This is now the permanent configuration (`REFIT_EVERY_N_DAYS=7` in both
`src/eval/run_lstm.py` and `src/eval/run_hybrid.py`). The gap to GARCH
narrowed on every combination, most notably on ETH where the LSTM (0.489)
and hybrid (0.488) are now within roughly 6-7% of GARCH's 0.458 -- close
enough that the earlier DM-test conclusions were worth re-checking (see
below). None of the four combinations closed the gap entirely; GARCH
remains the best model on both coins.

## Statistical significance (Diebold-Mariano test)

The plan requires every claimed improvement to be checked with a DM test,
not just the headline one. All four key comparisons are run per coin, with
the Harvey-Leybourne-Newbold small-sample correction and a Newey-West HAC
variance estimate. A negative DM statistic means the first model has the
lower (better) QLIKE.

| Coin | Comparison | DM statistic | p-value | Significant at 5%? |
|---|---|---|---|---|
| BTC | GARCH vs Rolling mean | -1.484 | 0.139 | **No** |
| BTC | GARCH vs LSTM | -2.387 | 0.018 | **Yes** |
| BTC | GARCH vs Hybrid | -1.977 | 0.049 | **Yes (barely)** |
| BTC | Hybrid vs LSTM | -2.234 | 0.026 | **Yes** |
| ETH | GARCH vs Rolling mean | -0.956 | 0.340 | **No** |
| ETH | GARCH vs LSTM | -0.801 | 0.423 | **No** |
| ETH | GARCH vs Hybrid | -0.664 | 0.507 | **No** |
| ETH | Hybrid vs LSTM | -0.448 | 0.654 | **No** |

Several honest findings fall out of this fuller table that a single
comparison would have hidden:

- **GARCH's edge over a plain 7-day rolling mean is NOT statistically
  significant on either coin** (BTC p=0.139, ETH p=0.340). GARCH has the
  lower QLIKE, but this test cannot distinguish that lead from sampling
  noise. This is the single most important caveat in the whole project: the
  40-year-old statistical workhorse does not provably beat a trivial moving
  average on this two-year sample.
- **On BTC**, GARCH's edge over the LSTM (p=0.018) and hybrid (p=0.049) IS
  significant, and the hybrid's edge over the plain LSTM is significant too
  (p=0.026) -- so feeding GARCH's forecast into the LSTM produced a real,
  statistically-detectable improvement on BTC.
- **On ETH**, none of the pairwise differences are significant. The models
  rank GARCH < hybrid < LSTM < rolling mean by QLIKE, but the test cannot
  tell any of them apart from noise on this coin.

## Calm vs. stress regime (top-decile realized-vol days)

### BTC

| Model | Calm QLIKE | Stress QLIKE | Degradation |
|---|---|---|---|
| Persistence | 0.825 | 1.809 | 2.2x |
| Rolling mean | 0.329 | 2.242 | 6.8x |
| **GARCH(1,1)** | **0.324** | **1.287** | **4.0x** |
| LSTM | 0.341 | 3.051 | 8.9x |
| Hybrid | 0.357 | 2.076 | 5.8x |

### ETH

| Model | Calm QLIKE | Stress QLIKE | Degradation |
|---|---|---|---|
| Persistence | 0.723 | 2.814 | 3.9x |
| Rolling mean | 0.321 | 3.267 | 10.2x |
| **GARCH(1,1)** | 0.419 | **0.873** | **2.1x** |
| LSTM | 0.335 | 2.150 | 6.4x |
| Hybrid | 0.340 | 2.042 | 6.0x |

The precise, correct statement: **GARCH has the lowest absolute stress-day
QLIKE of any model on both coins** (BTC 1.287, ETH 0.873). That is the
number that matters for a risk instrument -- the actual loss on the days
that count, not the ratio to a calm-day baseline.

Note the *degradation ratio* column does NOT make GARCH the best on that
measure: on BTC, persistence degrades only 2.2x vs. GARCH's 4.0x, because
persistence starts from a much worse calm-day QLIKE (0.825 vs. 0.324) and
so has less far to fall. A low degradation ratio built on a bad starting
point is not a virtue. So the honest claim is about absolute stress-day
loss, not "degrades the least" -- an earlier version of this document
conflated the two.

Among the serious contenders, GARCH's stress-day QLIKE is dramatically
lower than the LSTM's or hybrid's on both coins (BTC: GARCH 1.287 vs. LSTM
3.051 vs. hybrid 2.076). That gap is real and did not close with the
refit-cadence fix -- it appears to be a structural difference between the
models, not just a training-frequency artifact. Both neural models improved
slightly in the stress regime after the fix (BTC LSTM stress QLIKE 3.12 ->
3.05; hybrid 2.20 -> 2.08) but nowhere near enough to catch GARCH.

## A caveat on the hybrid's evaluation window

The hybrid is not evaluated on exactly the same footing as the other models,
and this is worth stating plainly. The hybrid consumes GARCH's walk-forward
forecast as an input feature, and that forecast series only exists for the
365-day test period (Phase 3 stored it, per the alignment-discipline rule).
As a result:

- The hybrid is scored on **345 days**, not the 364-365 the other models
  get -- the first ~20 test days have no preceding GARCH-feature history to
  form a training window from, so they are skipped.
- The hybrid's training window is effectively **expanding** early on (it
  starts from ~20 samples and grows), not the fixed rolling 12-month window
  the plan specifies and the other models use, until enough
  GARCH-feature-labelled days accumulate.

Re-scoring GARCH, LSTM, and hybrid on only the 345 shared days does not
change the model ranking, so the headline result is unaffected. But the
hybrid's numbers are not a strictly apples-to-apples comparison to GARCH's,
and that limitation is a consequence of the alignment discipline, not a bug.

## Bottom line

GARCH(1,1) has the best QLIKE and the lowest absolute stress-day QLIKE on
both coins. But two honest caveats temper that: its edge over a plain 7-day
rolling mean is not statistically significant on either coin (DM p=0.139 BTC,
p=0.340 ETH), and on ETH none of the pairwise model differences are
significant at all. Weekly refit (matching GARCH's cadence) is a real,
measured improvement for the LSTM and hybrid -- and on BTC the hybrid's edge
over the plain LSTM is statistically significant, so the GARCH feature
genuinely helped there. The neural models beat plain persistence on both
coins and beat the rolling mean on ETH (though not on BTC), so they learned
real structure. GARCH's combination of best QLIKE and best stress-day loss
is still the strongest result in this project -- but "strongest here" is not
the same as "provably better than a moving average," and this document says
so.
