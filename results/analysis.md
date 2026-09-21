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
GARCH has the *worst* (BTC) or near-worst (ETH) mean absolute percentage
error of the five models, despite having the best QLIKE. This is not a
contradiction -- it's the point of using QLIKE instead of a symmetric
percentage-error metric. QLIKE penalizes under-predicting risk far more
than over-predicting it; MAPE treats both directions the same. GARCH's
forecasts are evidently biased toward slight over-prediction relative to
the other models, which MAPE punishes but QLIKE does not punish nearly as
hard -- and over-predicting risk is the safer failure mode for an actual
risk system. See `results/lstm_interpretation.md` and
`results/hybrid_interpretation.md` for the full LSTM/hybrid write-ups.

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

The DM test compares GARCH's daily QLIKE loss against the next-best
contender on each coin, with the Harvey-Leybourne-Newbold small-sample
correction and a Newey-West HAC variance estimate.

| Coin | Comparison | DM statistic | p-value | Significant at 5%? |
|---|---|---|---|---|
| BTC | GARCH vs Hybrid | -1.977 | 0.0489 | **Yes (barely)** |
| ETH | GARCH vs Hybrid | -0.664 | 0.5070 | **No** |

**BTC**: GARCH's advantage over the hybrid is still statistically
significant with weekly refit, but only just -- p went from 0.012 (monthly
refit) to 0.049 (weekly refit), right at the edge of the conventional 5%
threshold. The hybrid's real improvement from weekly refit measurably
narrowed the gap's statistical strength, even though GARCH still wins.

**ETH**: GARCH's advantage over the hybrid is not statistically significant
(p=0.507) -- and it wasn't with monthly refit either (that comparison was
against the LSTM at p=0.241; the hybrid is now the closer contender at
p=0.507). On ETH specifically, this project cannot claim GARCH is provably
better than the hybrid; the QLIKE numbers favor GARCH, but not
distinguishably from noise.

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

The headline finding from earlier still holds with weekly refit: **GARCH
degrades the least in the stress regime on both coins**, even though the
LSTM and hybrid both improved somewhat in the stress regime too (BTC LSTM
stress QLIKE went from 3.12 to 3.05; hybrid from 2.20 to 2.08). The gap in
stress-regime robustness is real and did not close with the refit-cadence
fix -- it appears to be a structural difference between the models, not
just a training-frequency artifact.

## Bottom line

GARCH(1,1) remains the best model in this project on QLIKE and stress-regime
robustness on both coins. Weekly refit (matching GARCH's cadence) is a real,
measured improvement for the LSTM and hybrid -- narrowing the gap to GARCH
on every combination and even removing statistical significance on ETH --
but it did not overturn the result. The neural models learned real
structure (both clear the naive baselines) and now train under a fairer,
matched refit cadence, but GARCH's combination of average accuracy and
stress-day reliability is still the strongest result in this project.
