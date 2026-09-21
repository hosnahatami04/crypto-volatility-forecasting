# GARCH(1,1) parameter interpretation

Fit on the full available history of daily log returns (percent scale), one
fit per symbol, using the `arch` package's default GARCH(1,1) with a zero
mean model.

## BTC/USDT

| Parameter | Value | Meaning |
|---|---|---|
| omega | 0.773 | Baseline daily variance the process reverts to absent shocks |
| alpha | 0.129 | ~13% of yesterday's squared return feeds into today's variance forecast |
| beta | 0.728 | ~73% of yesterday's variance persists into today |
| alpha + beta | 0.857 | Strong volatility clustering; shocks decay slowly (half-life ~4.5 days) |

## ETH/USDT

| Parameter | Value | Meaning |
|---|---|---|
| omega | 4.589 | Baseline daily variance -- notably higher than BTC's, consistent with ETH's typically larger swings |
| alpha | 0.091 | ~9% of yesterday's squared return feeds into today's variance forecast |
| beta | 0.553 | ~55% of yesterday's variance persists into today |
| alpha + beta | 0.644 | Moderate volatility clustering, weaker and shorter-lived than BTC's |

## Reading these numbers

`alpha + beta` measures how long a volatility shock's influence lingers. A
value near 1 means today's turbulence strongly predicts tomorrow's; a value
near 0 means volatility reverts to baseline almost immediately.

BTC's persistence (0.857) is meaningfully higher than ETH's (0.644) on this
2-year window -- BTC's volatility clusters more strongly and takes longer to
decay back to baseline than ETH's does. This is measured directly from the
walk-forward fits, not assumed from prior literature.

## Walk-forward score (weekly refit, QLIKE/MAE)

| Symbol | QLIKE | MAE | n |
|---|---|---|---|
| BTC | 0.406 | 0.00766 | 365 |
| ETH | 0.458 | 0.01243 | 365 |

Both clear the naive baselines from Phase 2 (BTC rolling-mean QLIKE 0.492,
ETH rolling-mean QLIKE 0.572) -- GARCH beats the floor on both coins, as
expected.
