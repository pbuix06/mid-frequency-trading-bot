# Abnormal-volume continuation (5-min) — research log

**Family:** Stock alpha 2/8 · **Universe:** 31 survivor mega-caps (survivorship-biased) ·
**Window:** 2020-07-27 → 2023-06-30 (train+val) · **Lock-box:** SEALED.

## Hypothesis
Abnormal same-time-of-day volume + strong residual-vs-SPY move ⇒ continuation over 30–120 min.
Long up-move+high-vol, short down-move+high-vol, dollar-neutral top/bottom-N.

## Main question — gross edge vs short-term reversal
- Abnormal-volume continuation headline (**6 score h60 N5**): **+0.06 bps/trade gross**.
- Short-term reversal (family 1 headline): **+0.66 bps/trade gross**.
- ⇒ Abnormal-volume gross edge is **NOT larger than** reversal.

## Results (train + validation only; lock-box untouched)

| config          |   top_n |   hold_bars |   train_ic_mean |   train_ic_tstat |   bucket_monotonicity |   val_gross_bps_per_trade |   val_net_bps_per_trade |   val_sharpe_net |   val_trades_per_day |
|:----------------|--------:|------------:|----------------:|-----------------:|----------------------:|--------------------------:|------------------------:|-----------------:|---------------------:|
| 1 z>1 resid15   |       5 |           6 |          -0.014 |           -3.839 |                -0.400 |                    -1.172 |                  -5.172 |           -4.338 |                1.780 |
| 2 z>1 resid30   |       5 |          12 |          -0.010 |           -2.712 |                 0.400 |                     1.469 |                  -2.531 |           -1.233 |                1.370 |
| 3 z>1.5 resid30 |       5 |          12 |          -0.016 |           -2.911 |                -0.800 |                     7.674 |                   3.674 |            0.951 |                1.250 |
| 4 z>2 resid30   |       5 |          12 |          -0.016 |           -2.114 |                -0.200 |                     8.828 |                   4.828 |            0.482 |                1.000 |
| 5 z>1.5 resid60 |       5 |          24 |          -0.014 |           -2.265 |                -0.400 |                     5.385 |                   1.385 |            0.154 |                1.125 |
| 6 score h60 N5  |       5 |          12 |          -0.006 |           -4.822 |                 0.400 |                     0.059 |                  -3.941 |           -8.387 |                6.000 |
| 7 score h120 N5 |       5 |          24 |          -0.006 |           -3.974 |                -0.400 |                    -1.966 |                  -5.966 |           -5.950 |                3.000 |
| 8 score tod N5  |       5 |          12 |          -0.006 |           -4.822 |                 0.400 |                    -0.286 |                  -4.286 |           -3.840 |                2.000 |
| 6 score h60 N3  |       3 |          12 |          -0.006 |           -4.822 |                 0.400 |                     0.225 |                  -3.775 |           -6.300 |                6.000 |
| 7 score h120 N3 |       3 |          24 |          -0.006 |           -3.974 |                -0.400 |                    -2.111 |                  -6.111 |           -4.646 |                3.000 |
| 8 score tod N3  |       3 |          12 |          -0.006 |           -4.822 |                 0.400 |                     0.566 |                  -3.434 |           -2.456 |                2.000 |

- Configs with |train IC t| > 2: **11 / 11** (t inflated by overlapping windows — directional only).
- Best by **validation** net Sharpe: **3 z>1.5 resid30** (net 3.67 bps/trade).

## Cost sweep (headline config, validation)

| cost bps/side | net Sharpe | net bps/trade |
|---:|---:|---:|
| 0 | 0.13 | 0.06 |
| 1 | -4.13 | -1.94 |
| 2 | -8.39 | -3.94 |
| 5 | -21.16 | -9.94 |
| 10 | -42.44 | -19.94 |

## Critical verdict
The decisive numbers are **gross bps/trade vs cost bps/trade**. Round-trip cost = 2× the per-side
bps (the $1-gross book is opened and closed). If gross bps/trade < round-trip cost at a realistic
1–5 bps/side, the alpha is **taker-dead** regardless of how clean the IC looks. Diagnose which of:
(1) alpha too small, (2) too much turnover, (3) too few trades, (4) bad universe,
(5) weak IEX data, (6) needs maker/limit execution, (7) needs broader universe — see the summary
the runner prints. Reject if weak; a positive IC with sub-cost gross is *mechanism, not money*.

## Reading it honestly
- Read **net bps/trade**, not net Sharpe magnitude (near-constant cost drag inflates the ratio).
- IC t-stats overstate significance (overlapping forward windows).
- 31 survivors, one 2020–23 regime: not a deployable alpha. Engine + measurement only.
- Lock-box not touched; validation is the selection number.
