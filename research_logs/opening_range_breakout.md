# Opening-range breakout (cross-sectional, 5-min) — research log

**Family:** Stock alpha 3/8 · **Universe:** 31 survivor mega-caps (survivorship-biased) ·
**Window:** 2020-07-27 → 2023-06-30 (train+val) · **Lock-box:** SEALED · **top_n:** 3.

## Hypothesis
Names breaking beyond their opening range with strength/volume CONTINUE over 30–120 min. Long
strongest up-breakouts, short strongest down-breakouts, dollar-neutral. Cross-sectional reading
(distinct from the single-name event ORB already tested taker-dead in T0052–T0054).

## Results (train + validation; lock-box untouched)

| config                 |   hold_bars |   train_ic_mean |   train_ic_tstat |   bucket_monotonicity |   val_gross_bps_per_trade |   val_net_bps_per_trade |   val_sharpe_net |   val_long_leg_bps |   val_short_leg_bps |   val_trades_per_day |   val_n_trades | too_few_trades   |
|:-----------------------|------------:|----------------:|-----------------:|----------------------:|--------------------------:|------------------------:|-----------------:|-------------------:|--------------------:|---------------------:|---------------:|:-----------------|
| 1 OR15 b0 h30          |           6 |          -0.011 |           -7.015 |                 0.000 |                     0.658 |                  -3.342 |          -10.925 |              2.569 |               1.253 |               10.968 |           1360 | False            |
| 2 OR15 b5 h30          |           6 |          -0.012 |           -7.060 |                -1.000 |                     0.622 |                  -3.378 |          -11.137 |              2.554 |               1.309 |               10.911 |           1353 | False            |
| 3 OR15 b10 vz1 h60     |          12 |          -0.012 |           -2.519 |                -0.200 |                    -0.474 |                  -4.474 |           -3.303 |             12.126 |              13.074 |                1.643 |             69 | False            |
| 4 OR30 b0 h60          |          12 |          -0.016 |           -8.447 |                -1.000 |                     1.266 |                  -2.734 |           -5.096 |              4.693 |               2.160 |                4.968 |            616 | False            |
| 5 OR30 b5 vz1 h60      |          12 |          -0.015 |           -2.917 |                 0.200 |                     0.334 |                  -3.666 |           -2.199 |              8.890 |               8.222 |                1.564 |             61 | False            |
| 6 OR30 b10 vz1.5 h120  |          24 |          -0.008 |           -0.931 |                 0.600 |                     1.588 |                  -2.412 |           -0.471 |              1.463 |              -1.713 |                1.222 |             11 | True             |
| 7 OR60 b5 h120         |          24 |          -0.011 |           -3.989 |                -0.400 |                     0.461 |                  -3.539 |           -3.003 |              7.459 |               6.537 |                1.811 |            221 | False            |
| 8 OR15 SPYadj vz1 h60  |          12 |          -0.007 |           -1.697 |                 1.000 |                     3.541 |                  -0.459 |           -0.300 |             14.042 |               6.959 |                1.844 |            118 | False            |
| 9 OR30 SPYadj vz1 h120 |          24 |          -0.006 |           -1.245 |                -0.400 |                     2.275 |                  -1.725 |           -0.767 |             12.043 |               7.492 |                1.242 |             41 | True             |

> The short leg is the *weakest* breakouts among genuine breakouts; on up-trending bars those can
> still be up-breakouts (not literal down-breakouts) — read the leg columns with that in mind.

## Cost sweep (headline 4 OR30 b0 h60, validation)

| cost bps/side | net Sharpe | net bps/trade |
|---:|---:|---:|
| 0 | 2.36 | 1.27 |
| 1 | -1.37 | -0.73 |
| 2 | -5.10 | -2.73 |
| 5 | -16.28 | -8.73 |
| 10 | -34.92 | -18.73 |

Net bps/trade by ET hour (headline): 10:00=-2.2  11:00=-2.4  12:00=-2.9  13:00=-5.4  14:00=-0.8

## Answers to the 6 questions
1. **Gross/trade vs reversal:** ORB headline **+1.27** bps vs reversal **+0.66** bps.
2. **Survives 1–2 bps/side?** See cost sweep — positive only if net stays >0 at 1–2 bps.
3. **Consistent across OR windows?** Compare the no-vol OR15/30/60 rows — a real edge is a plateau,
   not one setting.
4. **Volume confirmation help/hurt?** Compare config 4 (no vol) vs 5 (vol_z>1) at OR30.
5. **IC/bucket-supported?** Headline mean IC -0.0160, t -8.4 — a
   positive monotonic IC is required, else the backtest is an artifact.
6. **Leg attribution:** long-leg +4.69 bps, short-leg
   +2.16 bps.

**Net-positive & enough-trades configs: 0 / 9.**

## Reading it honestly
- Read **net bps/trade**, not net Sharpe magnitude. IC t-stats overstate significance (overlap).
- Reject configs with **too few trades** (n_trades < 60), no IC support, no OR-window
  plateau, or one lucky month.
- 31 survivors, one 2020–23 regime, IEX partial tape. Not a deployable alpha. Lock-box untouched.
