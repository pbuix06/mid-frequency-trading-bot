# Short-term cross-sectional reversal (5-min) — research log

**Family:** Stock alpha 1/8 · **Frequency:** 5-min · **Universe:** 31 survivor
mega-caps (survivorship-biased) · **Window:** 2020-07-27 → 2023-06-30 (train+val) · **Lock-box:** SEALED.

## Hypothesis
Names that idiosyncratically over-extend down (after removing SPY beta) partially revert over
15–60 min — a liquidity-provision premium. Signal = −recent return (fade); long losers, short
winners, dollar-neutral top/bottom-5.

## Method
IC-first: rank-IC vs forward 15/30/60/120-min returns (no fills/costs) → only then a
non-overlapping dollar-neutral LS backtest, gross & net (cost 2.0 bps/side),
entry lag 1 bar, flat overnight. Pre-registered grid; every config logged to `trials/trials.csv`.

## Results (train + validation only; lock-box untouched)

| variant   |   lookback_bars |   hold_bars |   train_ic_mean |   train_ic_tstat |   bucket_monotonicity |   train_sharpe_net |   val_sharpe |   val_net_per_trade_bps |   val_max_dd |
|:----------|----------------:|------------:|----------------:|-----------------:|----------------------:|-------------------:|-------------:|------------------------:|-------------:|
| raw       |               1 |           6 |           0.006 |            4.452 |                 1.000 |            -12.712 |      -14.520 |                  -3.735 |       -0.428 |
| raw       |               3 |           6 |           0.008 |            6.058 |                 0.000 |            -11.783 |      -11.529 |                  -3.083 |       -0.377 |
| raw       |               6 |           6 |           0.010 |            7.724 |                 0.200 |            -11.830 |      -12.034 |                  -3.350 |       -0.397 |
| residual  |               1 |           6 |           0.005 |            4.243 |                 0.400 |            -13.972 |      -16.475 |                  -4.031 |       -0.451 |
| residual  |               3 |           3 |           0.007 |            5.364 |                 0.000 |            -24.924 |      -28.045 |                  -3.495 |       -0.663 |
| residual  |               3 |           6 |           0.007 |            5.364 |                 0.000 |            -13.103 |      -13.609 |                  -3.342 |       -0.394 |
| residual  |               3 |          12 |           0.007 |            5.364 |                 0.000 |             -7.092 |       -8.556 |                  -4.255 |       -0.275 |
| residual  |               6 |           6 |           0.008 |            6.518 |                 1.000 |            -13.098 |      -14.558 |                  -3.701 |       -0.425 |

- Configs with |train IC t| > 2: **8 / 8**.
- Best by **validation** net Sharpe: **residual**, lookback 3 bars,
  hold 12 bars → **val Sharpe -8.56** (train net
  -7.09; train IC t 5.4;
  net/trade -4.25 bps).

## Reading it honestly
- **Gross vs net:** the table reports both; rank only on **net**. If gross is positive but net is
  ~0, the spread eats the edge — the recurring intraday taker verdict in this project.
- **Read net bps/trade, not the net Sharpe magnitude.** Net P&L per trade ≈ a near-constant cost
  drag (low variance), so the net *Sharpe ratio* blows up in magnitude (−8 to −28). That is an
  artifact of dividing a small, almost-deterministic negative mean by a tiny std — it does **not**
  mean "28× worse than a good book". The economically honest number is **net return per trade in
  bps** (here ≈ −3 to −4 bps): a small positive gross edge minus a larger round-trip cost.
- **IC t-stats are optimistic.** IC is computed on every bar's forward window, which overlaps
  heavily (consecutive 30-min windows share 29/30 of their span), so the effective number of
  independent periods is far smaller than the raw count → the t-stat overstates significance.
  Treat IC>2 here as *directional mechanism* evidence (the sign and monotonicity), not a precise p.
- **IC vs Sharpe:** a positive, monotonic IC is *mechanism* evidence; a positive validation net
  Sharpe is *tradeability* evidence. Need both. Here: mechanism yes (tiny), tradeability no.
- **Survivorship & regime:** 31 survivors over one 2020–23 regime cluster. Do **not** read any
  number as a deployable alpha. This proves the engine and measures the effect.
- **Lock-box:** not touched. The validation number is the selection number; the lock-box exam is
  a future one-time event via `splits.load_lockbox(i_am_done_tuning=True)`.

## Next
- If a variant shows stable IC>2 **and** net Sharpe survives 2× cost → carry to the next family
  and re-test combined (correlation screen). If net dies at cost (likely) → record as taker-dead,
  consistent with T0046/T0050, and note it as a candidate **maker** signal (needs quote data).
