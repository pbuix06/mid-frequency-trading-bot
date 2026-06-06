# Crypto funding-reversal — 365-day BACKFILL / OOS / regime validation

**Universe:** 10 USDT perp majors · **Span:** 365 days · **Split (pre-registered):** train 60% /
val 20% / **lock-box 20% (reported separately, not tuned)** · **Cost:** crypto taker 5.0 bps/side ·
**Same 10 configs as T0100–T0109 — no tuning, no new configs.**

## Per-config: OOS net@5 by split + PnL by BTC trend regime

| config            | family   |   hold_bars |   ic_full |   train_net5 |   val_net5 |   lock_net5 |   trend_up_bps |   trend_down_bps |   trend_side_bps |   val_n | decision                |
|:------------------|:---------|------------:|----------:|-------------:|-----------:|------------:|---------------:|-----------------:|-----------------:|--------:|:------------------------|
| 1 A fz 20% h4     | A        |          48 |     0.016 |       -9.260 |     -6.553 |      -9.065 |      -4913.017 |        -6314.760 |        -7367.438 |     437 | REJECT (dies OOS)       |
| 2 A fz 20% h8     | A        |          96 |     0.019 |       -8.452 |     -3.091 |      -8.193 |      -1696.066 |        -2574.816 |        -3589.856 |     218 | REJECT (dies OOS)       |
| 3 A fz 20% h24    | A        |         288 |     0.016 |       -0.992 |      1.855 |     -10.043 |       -669.672 |         1139.678 |        -1419.707 |      72 | REJECT (dies OOS)       |
| 4 A fz 10% h8     | A        |          96 |     0.019 |       -8.568 |     -3.317 |     -10.082 |      -1053.705 |        -2898.032 |        -4705.874 |     218 | REJECT (dies OOS)       |
| 5 A fz 10% h24    | A        |         288 |     0.016 |       -3.781 |     14.862 |      -2.173 |        847.597 |         2529.028 |        -3548.481 |      72 | REJECT (dies OOS)       |
| 6 B fund+wk4h h8  | B        |          96 |     0.007 |      -12.231 |     -6.598 |     -10.562 |      -2357.306 |         -858.292 |        -3909.528 |     140 | REJECT (dies OOS)       |
| 7 B fund+wk4h h24 | B        |         288 |     0.000 |      -14.348 |     -4.331 |     -15.911 |       -955.465 |          611.395 |        -1776.560 |      50 | REJECT (dies OOS)       |
| 8 C fund+OIup h8  | C        |          96 |     0.001 |      nan     |    nan     |      -9.896 |          0.000 |           52.621 |         -670.347 |       0 | REJECT (dies OOS) [few] |
| 9 C fund+OIup h24 | C        |         288 |    -0.022 |      nan     |    nan     |     -23.127 |          0.000 |          275.298 |         -742.934 |       0 | REJECT (dies OOS) [few] |
| 10 fund+basis h8  | basis    |          96 |     0.014 |       -9.405 |     -5.186 |      -9.029 |       -521.220 |        -1999.329 |        -1963.967 |     105 | REJECT (dies OOS)       |

> `*_net5` = net bps/trade after 10 bps round-trip. `trend_*_bps` = total net PnL (bps) accrued in up /
> down / sideways months. **Decision** per pre-registered rules (REJECT if dies in val/lock; CRASH-BETA if
> positive only in down months; CANDIDATE if OOS-positive in ≥2 regimes with sensible legs).

## Headline cfg3 (funding z 20%, 24h hold) attribution
- **Trend regime:** up +380 | down +2380 |
  sideways -180 bps. **Crash-only = False.**
- **BTC direction during trade:** {'btc_down': 2154.0, 'btc_up': 426.3} — if positive only when `btc_down`, the book is
  short/crash beta, not general alpha.
- **Legs (mean forward return):** long -9.7 bps, short
  -23.9 bps.
- **Price vs funding (full):** price +7.07 + funding +0.23
  bps/trade — carry still NEGLIGIBLE.
- **Per-symbol net (bps):** {'SOLUSDT': -1250.0, 'AVAXUSDT': -648.8, 'LTCUSDT': -328.0, 'BNBUSDT': -211.6, 'ETHUSDT': -167.6, 'BTCUSDT': 251.3, 'XRPUSDT': 738.3, 'LINKUSDT': 1226.0, 'DOGEUSDT': 1447.1, 'ADAUSDT': 1523.7}
- **Per-month net (bps):** {'2025-06': 190.1, '2025-07': 284.8, '2025-08': 1431.8, '2025-09': 132.8, '2025-10': 641.7, '2025-11': 82.5, '2025-12': -879.3, '2026-01': 272.2, '2026-02': 131.2, '2026-03': 412.5, '2026-04': -227.3, '2026-05': -354.5, '2026-06': 461.9}

## The eight questions
1. **365-day backfill succeed / 2. validation pass?** See the audit (`docs/crypto_data_audit.md`) — span,
   gaps, funding coverage, OI limit.
3. **Persist outside the 30-day crash?** Decisions column + trend attribution above.
4. **Up/down/sideways?** Trend regime row for cfg3 (and per config).
5. **Survive 5 bps/side?** `val_net5` and `lock_net5` columns.
6. **Price, carry, or both?** price +7.07 vs funding +0.23.
7. **Robust across symbols/months?** Per-symbol and per-month tables above + figures.
8. **Candidate / crash-beta / rejected?** See decisions; summary counts in the runner output.

## Figures
`crypto_funding_reversal_backfill_cfg3_equity_val.png` (equity), `crypto_funding_reversal_backfill_monthly.png`, `crypto_funding_reversal_backfill_regime.png`, `crypto_funding_reversal_backfill_symbol.png`.

**No alpha claimed. Lock-box reported separately and not used for any choice.**
