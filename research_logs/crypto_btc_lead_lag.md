# Crypto BTC→alt lead-lag (perp 5-min) — research log  ·  PHASE-2 SMOKE TEST

**Family:** Crypto alpha 1 · **Universe:** 9 USDT-perp alts (BTC = leader, excluded) ·
**Data:** 30 days, 2026-05-06 → 2026-06-05, 24/7 continuous ·
**No OOS split** (30d too short) · **Cost:** crypto taker, canonical 5.0 bps/side.

> ⚠️ **NOT an alpha claim.** 30 days = one regime, one market move. This proves the crypto harness
> works end-to-end (24/7 backtest, IC, costs) and checks whether the lead-lag SIGN is plausible.

## Hypothesis & arms
BTC leads alts; laggards catch up (catch-up = long lag_gap). Controls: continuation (own momentum)
and reversal (own, fade movers). IC measured vs **BTC-relative** forward return.

## Results (full-sample, 30 days)

| config                         | arm          |   hold_bars |   ic_btcrel_mean |   ic_btcrel_tstat |   bucket_monotonicity |   gross_bps_per_trade |   net_bps_per_trade_5bps |   sharpe_net_5bps |   long_leg_bps |   short_leg_bps |   n_trades | too_few_trades   |
|:-------------------------------|:-------------|------------:|-----------------:|------------------:|----------------------:|----------------------:|-------------------------:|------------------:|---------------:|----------------:|-----------:|:-----------------|
| 1 catchup g15 |btc15|>.25% h15 | catchup      |           3 |            0.017 |             1.493 |               nan     |                -0.075 |                  -10.075 |           -81.211 |         -0.935 |          -0.785 |        409 | False            |
| 2 catchup g15 |btc15|>.25% h30 | catchup      |           6 |            0.029 |             2.534 |               nan     |                -0.802 |                  -10.802 |           -43.814 |         -2.489 |          -0.885 |        212 | False            |
| 3 catchup g30 |btc30|>.5% h30  | catchup      |           6 |            0.034 |             2.172 |               nan     |                 1.253 |                   -8.747 |           -24.310 |          4.451 |           1.946 |         96 | False            |
| 4 catchup g30 |btc30|>.5% h60  | catchup      |          12 |            0.019 |             1.208 |               nan     |                 0.941 |                   -9.059 |           -12.588 |         -3.371 |          -5.253 |         45 | True             |
| 5 rank lag_gap15 h30           | catchup      |           6 |            0.047 |            11.444 |               nan     |                 0.824 |                   -9.176 |          -132.994 |         -1.375 |          -3.024 |       1438 | False            |
| 6 rank lag_gap30 h60           | catchup      |          12 |            0.050 |            11.937 |               nan     |                 1.682 |                   -8.318 |           -63.560 |         -2.484 |          -5.848 |        718 | False            |
| 7 BTC+ETH leader g15 h30       | catchup      |           6 |            0.053 |            12.249 |               nan     |                 0.926 |                   -9.074 |          -139.710 |         -1.244 |          -3.095 |       1438 | False            |
| 8 lag_gap30 volz>1 h60         | catchup      |          12 |            0.050 |             3.068 |                -0.400 |                 0.923 |                   -9.077 |           -18.273 |        -14.484 |         -16.330 |         33 | True             |
| 9 continuation +altret15 h30   | continuation |           6 |           -0.047 |           -11.444 |               nan     |                -0.809 |                  -10.809 |          -156.735 |         -3.024 |          -1.406 |       1438 | False            |
| 10 reversal -altret15 h30      | reversal     |           6 |            0.047 |            11.444 |               nan     |                 0.824 |                   -9.176 |          -132.994 |         -1.375 |          -3.024 |       1438 | False            |

## Headline cfg5 (rank lag_gap15, hold 30m)
- IC(BTC-rel) +0.0475 (t +11.4);
  gross **+0.82** bps/trade; net@5bps **-9.18**.
- Alpha decay (IC vs BTC-rel by horizon): 15m:+0.0390, 30m:+0.0475, 60m:+0.0442, 120m:+0.0377
- Long leg -1.38 bps, short leg -3.02 bps; win rate 0.11.
- **Concentration:** top single-day share of net P&L = nan%. Weekday -9.15
  vs weekend -9.26 bps/trade.

### Cost sweep (cfg5)
| cost bps/side | net Sharpe | net bps/trade |
|---:|---:|---:|
| 0 | 11.95 | 0.82 |
| 2 | -46.03 | -3.18 |
| 5 | -132.99 | -9.18 |
| 10 | -277.93 | -19.18 |

## Critical interpretation (per the rules)
- **Sign:** positive catch-up IC ⇒ laggards catch up (hypothesis holds); negative ⇒ they continue.
- Net-positive **and** IC-positive **and** enough-trades configs: **0 / 10**.
- **Do NOT claim alpha from 30 days.** Reject too-few-trade configs; reject backtest-positive-but-IC-
  negative configs; flag if one day/coin carries it (see concentration).
- Gross bps/trade here are on **30 days of crypto** — not comparable in significance to the (rejected)
  stock families' multi-year samples; a bigger gross number over 30d is NOT a stronger result.

## Post-run verdict (corrections + key insight)

**(a) Concentration — computed on GROSS (the auto-line printed `nan` because the book is net-negative).**
cfg5 gross totals +1185 bps over 1438 trades across 31 days; **22/31 days positive, top single day only
10.5%** of positive gross. So the gross edge is **spread across the month, NOT a one-day fluke** — a point
in favour of the IC being real over this window.

**(b) KEY ANALYTICAL INSIGHT — the cross-sectional "lead-lag" reduces to alt reversal.** BTC's return is the
SAME constant for every alt at a timestamp, so `lag_gap = BTC_ret − alt_ret` has the **identical
cross-sectional ranking** as `−alt_ret` (verified: cfg5 ranks ≡ cfg10 ranks exactly; their backtests match
to the digit). In a dollar-neutral cross-section the BTC term **cancels**. So the positive catch-up IC is
really **short-term cross-sectional mean-reversion among the alts** (recent under-performers bounce) — the
same gross effect the equity families showed. The genuinely BTC-dependent part is only the strong-BTC
**gate** (configs 1–4: trade only when |BTC move| > threshold), which is a *timing* filter, not a separable
cross-sectional signal.

**(c) bucket monotonicity = nan** — only 9 alts split into 5 buckets is too sparse; the IC and the
control arms (continuation IC −0.0475 / reversal IC +0.0475) establish the sign instead.

**(d) Sharpe magnitudes (−130s)** are the near-constant-cost-drag × 30-day-annualization artifact — read
**bps/trade**, not the ratio.

## Six-question smoke-test answers
1. **24/7 fix passes tests?** Yes — continuous uniform grid, no ET reset, weekend bars kept, holds span
   midnight, `tod_windows` rejected; 202 tests pass.
2. **Expected sign?** Yes — 8/8 catch-up configs positive BTC-relative IC; continuation control negative;
   reversal control positive. Laggards catch up (= cross-sectional alt reversion).
3. **IC-supported?** Yes, strongly — IC ≈ 0.05, t ≈ 11–12 (≈3–10× the equity families), gross spread over
   22/31 days. Buckets too sparse (9 names) to add.
4. **Robust across configs / one setting?** SIGN robust across all catch-up configs and both controls;
   mechanism is consistent. **Tradability: 0/10 net-positive.**
5. **Survive taker costs?** **No.** Gross 0.8–1.7 bps/trade vs crypto taker ~5 bps/side (~10 bps round-trip)
   → dead by 2 bps/side; net ≈ −9 bps/trade at 5 bps. Same taker wall as equities, worse (crypto fees higher).
6. **Data length next?** 30 days = one regime. The IC is strong enough to justify backfilling **6–12 months**
   (`ingest_crypto.py --days 365`) + an **OOS split** — but more data will NOT fix the taker-cost wall; it
   only firms up the IC. The open question is **maker** execution (earn the spread on the reversion), exactly
   as for equities.

## Next
Backfill 6–12 months + OOS split before any alpha claim. The real fork is taker (dead here) vs **maker**.
**No alpha claimed; 30-day smoke test only.**
