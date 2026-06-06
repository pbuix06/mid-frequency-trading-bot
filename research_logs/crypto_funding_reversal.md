# Crypto funding-rate reversal (perp 5-min) — research log  ·  PHASE-1 SMOKE TEST

**Universe:** 10 USDT perp majors · **Data:** ~30 days, funding 8h (~90 settlements) · **24/7 continuous** ·
**Cost:** crypto taker 5.0 bps/side · **No funding look-ahead** (known-only mapping, entry next bar).

> ⚠️ **NOT an alpha claim.** ~30 days / ~90 funding settlements is very thin. Longer-hold configs have few
> trades. This is a smoke test of a NEW (positioning) signal + price-vs-carry attribution.

## Results (full-sample, 30 days)

| config            | family_tag   |   hold_bars |   top_n |   ic_mean |   ic_tstat |   price_bps |   funding_bps |   total_gross_bps |   net_bps_5 |   n_trades | too_few_trades   |
|:------------------|:-------------|------------:|--------:|----------:|-----------:|------------:|--------------:|------------------:|------------:|-----------:|:-----------------|
| 1 A fz 20% h4     | A            |          48 |       2 |     0.006 |      1.454 |       1.804 |         0.046 |             1.851 |      -8.149 |        159 | False            |
| 2 A fz 20% h8     | A            |          96 |       2 |     0.014 |      3.310 |       3.571 |         0.093 |             3.664 |      -6.336 |         79 | False            |
| 3 A fz 20% h24    | A            |         288 |       2 |     0.031 |      7.770 |      23.836 |         0.380 |            24.216 |      14.216 |         25 | False            |
| 4 A fz 10% h8     | A            |          96 |       1 |     0.014 |      3.310 |      -4.070 |         0.135 |            -3.936 |     -13.936 |         79 | False            |
| 5 A fz 10% h24    | A            |         288 |       1 |     0.031 |      7.770 |      41.940 |         0.499 |            42.440 |      32.440 |         25 | False            |
| 6 B fund+wk4h h8  | B            |          96 |       2 |     0.058 |      8.833 |      10.160 |         0.002 |            10.162 |       0.162 |         53 | False            |
| 7 B fund+wk4h h24 | B            |         288 |       2 |     0.092 |     14.266 |      35.910 |         0.163 |            36.073 |      26.073 |         17 | True             |
| 8 C fund+OIup h8  | C            |          96 |       2 |     0.020 |      3.153 |       8.694 |         0.025 |             8.718 |      -1.282 |         50 | False            |
| 9 C fund+OIup h24 | C            |         288 |       2 |     0.006 |      0.970 |      12.008 |         0.244 |            12.252 |       2.252 |         14 | True             |
| 10 fund+basis h8  | basis        |          96 |       2 |     0.022 |      2.797 |       0.537 |         0.030 |             0.567 |      -9.433 |         44 | False            |

> `price_bps` = perp price-reversion PnL/trade; `funding_bps` = funding carry PnL/trade (short receives
> +funding, long pays); `total_gross` = price+funding; `net_bps_5` = total − 10 bps round-trip (5/side).

## Cost sweep (headline cfg2: Family A, 20%, 8h hold)
| cost bps/side | net Sharpe | net bps/trade |
|---:|---:|---:|
| 0 | 2.82 | 3.66 |
| 2 | -0.26 | -0.34 |
| 5 | -4.87 | -6.34 |
| 10 | -12.56 | -16.34 |

## Answers (the six questions)
1. **Gross vs 5m reversal (~1.2 bps/trade):** headline total gross **+3.66**
   bps/trade over **8h** holds — LARGER,
   as expected for longer holds (bigger moves), but on far fewer trades.
2. **Survives taker cost?** Net@5bps = **-6.34** bps/trade — see cost sweep.
3. **IC-supported?** Headline IC +0.0136
   (t +3.3); Family-A IC-positive 5/5.
4. **Edge from price, funding, or both?** Headline: PRICE +3.57 + FUNDING
   +0.09 bps/trade. (If most of `total` is `funding`, it is a CARRY trade, not
   a price-prediction edge — important distinction.)
5. **Robust or one coin/day?** Net-positive & enough-trades configs: **3/10**; top single
   day = 32% of headline net P&L.
6. **30 days too short?** Yes — funding history is ~30d/~90 settlements; 24h configs have ~25–30 trades.
   Backfill before any belief (see below).

## Critical checks
- No future funding used (known-only mapping + entry next bar; tested in `tests/test_funding.py`).
- Reject too-few-trade configs (flagged). Reject backtest-positive-but-IC-negative.
- **Price vs carry:** read the split — a funding *carry* edge (mechanical) is different from a price
  *reversion* edge (predictive). Both shown per config.

## Limitations / next
- ~30 days, 10 majors — thin; longer holds especially. No spot-perp execution modelled; funding assumed
  received/paid in full (no funding-prediction error). **Backfill 6–12 months of funding + perp before
  any further work** if a config is net-positive with adequate trades AND IC-supported.

## POST-RUN VERDICT (the decisive caveats)

**This is the FIRST signal in the project to survive realistic taker costs net-positive** — but with
three load-bearing caveats:

1. **REGIME-CONFOUNDED.** The 30-day window was a **−25.4% BTC crash**. The funding-reversal book = short
   the crowded longs / majors, which got liquidated hardest in the fall. Both legs have negative forward
   returns (long −81 bps, short −129 bps at 24h); the book is positive only because the crowded side fell
   *more*. We **cannot distinguish funding-positioning-reversion alpha from "short crowded majors in a
   crash" beta** with one down-month. An up-month (short squeeze) and a sideways month are untested.
2. **The edge is PRICE, not carry.** Funding carry is **negligible (~0.1–0.5 bps/trade, <2% of total)**.
   Funding is the *signal* (positioning predicts reversion), not an income stream. The "funding carry"
   half of the hypothesis is debunked here.
3. **Not one coin, but tiny sample.** Per-symbol net contribution is spread (LINK +347, BTC +228, SOL
   +115, BNB +105 positive; AVAX −286, XRP −100 negative) — *not* a single-coin artifact. BUT the
   net-positive 24h configs have only **~25 trades** (config 3) / ~17 (config 7) — far too few to believe.
   The 10% (single-name) configs are 2-name books — ignore.

**What IS established:** funding has **positive, horizon-increasing IC** (0.006→0.031 Family A; 0.092 t≈14
Family B at 24h), Family-A IC-positive 5/5, and the **longer-hold / lower-turnover thesis works**: a 24h
hold's ~24 bps gross clears the ~10–20 bps round-trip cost where the 5-min reversal's ~1 bps never could.

**Recommendation:** unlike the reversal branch, this thread **is worth backfilling 6–12 months** — with a
specific kill/confirm test: does the edge persist in **up and sideways** regimes, or is it only "crowded
longs get liquidated in crashes"? Until then: **no alpha claimed, regime-confounded, 25-trade sample.**
