# Crypto cross-sectional MOMENTUM — 365-day OOS / regime validation

**Universe:** 10 USDT perp majors · **Span:** 365d (2025-06-06→2026-06-06) · **Split:** train 60% /
val 20% / **lock-box 20% (reported, not tuned)** · **Cost:** taker 5.0 bps/side · **Funding = context only.**
Long winners / short losers, dollar-neutral, continuous 24/7, entry next bar. Pre-registered; no tuning.

## Per-config: IC, OOS net@5 by split, PnL by BTC trend regime, price vs funding

| config                | family   |   hold_bars |   ic_full |   train_net5 |   val_net5 |   lock_net5 |   trend_up_bps |   trend_down_bps |   trend_side_bps |   full_price |   full_fund |   val_n | decision                |
|:----------------------|:---------|------------:|----------:|-------------:|-----------:|------------:|---------------:|-----------------:|-----------------:|-------------:|------------:|--------:|:------------------------|
| 1 raw mom lb4 h4      | raw      |          48 |    -0.036 |      -10.237 |    -12.546 |     -12.000 |      -7025.131 |        -6820.241 |        -9976.612 |       -0.858 |      -0.035 |     437 | REJECT (dies OOS)       |
| 2 raw mom lb8 h8      | raw      |          96 |    -0.040 |      -10.388 |    -11.782 |     -11.943 |      -5374.297 |        -2450.103 |        -4342.461 |       -1.001 |      -0.130 |     218 | REJECT (dies OOS)       |
| 3 raw mom lb24 h24    | raw      |         288 |    -0.044 |       -3.621 |    -25.762 |     -11.979 |       -741.603 |        -1125.198 |        -1377.422 |        1.292 |      -0.230 |      72 | REJECT (dies OOS)       |
| 4 btcrel lb4 h4       | btcrel   |          48 |    -0.036 |      -10.237 |    -12.546 |     -12.000 |      -7025.131 |        -6820.241 |        -9976.612 |       -0.858 |      -0.035 |     437 | REJECT (dies OOS)       |
| 5 btcrel lb8 h8       | btcrel   |          96 |    -0.040 |      -10.388 |    -11.782 |     -11.943 |      -5374.297 |        -2450.103 |        -4342.461 |       -1.001 |      -0.130 |     218 | REJECT (dies OOS)       |
| 6 btcrel lb24 h24     | btcrel   |         288 |    -0.044 |       -3.621 |    -25.762 |     -11.979 |       -741.603 |        -1125.198 |        -1377.422 |        1.292 |      -0.230 |      72 | REJECT (dies OOS)       |
| 7 voladj r8/v24 h8    | voladj   |          96 |    -0.038 |      -10.181 |    -11.162 |     -10.206 |      -4647.110 |        -2629.467 |        -4383.626 |       -0.577 |      -0.111 |     218 | REJECT (dies OOS)       |
| 8 voladj r24/v24 h24  | voladj   |         288 |    -0.024 |       -7.408 |    -25.930 |      -6.261 |      -1108.948 |         -730.223 |        -1883.367 |       -0.086 |      -0.169 |      72 | REJECT (dies OOS)       |
| 9 mom8+volz4>0 h8     | volconf  |          96 |    -0.043 |      -10.429 |     -9.081 |     -23.248 |      -1231.303 |         -216.644 |         -692.312 |       -0.552 |      -0.096 |      45 | REJECT (dies OOS)       |
| 10 mom24+relvol>0 h24 | volconf  |         288 |    -0.051 |        8.897 |     -8.405 |       3.554 |        -36.003 |          439.247 |         -658.777 |        6.261 |      -0.253 |      11 | REJECT (dies OOS) [few] |

> `*_net5` = net bps/trade after 10 bps round-trip **including funding drag**. `full_price`/`full_fund` split
> the full-sample edge into momentum price PnL vs funding carry (long winners usually PAY funding).

## Headline cfg3 (raw momentum, 24h lookback/hold)
- **Trend regime:** up -742 | down -1125 |
  sideways -1377 bps.
- **BTC direction during trade:** {'btc_down': 712.8, 'btc_up': -327.0} — both-legs-positive + only-when-btc-up ⇒ long beta.
- **Legs (mean fwd):** long -12.8 bps, short -15.4 bps.
- **Price vs funding:** price +1.29 + funding -0.23 bps/trade
  (funding is a drag for momentum).
- **Per-symbol net (bps):** {'LINKUSDT': -875.6, 'XRPUSDT': -700.4, 'BTCUSDT': -596.1, 'LTCUSDT': -529.6, 'AVAXUSDT': -116.5, 'SOLUSDT': 130.8, 'DOGEUSDT': 390.9, 'ETHUSDT': 543.2, 'BNBUSDT': 817.5, 'ADAUSDT': 1321.7}
- **Per-month net (bps):** {'2025-06': -257.6, '2025-07': 382.4, '2025-08': 504.1, '2025-09': 673.1, '2025-10': -681.9, '2025-11': 363.2, '2025-12': 449.6, '2026-01': -50.4, '2026-02': -769.7, '2026-03': -56.0, '2026-04': -389.5, '2026-05': 150.9, '2026-06': 67.6}

### Cost sweep (cfg3 total net)
| cost bps/side | net Sharpe | net bps/trade |
|---:|---:|---:|
| 0 | 0.24 | 1.06 |
| 2 | -0.66 | -2.94 |
| 5 | -1.99 | -8.94 |
| 10 | -4.23 | -18.94 |

## The six questions
1. **Survive OOS / lock-box?** `val_net5` and `lock_net5` columns.
2. **Survive 5 bps/side?** same columns (net is after 10 bps round-trip).
3. **IC-supported?** `ic_full`; raw-mom mean IC -0.0403, btc-relative -0.0403
   (positive = momentum; **negative = reversal even here**).
4. **Alpha or beta?** trend-regime row + btc-direction + legs above.
5. **Robust across symbols/months/regimes?** per-symbol / per-month tables + figures.
6. **Candidate / reject / pause?** decisions column; counts in runner output.

## Figures
`crypto_cross_sectional_momentum_cfg3_equity_val.png`, `_monthly.png`, `_regime.png`, `_symbol.png`, `_ic_month.png`, `_cost.png`.

**No alpha claimed. Lock-box reported separately, not used to choose anything.**
