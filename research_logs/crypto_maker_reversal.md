# Crypto maker-execution reversal (perp 5-min) — PROTOTYPE / SIMULATION ONLY

**No real quotes** — spread is an assumption. No alpha claimed. Signal = cross-sectional short-horizon
reversal (= BTC lag_gap cross-sectionally). Passive limit fills only when price touches/trades through.

## Models at conservative headline (spread 5 bps, maker_fee 1, taker_fee 5, adverse 2, timeout 2, exit=taker, 30m)
| label                                |   fill_rate |   gross_bps |   net_fees_bps |   net_adv_bps |   unfilled_opp_bps |   win_rate |   sharpe_filled |   n_filled |
|:-------------------------------------|------------:|------------:|---------------:|--------------:|-------------------:|-----------:|----------------:|-----------:|
| model A (touch OPTIMISTIC)           |       0.842 |      -3.156 |         -9.156 |       -11.156 |             22.940 |      0.362 |         -69.110 |       7269 |
| model B (trade-through CONSERVATIVE) |       0.808 |      -4.075 |        -10.075 |       -12.075 |             22.321 |      0.354 |         -73.030 |       6979 |
| model C (probabilistic)              |       0.743 |      -5.804 |        -11.804 |       -13.804 |             20.685 |      0.339 |         -79.580 |       6414 |

**TAKER baseline (same legs, full spread + 2 taker fees): -14.07 bps/trade.**

## Sensitivity (model B conservative)
### spread_bps
|   spread_bps |   fill_rate |   gross_bps |   net_adv_bps |   n_filled |
|-------------:|------------:|------------:|--------------:|-----------:|
|        1.000 |       0.889 |      -2.059 |       -10.059 |   7679.000 |
|        2.000 |       0.866 |      -2.518 |       -10.518 |   7478.000 |
|        5.000 |       0.808 |      -4.075 |       -12.075 |   6979.000 |
|       10.000 |       0.684 |      -7.288 |       -15.288 |   5910.000 |
### adverse_selection_bps
|   adverse_bps |   fill_rate |   gross_bps |   net_adv_bps |   n_filled |
|--------------:|------------:|------------:|--------------:|-----------:|
|         0.000 |       0.808 |      -4.075 |       -10.075 |   6979.000 |
|         1.000 |       0.808 |      -4.075 |       -11.075 |   6979.000 |
|         2.000 |       0.808 |      -4.075 |       -12.075 |   6979.000 |
|         5.000 |       0.808 |      -4.075 |       -15.075 |   6979.000 |
### buffer_bps (trade-through)
|   buffer_bps |   fill_rate |   gross_bps |   net_adv_bps |   n_filled |
|-------------:|------------:|------------:|--------------:|-----------:|
|        0.000 |       0.842 |      -3.156 |       -11.156 |   7269.000 |
|        1.000 |       0.808 |      -4.075 |       -12.075 |   6979.000 |
|        2.000 |       0.745 |      -5.764 |       -13.764 |   6433.000 |
|        5.000 |       0.628 |      -8.767 |       -16.767 |   5426.000 |
### timeout_bars
|   timeout |   fill_rate |   gross_bps |   net_adv_bps |   n_filled |
|----------:|------------:|------------:|--------------:|-----------:|
|     1.000 |       0.742 |      -4.519 |       -12.519 |   6409.000 |
|     2.000 |       0.808 |      -4.075 |       -12.075 |   6979.000 |
|     3.000 |       0.838 |      -3.899 |       -11.899 |   7235.000 |
### exit_mode (1 taker, 2 maker-assume-fill, 3 maker+taker-fallback)
|   exit_mode |   fill_rate |   gross_bps |   net_adv_bps |   n_filled |
|------------:|------------:|------------:|--------------:|-----------:|
|       1.000 |       0.808 |      -4.075 |       -12.075 |   6979.000 |
|       2.000 |       0.808 |       0.924 |        -5.076 |   6979.000 |
|       3.000 |       0.808 |      -3.170 |        -9.545 |   6979.000 |
### horizon
|   hold_bars |   fill_rate |   gross_bps |   net_adv_bps |   n_filled |
|------------:|------------:|------------:|--------------:|-----------:|
|       3.000 |       0.820 |      -3.822 |       -11.822 |  14156.000 |
|       6.000 |       0.808 |      -4.075 |       -12.075 |   6979.000 |
|      12.000 |       0.802 |      -2.636 |       -10.636 |   3458.000 |

## Fill rate by symbol (headline) & performance by UTC hour
- Fill rate by symbol: {'ADAUSDT': 0.901, 'AVAXUSDT': 0.787, 'BNBUSDT': 0.769, 'DOGEUSDT': 0.842, 'ETHUSDT': 0.756, 'LINKUSDT': 0.803, 'LTCUSDT': 0.805, 'SOLUSDT': 0.814, 'XRPUSDT': 0.794}
- Net_adv bps by UTC hour: {0: -13.14, 1: -16.07, 2: -12.38, 3: -10.04, 4: -12.23, 5: -12.4, 6: -11.62, 7: -13.67, 8: -10.16, 9: -10.71, 10: -8.56, 11: -8.89, 12: -11.82, 13: -12.23, 14: -12.67, 15: -13.04, 16: -11.69, 17: -14.89, 18: -15.93, 19: -9.17, 20: -11.73, 21: -11.52, 22: -12.78, 23: -12.53}

## Answers (the five questions)
1. **Maker vs taker?** Maker net_adv vs taker -14.07 bps/trade — maker is better by saving the
   spread, but see whether it crosses zero.
2. **Survives conservative fills (model B, taker exit)?** Best conservative net_adv = **-5.08**
   bps/trade ⇒ still NEGATIVE.
3. **Only under optimistic assumptions?** Maker-exit mode-2 (assumes passive exit always fills) net_adv
   = -5.08 bps/trade — compare to conservative taker-exit to see how much of any edge needs the
   optimistic exit-fill assumption.
4. **What assumptions are required?** See the spread/adverse sweeps — a positive result needs WIDE spread
   captured AND LOW adverse selection. If it needs adverse≈0 or maker-exit-always-fills, it is fragile.
5. **Backfill 12 months?** Only if a CONSERVATIVE config (model B, realistic exit, adverse>0) is net
   positive with adequate fill rate. Otherwise the maker idea is not yet worth the data spend.

## Limitations
- No real bid/ask/queue — spread assumed, no queue-position model (you'd be at the back of the queue).
- Adverse selection is a flat per-fill penalty, not flow-conditioned.
- 30 days, 9 majors, one regime. Mode-2 "maker exit" assumes fills — optimistic.
- This is an execution PROTOTYPE, not a live strategy. Do not claim live-tradable alpha.
