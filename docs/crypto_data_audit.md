# Crypto data audit — Phase 1

**Generated:** 2026-06-06T02:13:33.880464+00:00 · **Venue:** binance (single venue) · **Quote:** USDT
**Window:** 2025-06-06T01:41 → 2026-06-06T01:41 (UTC, 24/7 — no session logic) ·
**OI window:** from 2026-05-07T01:41 (Binance retains ~30d) · **Ingest time:** 1926.0s

## What exists
- **Spot 1m OHLCV** for all 10 symbols → `data/crypto/spot_1m/`
- **USDT-perp 1m OHLCV** → `data/crypto/perp_1m/` 
- **8h funding rate** → `data/crypto/funding/`
- **5m open interest** (~30d) → `data/crypto/open_interest/`
- **Metadata** → `data/crypto/metadata/ingest_meta.json`

## Symbols covered
BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT, LINKUSDT, LTCUSDT

## Columns available (spot/perp bars)
`open, high, low, close, volume, vwap, trades, quote_volume, taker_buy_base, taker_buy_quote, funding_rate, open_interest, mark_price, index_price`
(`vwap` is derived as quote_volume/volume in the resampler; perp-only columns are NaN on spot.)
Funding: `funding_rate` (8h). Open interest: `open_interest`, `oi_value` (5m).

## Per-symbol summary

| symbol   |   spot_bars | spot_start       | spot_end         |   coverage% |   gaps |   missing_bars |   ohlc_bad |   neg_price |   zero_vol | clean   |   perp_bars |   funding_n | funding_8h   |   oi_n |   oi_spacing_min |
|:---------|------------:|:-----------------|:-----------------|------------:|-------:|---------------:|-----------:|------------:|-----------:|:--------|------------:|------------:|:-------------|-------:|-----------------:|
| BTCUSDT  |      525601 | 2025-06-06 01:41 | 2026-06-06 01:41 |         100 |      0 |              0 |          0 |           0 |          0 | True    |      525601 |        1095 | True         |   8624 |                5 |
| ETHUSDT  |      525601 | 2025-06-06 01:41 | 2026-06-06 01:41 |         100 |      0 |              0 |          0 |           0 |          0 | True    |      525601 |        1095 | True         |   8624 |                5 |
| SOLUSDT  |      525601 | 2025-06-06 01:41 | 2026-06-06 01:41 |         100 |      0 |              0 |          0 |           0 |          0 | True    |      525601 |        1095 | True         |   8624 |                5 |
| BNBUSDT  |      525601 | 2025-06-06 01:41 | 2026-06-06 01:41 |         100 |      0 |              0 |          0 |           0 |          0 | True    |      525601 |        1095 | True         |   8624 |                5 |
| XRPUSDT  |      525601 | 2025-06-06 01:41 | 2026-06-06 01:41 |         100 |      0 |              0 |          0 |           0 |          0 | True    |      525601 |        1095 | True         |   8624 |                5 |
| DOGEUSDT |      525601 | 2025-06-06 01:41 | 2026-06-06 01:41 |         100 |      0 |              0 |          0 |           0 |          0 | True    |      525601 |        1095 | True         |   8624 |                5 |
| ADAUSDT  |      525601 | 2025-06-06 01:41 | 2026-06-06 01:41 |         100 |      0 |              0 |          0 |           0 |        171 | True    |      525601 |        1095 | True         |   8624 |                5 |
| AVAXUSDT |      525601 | 2025-06-06 01:41 | 2026-06-06 01:41 |         100 |      0 |              0 |          0 |           0 |       4494 | True    |      525601 |        1095 | True         |   8624 |                5 |
| LINKUSDT |      525601 | 2025-06-06 01:41 | 2026-06-06 01:41 |         100 |      0 |              0 |          0 |           0 |       2634 | True    |      525601 |        1095 | True         |   8624 |                5 |
| LTCUSDT  |      525601 | 2025-06-06 01:41 | 2026-06-06 01:41 |         100 |      0 |              0 |          0 |           0 |          0 | True    |      525601 |        1095 | True         |   8624 |                5 |

## Data quality
- **All spot frames clean:** YES ✅ (UTC, monotonic,
  no dup timestamps, OHLC-consistent, no negative/zero prices, no NaN closes).
- **Gaps:** crypto is 24/7, so any gap is a real exchange outage / thin-trade minute (no bar printed),
  not a session close. `missing_bars` counts absent 1m bars vs a fully-continuous clock.
- **Funding 8h-aligned:** see `funding_8h` column. **OI spacing** ~5 min (see `oi_spacing_min`).
- **Coverage%** = actual bars / continuous-clock bars. <100% = thin minutes with no trade.

## Spot-perp basis
Computable now via `mft.research.crypto_panel.compute_basis(spot_close, perp_close)` (perp/spot − 1);
not pre-stored (it is a derived alignment of two stored series).

## Liquidation data
**Skipped** — Binance discontinued the public all-market liquidation stream; per-symbol forced-order
data is unreliable/incomplete. Not ingested (per the "reliable only" rule).

## Ready for alpha testing?
**Data is ingested, validated, and clean.**
Spot + perp + funding + OI are present for the 10 majors at 1m/5m. **Two items before alphas run:**
1. The window is **365 days** (Phase-1 sample). For robust research, backfill more:
   `python scripts/ingest_crypto.py --days 365` (spot/perp; OI stays ~30d by API limit).
2. The equity `xs_backtest` resets its non-overlapping rebalance grid per **ET calendar day**; for
   24/7 crypto this needs a continuous/UTC-midnight reset — a small change to make when alphas run.

**Per instruction, no crypto alpha tests have been run. Awaiting approval.**
