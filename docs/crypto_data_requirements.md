# Crypto data requirements (acquire before any crypto alpha runs)

**Status:** NO crypto data exists in this repo (verified — `data/` has none; the `BTC/ETH/SOL`
files in `data/pit/` are daily *equity* tickers, not crypto). The research engine
(`mft.research.*`) is asset-class-agnostic and will consume crypto bars through the SAME path as
equities once a provider is wired. Interface: `mft/data_layer/crypto_provider.py`. **We will not
fabricate crypto results.** This doc states exactly what to acquire.

## Why crypto is a strong next data bet (vs. buying more equity intraday)
- **Free / cheap & deep:** Binance/Bybit REST give years of 1-min OHLCV, funding, and OI at no cost.
- **24/7, no survivorship-by-listing on majors:** BTC/ETH/SOL have continuous history; delisting
  bias is small if you stick to majors (and manageable if you include the dead alts deliberately).
- **Real microstructure data is obtainable:** funding rate, open interest, and (paid/streamed)
  order book — the inputs the equity side is *missing*. Several specified crypto alphas (funding
  reversal, OI+price, basis) have **no equity analogue you can get cheaply.**
- **Less retail-saturated taker microstructure** than US mega-cap equities.

## Minimum dataset to acquire (phase 1)

| Field | Granularity | Needed for | Source |
|---|---|---|---|
| OHLCV (`open/high/low/close/volume`) | 1-min (resample→5-min) | momentum, reversal, BTC lead-lag | Binance/Bybit klines (free) |
| `quote_volume`, `trades` | 1-min | volume features | same klines payload |
| `funding_rate` | 8-hourly (ffill to bar) | funding-rate reversal | Binance/Bybit funding history (free) |
| `open_interest` | 5-min or hourly | OI + price | Binance/Bybit OI history (free) |
| spot `close` **and** perp `close`, aligned | 1-min | spot-perp basis | two endpoints, same exchange/quote |

**Universe (phase 1):** BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, plus 5–10 more liquid
USDT pairs (≥ a chosen $ ADV floor). Majors first; this is enough for lead-lag and momentum.

## Quality bar (non-negotiable, same discipline as equities)
1. **Timezone:** every timestamp **UTC**, tz-aware. Crypto is 24/7 — **no session masking** (the
   research engine must skip `regular_session()` for crypto; only equities use RTH).
2. **One exchange + one quote per series** (e.g. Binance, USDT). Don't blend venues silently —
   prices and funding differ across exchanges. Stamp `exchange`/`quote` in `df.attrs`.
3. **Spot vs perp kept separate**, aligned only to compute `basis`. Never mix them in one series.
4. **De-duplicated, gap-flagged, monotonic** index; drop or flag exchange-outage gaps.
5. **Corporate-action analogue:** watch for **token redenominations / ticker swaps / chain
   migrations** (e.g. LUNA→LUNC, rebrands). Document any adjustment.
6. **Funding sign convention** explicit: positive = longs pay shorts (crowded longs). OI units
   (contracts vs USD notional) explicit per vendor.
7. **Survivorship:** if you go beyond majors, **include delisted/dead alts** or state the bias
   loudly — pumped-then-dead alts are exactly where naive momentum/lead-lag overstates.

## Lock-box (propose)
Mirror the equity discipline: seal the **most recent 12 months** as the crypto lock-box; research
on everything before it. Set the constant in `crypto_provider.py` when data lands.

## Acquisition steps (when you choose to proceed)
1. Implement `BinanceProvider(spot)` + `BinancePerpProvider(perp + funding + OI)` subclassing
   `CryptoProvider`; map vendor columns → `normalize_crypto_bars(...)`.
2. Ingest phase-1 universe → `data/crypto/{spot,perp}/<SYMBOL>.parquet`.
3. Add a `crypto_panel` loader (or generalize `mft.research.panel.load_panel` with
   `session=False`) — the only equity-specific bit is RTH masking.
4. Set `CRYPTO_LOCKBOX`. Run **BTC lead-lag** + **short-term momentum** first (OHLCV-only, no
   extra data needed), IC-first, through the existing `signal_lab` / `xs_backtest`.
5. Then funding-reversal / OI / basis once those fields are validated clean.

## What NOT to do
- Don't buy a fancy paid crypto feed before free Binance/Bybit OHLCV has shown an IC pulse.
- Don't run a crypto alpha on majors-only and claim a broad effect (same survivorship trap as the
  33 equity names).
- Don't blend exchanges or spot/perp to "fill gaps" — that manufactures fake basis/funding signal.
