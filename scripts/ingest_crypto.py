"""
Crypto Phase 1 — REAL ingestion from Binance (public, no API key) + validation + audit.

Fetches, per symbol: spot 1m OHLCV, USDT-perp 1m OHLCV, 8h funding, 5m open interest (Binance
retains ~30d of OI). Saves to data/crypto/{spot_1m,perp_1m,funding,open_interest}/<SYM>.parquet,
writes metadata, runs mft.data_layer.crypto_validate on everything, and emits
docs/crypto_data_audit.md. No synthetic data is ever written.

Window is configurable; Phase 1 default = 30 days (aligns with OI retention; ~43k 1m bars/sym;
completes in a few minutes). Backfill more later with the SAME code:  --days 365.

Usage:
    python scripts/ingest_crypto.py                 # 30 days, all 10 symbols
    python scripts/ingest_crypto.py --days 90       # longer spot/perp window (OI still ~30d)
    python scripts/ingest_crypto.py --no-perp       # spot only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from mft.data_layer.crypto_provider import BinancePerpProvider, BinanceSpotProvider  # noqa: E402
from mft.data_layer.crypto_validate import (  # noqa: E402
    ohlcv_is_clean,
    validate_funding,
    validate_ohlcv,
    validate_oi,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
           "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]

CRYPTO = ROOT / "data" / "crypto"
DIRS = {k: CRYPTO / k for k in ("spot_1m", "perp_1m", "funding", "open_interest", "metadata")}
AUDIT = ROOT / "docs" / "crypto_data_audit.md"
OI_MAX_DAYS = 30  # Binance openInterestHist retention


def _save(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--no-perp", action="store_true")
    args = ap.parse_args()

    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    end = pd.Timestamp.now(tz="UTC").floor("min")
    start = end - pd.Timedelta(days=args.days)
    oi_start = max(start, end - pd.Timedelta(days=OI_MAX_DAYS))
    s_iso, e_iso = start.isoformat(), end.isoformat()
    print(f"Crypto ingest: {len(args.symbols)} symbols, {s_iso} -> {e_iso} "
          f"(venue=binance, perp={'no' if args.no_perp else 'yes'})\n")

    spot = BinanceSpotProvider()
    perp = None if args.no_perp else BinancePerpProvider()
    reports, meta, t0 = {}, {}, time.time()

    for sym in args.symbols:
        rep = {}
        sdf = spot.get_bars(sym, s_iso, e_iso, "1m")
        _save(sdf, DIRS["spot_1m"] / f"{sym}.parquet")
        rep["spot"] = validate_ohlcv(sdf)
        line = f"  {sym:<9} spot {len(sdf):>7} bars"

        if perp is not None:
            pdf = perp.get_bars(sym, s_iso, e_iso, "1m")
            _save(pdf, DIRS["perp_1m"] / f"{sym}.parquet")
            rep["perp"] = validate_ohlcv(pdf)

            fund = perp.get_funding(sym, s_iso, e_iso)
            _save(fund.to_frame(), DIRS["funding"] / f"{sym}.parquet")
            rep["funding"] = validate_funding(fund)

            oi = perp.get_open_interest(sym, oi_start.isoformat(), e_iso, "5m")
            if not oi.empty:
                _save(oi, DIRS["open_interest"] / f"{sym}.parquet")
            rep["oi"] = validate_oi(oi)
            line += f" | perp {len(pdf):>7} | funding {len(fund):>4} | OI {len(oi):>5}"

        reports[sym] = rep
        meta[sym] = {
            "spot_clean": ohlcv_is_clean(rep["spot"]),
            "spot_columns": list(sdf.columns),
            "spot_range": [rep["spot"].get("start"), rep["spot"].get("end")],
            "perp_clean": ohlcv_is_clean(rep["perp"]) if "perp" in rep else None,
        }
        flag = "OK" if rep["spot"].get("ohlc_inconsistent_bars", 1) == 0 else "CHECK"
        print(line + f"   [{flag}]")

    meta_blob = {
        "venue": "binance", "quote": "USDT", "ingested_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "window": [s_iso, e_iso], "oi_window_start": oi_start.isoformat(),
        "symbols": args.symbols, "per_symbol": meta,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (DIRS["metadata"] / "ingest_meta.json").write_text(json.dumps(meta_blob, indent=2))
    _write_audit(reports, meta_blob, args)
    print(f"\nDone in {meta_blob['elapsed_sec']}s. Audit -> docs/crypto_data_audit.md "
          f"| metadata -> data/crypto/metadata/ingest_meta.json")


def _write_audit(reports: dict, meta: dict, args) -> None:
    rows = []
    for sym, rep in reports.items():
        s = rep["spot"]
        p = rep.get("perp", {})
        f = rep.get("funding", {})
        oi = rep.get("oi", {})
        rows.append({
            "symbol": sym,
            "spot_bars": s.get("n_bars", 0),
            "spot_start": (s.get("start") or "")[:16],
            "spot_end": (s.get("end") or "")[:16],
            "coverage%": s.get("coverage_pct"),
            "gaps": s.get("n_gaps"),
            "missing_bars": s.get("missing_bars"),
            "ohlc_bad": s.get("ohlc_inconsistent_bars"),
            "neg_price": s.get("negative_or_zero_prices"),
            "zero_vol": s.get("zero_volume_bars"),
            "clean": ohlcv_is_clean(s),
            "perp_bars": p.get("n_bars"),
            "funding_n": f.get("n"),
            "funding_8h": f.get("looks_8h_aligned"),
            "oi_n": oi.get("n"),
            "oi_spacing_min": oi.get("median_spacing_minutes"),
        })
    df = pd.DataFrame(rows)
    all_clean = bool(df["clean"].all())
    cols_example = meta["per_symbol"][args.symbols[0]]["spot_columns"]

    body = f"""# Crypto data audit — Phase 1

**Generated:** {meta['ingested_at']} · **Venue:** {meta['venue']} (single venue) · **Quote:** {meta['quote']}
**Window:** {meta['window'][0][:16]} → {meta['window'][1][:16]} (UTC, 24/7 — no session logic) ·
**OI window:** from {meta['oi_window_start'][:16]} (Binance retains ~{OI_MAX_DAYS}d) · **Ingest time:** {meta['elapsed_sec']}s

## What exists
- **Spot 1m OHLCV** for all {len(args.symbols)} symbols → `data/crypto/spot_1m/`
- **USDT-perp 1m OHLCV** → `data/crypto/perp_1m/` {'(skipped: --no-perp)' if args.no_perp else ''}
- **8h funding rate** → `data/crypto/funding/`
- **5m open interest** (~{OI_MAX_DAYS}d) → `data/crypto/open_interest/`
- **Metadata** → `data/crypto/metadata/ingest_meta.json`

## Symbols covered
{', '.join(args.symbols)}

## Columns available (spot/perp bars)
`{', '.join(cols_example)}`
(`vwap` is derived as quote_volume/volume in the resampler; perp-only columns are NaN on spot.)
Funding: `funding_rate` (8h). Open interest: `open_interest`, `oi_value` (5m).

## Per-symbol summary

{df.to_markdown(index=False)}

## Data quality
- **All spot frames clean:** {'YES ✅' if all_clean else 'NO — see CHECK rows above'} (UTC, monotonic,
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
{'**Data is ingested, validated, and clean.**' if all_clean else '**Some frames flagged — review before use.**'}
Spot + perp + funding + OI are present for the 10 majors at 1m/5m. **Two items before alphas run:**
1. The window is **{args.days} days** (Phase-1 sample). For robust research, backfill more:
   `python scripts/ingest_crypto.py --days 365` (spot/perp; OI stays ~{OI_MAX_DAYS}d by API limit).
2. The equity `xs_backtest` resets its non-overlapping rebalance grid per **ET calendar day**; for
   24/7 crypto this needs a continuous/UTC-midnight reset — a small change to make when alphas run.

**Per instruction, no crypto alpha tests have been run. Awaiting approval.**
"""
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(body)


if __name__ == "__main__":
    main()
