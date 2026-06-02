"""
Multi-asset data ingestion — bond ETFs, commodity ETFs, FX pairs.

Downloads daily OHLCV for the cross-asset universe needed for TSMOM and
portfolio diversification. Saves to data/pit/ alongside US stocks.

Symbol mapping (internal name → EODHD symbol):
  - US ETFs use the .US exchange suffix automatically
  - FX pairs use the .FOREX exchange suffix
  Stored internally without exchange suffix (e.g. "EURUSD", not "EURUSD.FOREX")
  so they load identically to stocks via load_ticker().

Usage:
    python scripts/ingest_multiasset.py
    python scripts/ingest_multiasset.py --force     # re-download even if on disk
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parents[1]))
from mft.data_layer.eodhd_ingest import MIN_BARS, fetch_ohlcv_symbol, save_ticker

PIT_DIR = Path(__file__).parents[1] / "data" / "pit"
console = Console()

# Internal name → (EODHD symbol, description, asset class)
MULTIASSET_TARGETS: dict[str, tuple[str, str, str]] = {
    # ── Bond ETFs ──────────────────────────────────────────────────────────
    "TLT":    ("TLT.US",      "iShares 20+ Year Treasury Bond",   "Bond"),
    "IEF":    ("IEF.US",      "iShares 7-10 Year Treasury Bond",  "Bond"),
    "SHY":    ("SHY.US",      "iShares 1-3 Year Treasury Bond",   "Bond"),
    "AGG":    ("AGG.US",      "iShares US Aggregate Bond",        "Bond"),
    "HYG":    ("HYG.US",      "iShares High Yield Corporate Bond","Bond"),
    # ── Commodity ETFs ─────────────────────────────────────────────────────
    "GLD":    ("GLD.US",      "SPDR Gold Shares",                 "Commodity"),
    "SLV":    ("SLV.US",      "iShares Silver Trust",             "Commodity"),
    "USO":    ("USO.US",      "United States Oil Fund",           "Commodity"),
    "DBA":    ("DBA.US",      "Invesco DB Agriculture Fund",      "Commodity"),
    "GDX":    ("GDX.US",      "VanEck Gold Miners ETF",          "Commodity"),
    # ── FX proxy ETFs ──────────────────────────────────────────────────────
    "UUP":    ("UUP.US",      "Invesco DB USD Index Bullish",     "FX-ETF"),
    "FXE":    ("FXE.US",      "CurrencyShares Euro Trust",        "FX-ETF"),
    "FXY":    ("FXY.US",      "CurrencyShares Japanese Yen",      "FX-ETF"),
    "FXB":    ("FXB.US",      "CurrencyShares British Pound",     "FX-ETF"),
    "FXA":    ("FXA.US",      "CurrencyShares Australian Dollar", "FX-ETF"),
    # ── International equity ETFs ──────────────────────────────────────────
    "EFA":    ("EFA.US",      "iShares MSCI EAFE (Developed)",    "Equity-Intl"),
    "EEM":    ("EEM.US",      "iShares MSCI Emerging Markets",    "Equity-Intl"),
    "MDY":    ("MDY.US",      "SPDR S&P MidCap 400 ETF",         "Equity-US"),
    # ── FX direct (FOREX exchange) ─────────────────────────────────────────
    "EURUSD": ("EURUSD.FOREX","EUR/USD Spot FX",                  "FX"),
    "GBPUSD": ("GBPUSD.FOREX","GBP/USD Spot FX",                  "FX"),
    "USDJPY": ("USDJPY.FOREX","USD/JPY Spot FX",                  "FX"),
    "AUDUSD": ("AUDUSD.FOREX","AUD/USD Spot FX",                  "FX"),
    "USDCAD": ("USDCAD.FOREX","USD/CAD Spot FX",                  "FX"),
}


def main() -> None:
    load_dotenv()
    api_key = os.getenv("EODHD_API_KEY", "")
    if not api_key:
        console.print("[red]✗ EODHD_API_KEY not found in .env[/red]")
        sys.exit(1)

    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Re-download even if already on disk")
    p.add_argument("--from-date", default="2005-01-01", metavar="DATE",
                   help="Start date (default: 2005-01-01 — more history for TSMOM)")
    args = p.parse_args()

    PIT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, str, str, str]] = []  # (name, class, status, detail)

    for internal_name, (eodhd_sym, description, asset_class) in MULTIASSET_TARGETS.items():
        path = PIT_DIR / f"{internal_name}.parquet"

        if path.exists() and not args.force:
            results.append((internal_name, asset_class, "skip", "already on disk"))
            continue

        console.print(f"  Downloading [cyan]{internal_name}[/cyan] ({eodhd_sym}) …", end=" ")
        try:
            df = fetch_ohlcv_symbol(eodhd_sym, api_key, from_date=args.from_date)
        except Exception as e:
            results.append((internal_name, asset_class, "fail", str(e)[:60]))
            console.print(f"[red]✗ {e}[/red]")
            continue

        if df.empty or len(df) < MIN_BARS:
            results.append((internal_name, asset_class, "skip", f"thin data ({len(df)} bars)"))
            console.print(f"[yellow]⚠ thin ({len(df)} bars)[/yellow]")
            continue

        save_ticker(df, internal_name, PIT_DIR)
        detail = f"{len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}"
        results.append((internal_name, asset_class, "ok", detail))
        console.print(f"[green]✓[/green] {detail}")

    # Summary table
    console.print()
    table = Table(title="Multi-asset download summary", show_lines=False)
    table.add_column("Ticker", style="cyan")
    table.add_column("Class")
    table.add_column("Status")
    table.add_column("Detail")
    for name, cls, status, detail in results:
        color = {"ok": "green", "skip": "yellow", "fail": "red"}.get(status, "white")
        table.add_row(name, cls, f"[{color}]{status}[/{color}]", detail)
    console.print(table)

    ok = sum(1 for _, _, s, _ in results if s == "ok")
    skipped = sum(1 for _, _, s, _ in results if s == "skip")
    failed = sum(1 for _, _, s, _ in results if s == "fail")
    console.print(
        f"\n[green]✓ Downloaded:[/green] {ok}  "
        f"[yellow]Skipped:[/yellow] {skipped}  "
        f"[red]Failed:[/red] {failed}\n"
    )


if __name__ == "__main__":
    main()
