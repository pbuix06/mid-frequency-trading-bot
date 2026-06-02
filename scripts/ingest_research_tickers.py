"""
Download / refresh the specific tickers used in Phase 3+ alpha research.

Uses RESEARCH_FROM = 2000-01-01 so the full modern regime history is captured:
  2000-2002  dot-com crash        — worst period ever for momentum
  2003-2006  credit-bubble growth
  2007-2009  financial crisis
  2010-2019  long low-vol bull
  2020       COVID crash
  2021-2022  rate-hike bear through the lock-box cutoff

Always force-overwrites existing files so start dates are correct.
Multi-asset ETFs are requested from 2000 too; EODHD returns from their
actual inception dates (TLT→2002, GLD→2004, etc.).

Usage:
    python scripts/ingest_research_tickers.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parents[1]))
from mft.data_layer.eodhd_ingest import (
    LOCKBOX_CUTOFF,
    MIN_BARS,
    RESEARCH_FROM,
    fetch_ohlcv,
    fetch_ohlcv_symbol,
    save_ticker,
)

PIT_DIR = Path(__file__).parents[1] / "data" / "pit"
console = Console()

# ── US equity (fetch_ohlcv appends .US automatically) ────────────────────────
US_EQUITY = [
    # Broad index ETFs
    "SPY", "QQQ", "IWM", "MDY",
    # Large-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
    # Other sectors
    "JPM", "BAC", "GS", "WFC",
    "XOM", "CVX",
    "JNJ", "PFE",
    "WMT", "KO", "PG",
    "TSLA",
]

# ── Multi-asset (full EODHD symbol with exchange suffix) ─────────────────────
# Internal name → EODHD symbol
MULTI_ASSET = {
    # Bonds
    "TLT":    "TLT.US",
    "IEF":    "IEF.US",
    "SHY":    "SHY.US",
    "AGG":    "AGG.US",
    "HYG":    "HYG.US",
    # Commodities
    "GLD":    "GLD.US",
    "SLV":    "SLV.US",
    "USO":    "USO.US",
    "DBA":    "DBA.US",
    "GDX":    "GDX.US",
    # FX proxy ETFs
    "UUP":    "UUP.US",
    "FXE":    "FXE.US",
    "FXY":    "FXY.US",
    "FXB":    "FXB.US",
    "FXA":    "FXA.US",
    # International equity
    "EFA":    "EFA.US",
    "EEM":    "EEM.US",
    # FX direct
    "EURUSD": "EURUSD.FOREX",
    "GBPUSD": "GBPUSD.FOREX",
    "USDJPY": "USDJPY.FOREX",
    "AUDUSD": "AUDUSD.FOREX",
    "USDCAD": "USDCAD.FOREX",
}


def _save(name: str, df, results: list) -> None:
    if df.empty or len(df) < MIN_BARS:
        results.append((name, "skip", f"thin ({len(df)} bars)"))
        console.print(f"[yellow]⚠ {name}: thin ({len(df)} bars)[/yellow]")
        return
    save_ticker(df, name, PIT_DIR)
    detail = f"{len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}"
    results.append((name, "ok", detail))
    console.print(f"  [green]✓[/green] {name:<10} {detail}")


def main() -> None:
    load_dotenv()
    key = os.getenv("EODHD_API_KEY", "")
    if not key:
        console.print("[red]EODHD_API_KEY not in .env[/red]")
        sys.exit(1)

    PIT_DIR.mkdir(parents=True, exist_ok=True)
    console.print(
        f"\n[bold]Research data refresh[/bold]  "
        f"from=[cyan]{RESEARCH_FROM}[/cyan]  "
        f"lock-box=[yellow]{LOCKBOX_CUTOFF.date()}[/yellow]  "
        f"(always force-overwrites)\n"
    )

    results: list[tuple[str, str, str]] = []

    console.print("[bold]US equity / ETF[/bold]")
    for ticker in US_EQUITY:
        console.print(f"  {ticker:<10}", end=" ")
        try:
            df = fetch_ohlcv(ticker, key, from_date=RESEARCH_FROM)
            _save(ticker, df, results)
        except Exception as e:
            results.append((ticker, "fail", str(e)[:60]))
            console.print(f"[red]✗ {e}[/red]")

    console.print("\n[bold]Multi-asset[/bold]")
    for name, symbol in MULTI_ASSET.items():
        console.print(f"  {name:<10}", end=" ")
        try:
            df = fetch_ohlcv_symbol(symbol, key, from_date=RESEARCH_FROM)
            _save(name, df, results)
        except Exception as e:
            results.append((name, "fail", str(e)[:60]))
            console.print(f"[red]✗ {e}[/red]")

    ok  = sum(1 for _, s, _ in results if s == "ok")
    skp = sum(1 for _, s, _ in results if s == "skip")
    err = sum(1 for _, s, _ in results if s == "fail")
    console.print(
        f"\n[green]✓ {ok} updated[/green]  "
        f"[yellow]{skp} skipped[/yellow]  "
        f"[red]{err} failed[/red]\n"
    )


if __name__ == "__main__":
    main()
