"""
Ingest Alpaca minute bars for a small LIQUID universe (intraday prototype).

Free tier = IEX feed, ~2016+, not survivorship-clean — so we deliberately use a
handful of mega-cap names that were liquid throughout, where the IEX-thinness and
survivorship issues are smallest. This is a scoped proof-of-concept to test
whether intraday resolution rescues the reversal edge, NOT a broad clean study.

Setup: add ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY to .env (free account).

Usage:
    python scripts/ingest_alpaca.py
    python scripts/ingest_alpaca.py --start 2018-01-01 --symbols AAPL MSFT SPY
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parents[1]))
from mft.data_layer.alpaca_ingest import fetch_minute_bars, save_intraday

INTRADAY_DIR = Path(__file__).parents[1] / "data" / "intraday"
console = Console()

# Mega-cap, liquid-throughout names + the most liquid ETFs (least bad on IEX).
DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "JPM", "XOM", "JNJ", "WMT", "KO",
]


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--feed", default="iex", choices=["iex", "sip"])
    args = p.parse_args()

    INTRADAY_DIR.mkdir(parents=True, exist_ok=True)
    console.print(
        f"Alpaca minute bars  feed=[cyan]{args.feed}[/cyan]  "
        f"{args.start} → {args.end}  ({len(args.symbols)} symbols)\n"
    )
    ok = fail = 0
    for sym in args.symbols:
        console.print(f"  {sym:<8}", end=" ")
        try:
            df = fetch_minute_bars(sym, args.start, args.end, feed=args.feed)
            if df.empty:
                console.print("[yellow]empty[/yellow]")
                fail += 1
                continue
            save_intraday(df, sym, INTRADAY_DIR)
            console.print(
                f"[green]✓[/green] {len(df):>8,} bars  "
                f"{df.index[0].date()} → {df.index[-1].date()}"
            )
            ok += 1
        except Exception as e:
            console.print(f"[red]✗ {str(e)[:70]}[/red]")
            fail += 1

    console.print(f"\n[green]✓ {ok} saved[/green]  [red]{fail} failed[/red]\n")


if __name__ == "__main__":
    main()
