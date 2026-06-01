"""
Phase 1 — Ingest US stock universe from EODHD into data/pit/.

Downloads split/dividend-adjusted daily OHLCV for active + delisted US
common stocks. Skips tickers already on disk (fully resumable).

Usage:
    # Quick start — SPY + a seed list to verify the pipeline:
    python scripts/ingest_universe.py --seed

    # Full universe (takes ~30–60 min for 8k+ tickers):
    python scripts/ingest_universe.py

    # Specific tickers only:
    python scripts/ingest_universe.py --tickers SPY QQQ AAPL MSFT

    # Limit for testing:
    python scripts/ingest_universe.py --limit 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

sys.path.insert(0, str(Path(__file__).parents[1]))

from mft.data_layer.eodhd_ingest import MIN_BARS, fetch_ohlcv, fetch_us_tickers, save_ticker

PIT_DIR = Path(__file__).parents[1] / "data" / "pit"
console = Console()

# A small seed list: major liquid names + a few confirmed-delisted tickers
# Delisted examples: ETOYS (eToys, bankrupt 2001), WCOM (WorldCom, bankrupt 2002)
SEED_TICKERS = [
    # Active — core liquid universe
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "BAC", "GS", "WFC",
    "XOM", "CVX", "JNJ", "PFE",
    "WMT", "KO", "PG",
    # Delisted — proves survivorship-bias-free
    "ETOYS", "WCOM", "LEHM", "PALM", "BBBY",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", action="store_true", help="Download seed list only (fast, ~30 tickers)")
    p.add_argument("--tickers", nargs="+", metavar="TICKER", help="Specific tickers to download")
    p.add_argument("--from-date", default="2010-01-01", metavar="DATE")
    p.add_argument("--limit", type=int, default=None, help="Cap universe size (for testing)")
    p.add_argument("--no-delisted", action="store_true", help="Skip fetching delisted tickers list")
    return p.parse_args()


def build_universe(api_key: str, args: argparse.Namespace) -> list[str]:
    if args.tickers:
        return sorted(set(args.tickers))
    if args.seed:
        return SEED_TICKERS

    console.print("Fetching active US tickers from EODHD...")
    active = fetch_us_tickers(api_key, delisted=False)
    console.print(f"  [green]{len(active)}[/green] active tickers")

    delisted: list[str] = []
    if not args.no_delisted:
        console.print("Fetching delisted US tickers from EODHD...")
        delisted = fetch_us_tickers(api_key, delisted=True)
        console.print(f"  [green]{len(delisted)}[/green] delisted tickers")

    universe = sorted(set(active) | set(delisted))
    if args.limit:
        universe = universe[: args.limit]
    return universe


def main() -> None:
    load_dotenv()
    api_key = os.getenv("EODHD_API_KEY", "")
    if not api_key:
        console.print("[red]✗ EODHD_API_KEY not found in .env[/red]")
        sys.exit(1)

    args = parse_args()
    universe = build_universe(api_key, args)

    existing = {p.stem for p in PIT_DIR.glob("*.parquet")}
    todo = [t for t in universe if t not in existing]

    console.print(
        f"\n[bold]Universe:[/bold] {len(universe)} tickers  "
        f"| Already on disk: {len(existing)}  "
        f"| To download: [yellow]{len(todo)}[/yellow]"
    )

    if not todo:
        console.print("[green]All tickers already downloaded.[/green]")
        return

    ok, skipped, failed = 0, 0, []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading", total=len(todo))

        for ticker in todo:
            try:
                df = fetch_ohlcv(ticker, api_key, from_date=args.from_date)
                if df.empty or len(df) < MIN_BARS:
                    skipped += 1
                else:
                    save_ticker(df, ticker, PIT_DIR)
                    ok += 1
            except requests.RequestException as e:
                failed.append((ticker, str(e)))
            except Exception as e:
                failed.append((ticker, str(e)))
            finally:
                progress.advance(task)

    console.print(
        f"\n[green]✓ Downloaded:[/green] {ok}  "
        f"[yellow]Skipped (thin data):[/yellow] {skipped}  "
        f"[red]Failed:[/red] {len(failed)}"
    )
    if failed:
        console.print("\nFailed tickers (first 20):")
        for t, err in failed[:20]:
            console.print(f"  [red]{t}[/red]: {err}")

    # Print lock-box info
    if ok > 0:
        from mft.data_layer.eodhd_ingest import lockbox_cutoff
        cutoff = lockbox_cutoff(PIT_DIR)
        console.print(f"\n[bold]Lock-box cutoff:[/bold] {cutoff.date()} — never use data after this date in backtests")


if __name__ == "__main__":
    main()
