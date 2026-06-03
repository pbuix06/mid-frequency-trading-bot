"""
Ingest SEC EDGAR fundamentals for the liquid research universe.

Pulls companyfacts for each US common stock in the liquid pool (the same
survivorship-free universe the momentum study uses), extracts the value/quality
tags, and saves a small per-ticker parquet to data/edgar/. Resumable and polite
(stays under SEC's rate limit). ETFs/foreign names without CIKs or XBRL facts
are skipped.

Set EDGAR_UA in .env to "Your Name your@email" (SEC fair-access requirement).

Usage:
    python scripts/ingest_edgar.py                # full liquid pool
    python scripts/ingest_edgar.py --max-names 50 # quick smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

sys.path.insert(0, str(Path(__file__).parents[1]))
from mft.data_layer.edgar_ingest import (
    extract_fundamentals,
    fetch_companyfacts,
    load_cik_map,
    save_fundamentals,
)

PIT_DIR = Path(__file__).parents[1] / "data" / "pit"
EDGAR_DIR = Path(__file__).parents[1] / "data" / "edgar"
console = Console()


def liquid_universe(max_names: int, min_adv: float) -> list[str]:
    s = pd.read_parquet(PIT_DIR / "_summary.parquet")
    s["start_date"] = pd.to_datetime(s["start_date"], utc=True)
    pool = s[
        (s["start_date"] <= pd.Timestamp("2010-06-01", tz="UTC"))
        & (s["avg_dollar_vol"] >= min_adv)
        & (s["n_bars"] >= 500)
    ].sort_values("avg_dollar_vol", ascending=False).head(max_names)
    return sorted(pool["ticker"].tolist())


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--max-names", type=int, default=800)
    p.add_argument("--min-adv", type=float, default=20_000_000)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    cik_map = load_cik_map(EDGAR_DIR / "_cik_map.json")
    universe = liquid_universe(args.max_names, args.min_adv)
    existing = {q.stem for q in EDGAR_DIR.glob("*.parquet")}
    todo = [t for t in universe if args.force or t not in existing]

    console.print(
        f"Universe: {len(universe)}  |  CIK map: {len(cik_map)}  |  "
        f"on disk: {len(existing)}  |  to fetch: [yellow]{len(todo)}[/yellow]"
    )

    ok = no_cik = no_facts = empty = 0
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(), console=console,
    ) as prog:
        task = prog.add_task("EDGAR", total=len(todo))
        for ticker in todo:
            cik = cik_map.get(ticker.upper())
            if cik is None:
                no_cik += 1
                prog.advance(task)
                continue
            try:
                facts = fetch_companyfacts(cik)
            except Exception:
                facts = None
            if facts is None:
                no_facts += 1
                prog.advance(task)
                continue
            df = extract_fundamentals(facts)
            if df.empty:
                empty += 1
            else:
                save_fundamentals(df, ticker, EDGAR_DIR)
                ok += 1
            prog.advance(task)

    console.print(
        f"\n[green]✓ saved {ok}[/green]  "
        f"[yellow]no CIK {no_cik}[/yellow]  no XBRL {no_facts}  empty {empty}\n"
    )


if __name__ == "__main__":
    main()
