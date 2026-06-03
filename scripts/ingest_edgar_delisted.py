"""
Survivorship fix: ingest EDGAR fundamentals for DELISTED companies.

The current SEC ticker map omits dead companies (only 1% of our delisted names
map to it), so value/quality were measured on a survivor-only universe. This
script recovers the dead companies via NAME matching:

  delisted price ticker -> EODHD company name -> normalized exact match against
  SEC's cik-lookup-data.txt (1M historical names incl. delisted) -> CIK ->
  companyfacts -> data/edgar/{ticker}.parquet

Matching is CONSERVATIVE (exact normalized name) on purpose: a wrong match would
inject another company's fundamentals, which is worse than a known gap. Exact
matching also avoids subsidiaries (e.g. DRYSHIPS FINANCE CORP != DRYSHIPS INC).
Coverage is partial (missing names, pre-XBRL delistings, name variants) but it
materially cuts the survivorship bias. Resumable.

Usage:
    python scripts/ingest_edgar_delisted.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

sys.path.insert(0, str(Path(__file__).parents[1]))
from mft.data_layer.edgar_ingest import (
    extract_fundamentals,
    fetch_companyfacts,
    load_name_cik_map,
    normalize_company_name,
    save_fundamentals,
)

PIT_DIR = Path(__file__).parents[1] / "data" / "pit"
EDGAR_DIR = Path(__file__).parents[1] / "data" / "edgar"
console = Console()


def delisted_liquid_tickers(min_adv: float) -> list[str]:
    s = pd.read_parquet(PIT_DIR / "_summary.parquet")
    s["start_date"] = pd.to_datetime(s["start_date"], utc=True)
    s["end_date"] = pd.to_datetime(s["end_date"], utc=True)
    dead = s[
        (s["start_date"] <= pd.Timestamp("2010-06-01", tz="UTC"))
        & (s["avg_dollar_vol"] >= min_adv)
        & (s["n_bars"] >= 500)
        & (s["end_date"] < pd.Timestamp("2022-01-01", tz="UTC"))  # delisted in-window
    ]
    return sorted(dead["ticker"].tolist())


def eodhd_delisted_names(api_key: str) -> dict[str, str]:
    url = "https://eodhistoricaldata.com/api/exchange-symbol-list/US"
    r = requests.get(url, params={"api_token": api_key, "fmt": "json", "delisted": 1}, timeout=90)
    r.raise_for_status()
    return {row["Code"].upper(): (row.get("Name") or "") for row in r.json()}


def main() -> None:
    load_dotenv()
    api_key = os.getenv("EODHD_API_KEY", "")
    if not api_key:
        console.print("[red]EODHD_API_KEY not in .env[/red]")
        sys.exit(1)

    console.print("Loading SEC name->CIK map (cik-lookup-data.txt)…")
    name_cik = load_name_cik_map(EDGAR_DIR / "_cik_lookup.txt")
    console.print(f"  {len(name_cik):,} normalized company names")
    console.print("Loading EODHD delisted names…")
    eod_names = eodhd_delisted_names(api_key)

    tickers = delisted_liquid_tickers(20_000_000)
    existing = {q.stem for q in EDGAR_DIR.glob("*.parquet")}
    todo = [t for t in tickers if t not in existing]
    console.print(f"Delisted-in-window liquid names: {len(tickers)}  to fetch: [yellow]{len(todo)}[/yellow]")

    no_name = no_match = ambig = no_facts = ok = 0
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(), console=console,
    ) as prog:
        task = prog.add_task("delisted", total=len(todo))
        for ticker in todo:
            prog.advance(task)
            base = ticker.split("_")[0].upper()       # strip EODHD _old suffixes
            name = eod_names.get(base) or eod_names.get(ticker.upper())
            if not name:
                no_name += 1
                continue
            ciks = name_cik.get(normalize_company_name(name), [])
            if not ciks:
                no_match += 1
                continue
            # Resolve: keep CIKs that actually have XBRL fundamentals.
            with_facts = []
            for cik in ciks[:4]:                       # cap probes per name
                try:
                    facts = fetch_companyfacts(cik)
                except Exception:
                    facts = None
                if facts is not None:
                    df = extract_fundamentals(facts)
                    if not df.empty:
                        with_facts.append((cik, df))
            if not with_facts:
                no_facts += 1
            elif len(with_facts) > 1:
                ambig += 1                             # ambiguous -> skip (conservative)
            else:
                save_fundamentals(with_facts[0][1], ticker, EDGAR_DIR)
                ok += 1

    console.print(
        f"\n[green]✓ matched+saved {ok}[/green]  no EODHD name {no_name}  "
        f"no SEC match {no_match}  ambiguous {ambig}  no XBRL {no_facts}\n"
    )


if __name__ == "__main__":
    main()
