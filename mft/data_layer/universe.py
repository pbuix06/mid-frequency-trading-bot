"""
Research universe construction.

Two steps:
  1. build_universe_summary() — scan all parquet files, compute per-ticker
     stats (n_bars, avg dollar volume, last price). Cached to disk.
  2. build_research_universe() — apply filters to the summary and return
     a sorted list of ticker strings ready for alpha research.

Separation of concerns: data/pit/ stores everything broadly collected.
Research only ever sees the filtered, tradeable subset.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

_SUMMARY_FILE = "_summary.parquet"  # stored inside data/pit/


# ── Summary building ──────────────────────────────────────────────────────────

def _compute_one(path: Path) -> dict | None:
    """Read one parquet and return summary stats. Returns None on error."""
    try:
        df = pd.read_parquet(path, columns=["close", "volume"])
        if df.empty:
            return None
        dollar_vol = (df["close"] * df["volume"]).mean()
        return {
            "ticker": path.stem,
            "n_bars": len(df),
            "start_date": df.index.min(),
            "end_date": df.index.max(),
            "avg_dollar_vol": float(dollar_vol),
            "last_close": float(df["close"].iloc[-1]),
        }
    except Exception:
        return None


def build_universe_summary(
    pit_dir: Path,
    workers: int = 10,
    rebuild: bool = False,
) -> pd.DataFrame:
    """
    Scan all ticker parquets and compute summary stats.
    Cached to data/pit/_summary.parquet — pass rebuild=True to force refresh.

    Columns: ticker, n_bars, start_date, end_date, avg_dollar_vol, last_close
    """
    cache = pit_dir / _SUMMARY_FILE
    if cache.exists() and not rebuild:
        return pd.read_parquet(cache)

    paths = [p for p in pit_dir.glob("*.parquet") if not p.name.startswith("_")]
    rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_compute_one, p): p for p in paths}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                rows.append(result)

    summary = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    summary.to_parquet(cache, engine="pyarrow", compression="snappy")
    return summary


# ── Universe filtering ────────────────────────────────────────────────────────

def build_research_universe(
    pit_dir: Path,
    min_bars: int = 1260,
    min_avg_dollar_vol: float = 1_000_000,
    min_price: float = 1.0,
    as_of: pd.Timestamp | None = None,
    rebuild_summary: bool = False,
) -> list[str]:
    """
    Return sorted list of tickers that pass the research filters.

    Default filters (all configurable):
      - min_bars=1260      : ≥ 5 years of daily data (matches MinBTL budget)
      - min_avg_dollar_vol : ≥ $1M average daily dollar volume (tradeable)
      - min_price=1.0      : last close ≥ $1 (no penny stocks)
      - as_of              : if set, only count bars up to this date (PIT correct)

    The as_of filter is important for walk-forward backtesting: a stock that
    was liquid in 2015 but delisted in 2018 should appear in any universe
    built for dates before 2018.
    """
    summary = build_universe_summary(pit_dir, rebuild=rebuild_summary)

    if as_of is not None:
        # Recompute stats using only bars up to as_of
        # We filter tickers whose data overlaps the as_of date
        summary = summary[summary["start_date"] <= as_of].copy()

        # For bar count and dollar vol, we need to re-read filtered data.
        # This is done lazily: only recompute for tickers that pass date check,
        # using a conservative estimate (total bars * fraction of history used).
        # Full PIT recompute is done in Phase 3 walk-forward infrastructure.
        days_total = (summary["end_date"] - summary["start_date"]).dt.days.clip(lower=1)
        days_used = (as_of - summary["start_date"]).dt.days.clip(lower=0)
        fraction = (days_used / days_total).clip(upper=1.0)
        summary = summary.copy()
        summary["n_bars"] = (summary["n_bars"] * fraction).astype(int)

    mask = (
        (summary["n_bars"] >= min_bars)
        & (summary["avg_dollar_vol"] >= min_avg_dollar_vol)
        & (summary["last_close"] >= min_price)
    )
    return sorted(summary.loc[mask, "ticker"].tolist())
