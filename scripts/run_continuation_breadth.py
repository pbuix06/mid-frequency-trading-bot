"""
BREADTH TEST for afternoon continuation — the free diagnostic that decides whether
paid broad-universe data is worth buying (RESEARCH_LOG §12, option 1).

The question: is the execution-robust continuation edge (T0055, book 0.38 but
TSLA+AAPL-carried on 14 names) BROAD-but-weak (breadth will lift it to significance)
or CONCENTRATED-and-idiosyncratic (breadth won't help)? We answer it on the FREE
data by expanding to ~50 liquid names — before spending a dime.

Method (all realistic execution, lock-box < 2023-07-01):
  - per name: IntradayAfternoonContinuation -> simulate_intraday -> daily returns;
  - rank names by ex-ante realized intraday vol (the production universe rule);
  - sweep the top-K high-vol book (K = 5,10,20,30,all): does Sharpe + t-stat GROW
    with breadth (broad) or stall (concentrated)?
  - report the per-name Sharpe distribution (how many names actually contribute).

Decision rule (pre-registered here): breadth is worth paying for ONLY if the book
Sharpe rises materially with K AND the t-stat clears ~1.5 by the widest book AND a
majority of names are individually positive. Otherwise it's TSLA+AAPL — don't buy.

Usage:
    python scripts/run_continuation_breadth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from mft.alphas.intraday_continuation import IntradayAfternoonContinuation  # noqa: E402
from mft.data_layer.alpaca_ingest import INTRADAY_LOCKBOX  # noqa: E402
from mft.data_layer.intraday_provider import (  # noqa: E402
    ParquetIntradayProvider,
    enforce_intraday_lockbox,
)
from mft.execution.intraday_sim import IntradayExecConfig, simulate_intraday  # noqa: E402
from mft.features.intraday import daily_session_features  # noqa: E402
from mft.validation.metrics import full_metrics  # noqa: E402

INTRADAY_DIR = ROOT / "data" / "intraday"


def _name_daily_and_vol(bars, sym, cfg) -> tuple[pd.Series, float]:
    alpha = IntradayAfternoonContinuation(sym, decision_time="12:00")
    res = simulate_intraday(bars, alpha.target_series(bars), cfg)
    et_date = res.returns.index.tz_convert("America/New_York").date
    daily = res.returns.groupby(et_date).sum()
    daily.index = pd.to_datetime(daily.index)
    feat = daily_session_features(bars)
    vol = float(feat["intraday_ret"].std())
    return daily, vol


def _book_stats(cols: dict, names: list[str]) -> tuple[float, float, float]:
    book = pd.concat({s: cols[s] for s in names}, axis=1).mean(axis=1).dropna()
    m = full_metrics(book, periods_per_year=252)
    t = m["sharpe"] * (len(book) / 252) ** 0.5
    return m["sharpe"], t, m["max_drawdown"]


def main() -> None:
    syms = sorted(q.stem for q in INTRADAY_DIR.glob("*.parquet"))
    if not syms:
        print("No intraday data.")
        return
    provider = ParquetIntradayProvider(INTRADAY_DIR, source_tag="alpaca-iex")
    cfg = IntradayExecConfig(slippage_bps=1.5, assumed_spread_bps=0.0,
                             max_participation=1.0, flatten_at_close=True)

    print(f"\nCONTINUATION BREADTH TEST — {len(syms)} names, realistic exec, "
          f"lock-box {INTRADAY_LOCKBOX.date()}\n")

    cols, vols, sharpes = {}, {}, {}
    for sym in syms:
        bars = enforce_intraday_lockbox(provider.get_bars(sym))
        if len(bars) < 5000:
            continue
        daily, vol = _name_daily_and_vol(bars, sym, cfg)
        if len(daily) < 250:
            continue
        cols[sym] = daily
        vols[sym] = vol
        sharpes[sym] = full_metrics(daily, periods_per_year=252)["sharpe"]

    ranked = sorted(vols, key=lambda s: -vols[s])   # ex-ante high-vol first
    n = len(ranked)
    npos = sum(1 for s in sharpes.values() if s > 0)
    med = float(np.median(list(sharpes.values())))
    print(f"  Usable names: {n}.  Per-name realistic Sharpe: "
          f"{npos}/{n} positive, median {med:.3f}")
    top5 = ", ".join(f"{s}:{sharpes[s]:.2f}" for s in ranked[:8])
    print(f"  Highest-vol names (Sharpe): {top5}\n")

    print(f"  {'top-K high-vol book':<22}{'Sharpe':>8}{'t-stat':>8}{'MaxDD':>8}")
    for k in (5, 10, 15, 20, 30, n):
        if k > n:
            continue
        shp, t, dd = _book_stats(cols, ranked[:k])
        print(f"  K={k:<20}{shp:>8.3f}{t:>8.2f}{dd:>8.1%}")

    print("\n  Read: if Sharpe & t-stat GROW with K -> broad edge, paid breadth justified.")
    print("        if they stall near the small-K value -> concentrated, don't buy.\n")


if __name__ == "__main__":
    main()
