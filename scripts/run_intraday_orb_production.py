"""
Run the FROZEN ORB candidate through the PRODUCTION path (one code path):

    ParquetIntradayProvider -> IntradayORB (AlphaBase) -> simulate_intraday (realistic
    fills) -> equal-weight book.

Purpose: measure the EXECUTION-REALISM HAIRCUT. The research session number (0.84)
used an optimistic fill (enter at the breakout LEVEL, exit at the close, flat
per-trade cost). This re-runs the SAME frozen candidate with next-bar fills, spread
crossing, and forced EOD liquidation — on the SAME free data — to see how much of
the edge is real once execution is honest.

THIS IS NOT the Step-7 validation. Step 7 needs longer/cleaner data (spec §4). This
is a one-trial plumbing + realism check on the free 3-yr data: a cheap read on
whether buying data is even worth it.

No tweaking, no parameter search — the frozen candidate, run once, with cost stress.

Usage:
    python scripts/run_intraday_orb_production.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from mft.alphas.intraday_orb import IntradayORB  # noqa: E402
from mft.backtest.intraday_session import opening_range_break_returns  # noqa: E402
from mft.data_layer.alpaca_ingest import INTRADAY_LOCKBOX  # noqa: E402
from mft.data_layer.intraday_provider import (  # noqa: E402
    ParquetIntradayProvider,
    enforce_intraday_lockbox,
)
from mft.execution.intraday_sim import IntradayExecConfig, simulate_intraday  # noqa: E402
from mft.features.intraday import daily_session_features  # noqa: E402
from mft.validation.metrics import full_metrics  # noqa: E402

INTRADAY_DIR = ROOT / "data" / "intraday"

# Frozen high-beta book (RESEARCH_LOG §11d) — used here for a LIKE-FOR-LIKE compare
# to the 0.84. The live path selects this ex-ante via select_high_vol_universe.
HIGH_BETA = ["TSLA", "NVDA", "META", "AMZN", "XOM", "AAPL", "GOOGL"]


def _daily_returns(sym: str, provider, alpha, cfg) -> pd.Series:
    bars = provider.get_bars(sym)
    bars = enforce_intraday_lockbox(bars)
    target = alpha.target_series(bars)
    res = simulate_intraday(bars, target, cfg)
    et_date = res.returns.index.tz_convert("America/New_York").date
    daily = res.returns.groupby(et_date).sum()
    daily.index = pd.to_datetime(daily.index)
    return daily, res


def _book(provider, slippage_bps: float) -> tuple[pd.Series, dict]:
    cfg = IntradayExecConfig(
        slippage_bps=slippage_bps,
        assumed_spread_bps=0.0,         # free IEX has no reliable NBBO; slippage only
        max_participation=1.0,          # liquid mega-caps, 1 round-trip/day — not binding
        flatten_at_close=True,
    )
    cols, stats = {}, {}
    for sym in HIGH_BETA:
        alpha = IntradayORB(sym, or_minutes=30)
        daily, res = _daily_returns(sym, provider, alpha, cfg)
        cols[sym] = daily
        stats[sym] = (full_metrics(daily, periods_per_year=252)["sharpe"], res.n_fills, len(daily))
    book = pd.concat(cols, axis=1).mean(axis=1).dropna()
    return book, stats


def main() -> None:
    if not list(INTRADAY_DIR.glob("*.parquet")):
        print("No intraday data. Run scripts/ingest_alpaca.py first.")
        return
    provider = ParquetIntradayProvider(INTRADAY_DIR, source_tag="alpaca-iex")

    print(f"\nPRODUCTION-PATH realism check — frozen ORB, lock-box {INTRADAY_LOCKBOX.date()}")
    print("Realistic execution: next-bar fills + slippage + forced EOD flat.")
    print("Compare vs research (optimistic at-level fill): 0.84 / 0.35 / -0.14 @ 1x/2x/3x.\n")

    book_1x, stats = _book(provider, slippage_bps=1.5)
    book_2x, _ = _book(provider, slippage_bps=3.0)
    book_3x, _ = _book(provider, slippage_bps=4.5)

    print(f"  {'sym':<7}{'NET Sharpe':>11}{'fills':>9}{'days':>7}")
    for sym, (shp, nf, nd) in stats.items():
        print(f"  {sym:<7}{shp:>11.3f}{nf:>9}{nd:>7}")

    m1 = full_metrics(book_1x, periods_per_year=252)
    print("\n  HIGH-beta BOOK (realistic NEXT-BAR taker execution):")
    print(f"    1x cost  Sharpe={m1['sharpe']:>6.3f}  CAGR={m1['cagr']:>7.2%}  MaxDD={m1['max_drawdown']:>7.2%}")
    print(f"    2x cost  Sharpe={full_metrics(book_2x, periods_per_year=252)['sharpe']:>6.3f}")
    print(f"    3x cost  Sharpe={full_metrics(book_3x, periods_per_year=252)['sharpe']:>6.3f}")
    t = m1["sharpe"] * (len(book_1x) / 252) ** 0.5
    print(f"    t-stat ~ {t:.2f} over {len(book_1x)/252:.1f}yr ({'NOT ' if t < 1.65 else ''}significant).")

    # Entry-timing sensitivity (spec §8): the edge lives between the optimistic
    # fill-at-level (research) and the next-bar taker fill. Sweep adverse entry
    # slippage on the at-level path to find the breakeven — i.e. how good the
    # breakout fill must be. This is unmeasurable on free IEX (no NBBO/tick).
    print("\n  Entry-slippage sensitivity (at-level fill + adverse bps, research path):")
    feats = {s: daily_session_features(enforce_intraday_lockbox(provider.get_bars(s)), or_minutes=30)
             for s in HIGH_BETA}
    for slip in (0, 2, 5, 10, 20):
        rs = [opening_range_break_returns(feats[s], entry_slippage_bps=slip) for s in HIGH_BETA]
        b = pd.concat(rs, axis=1).mean(axis=1).dropna()
        print(f"    entry +{slip:>2}bp adverse  Sharpe={full_metrics(b, periods_per_year=252)['sharpe']:>6.3f}")
    print()


if __name__ == "__main__":
    main()
