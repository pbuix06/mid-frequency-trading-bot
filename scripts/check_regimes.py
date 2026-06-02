"""
Phase 3 regime-consistency check (analysis, NOT search — no trial rows logged).

Runs each final-suite alpha once over the full research window (2000 → lockbox),
then slices the daily return series into 4 market regimes and reports per-regime
Sharpe. Positions stay continuous across boundaries (realistic), unlike
re-running the backtest per period.

Regimes (event-driven boundaries, not equal calendar splits):
  1. 2000-01 → 2007-06   Dot-com crash + recovery
  2. 2007-07 → 2012-12   Financial crisis + deleveraging
  3. 2013-01 → 2019-12   QE bull market, low vol
  4. 2020-01 → 2022-06   COVID + rate-hike shock

Interpretation:
  - 4/4 positive  → very robust
  - 3/4 positive  → acceptable; note which fails and why
  - only regime 3 → easy-money artifact, will not survive live
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from mft.alphas import LongShortMomentum, TSMomentum
from mft.backtest.event_harness import equity_to_returns, run_event_driven
from mft.backtest.vectorbt_harness import run_research_xs
from mft.data_layer.eodhd_ingest import LOCKBOX_CUTOFF, load_ticker
from mft.validation.metrics import sharpe, max_drawdown

PIT_DIR = Path(__file__).parents[1] / "data" / "pit"

REGIMES = [
    ("1. Dotcom+recovery", "2000-01-01", "2007-06-30"),
    ("2. Financial crisis", "2007-07-01", "2012-12-31"),
    ("3. QE bull / low-vol", "2013-01-01", "2019-12-31"),
    ("4. COVID+rate hikes", "2020-01-01", "2022-06-30"),
]

EQUITY_UNIVERSE = [
    "SPY", "QQQ", "IWM", "MDY", "AAPL", "MSFT", "AMZN", "NVDA",
    "JPM", "BAC", "GS", "WFC", "XOM", "CVX", "JNJ", "PFE", "WMT", "KO", "PG",
]


def _slice_sharpe(returns: pd.Series) -> None:
    """Print per-regime Sharpe / drawdown / n-days for one return series."""
    for label, start, end in REGIMES:
        lo = pd.Timestamp(start, tz="UTC")
        hi = pd.Timestamp(end, tz="UTC")
        seg = returns[(returns.index >= lo) & (returns.index <= hi)]
        if len(seg) < 60:
            print(f"    {label:<22} : (insufficient data — {len(seg)} bars)")
            continue
        s = sharpe(seg)
        dd = max_drawdown(seg)
        mark = "✓" if s > 0.3 else ("~" if s > 0 else "✗")
        print(f"    {label:<22} : Sharpe={s:>7.3f}  DD={dd:>7.2%}  ({len(seg)} bars) {mark}")


def main() -> None:
    LB = LOCKBOX_CUTOFF
    kwargs = {"slippage_pct": 0.001, "commission_pct": 0.001}
    print(f"\nRegime check — research window 2000 → {LB.date()} (lockbox enforced)\n")

    # ── Single-asset TSMOM sleeves ────────────────────────────────────────────
    for ticker in ["SPY", "GLD", "TLT"]:
        df = load_ticker(ticker, PIT_DIR)
        r = equity_to_returns(
            run_event_driven(TSMomentum(ticker), df, ticker, end_date=LB, **kwargs)
        )
        full = sharpe(r)
        print(f"  TSMomentum({ticker})  [full-period Sharpe = {full:.3f}]")
        _slice_sharpe(r)
        print()

    # ── Dollar-neutral WML sleeve ─────────────────────────────────────────────
    eq_data = {t: load_ticker(t, PIT_DIR) for t in EQUITY_UNIVERSE}
    ls = run_research_xs(
        LongShortMomentum(universe=EQUITY_UNIVERSE), eq_data,
        commission_pct=0.001, end_date=LB,
    )
    r_ls = ls["returns"]
    print(f"  LongShortMomentum  [full-period Sharpe = {sharpe(r_ls):.3f}]")
    _slice_sharpe(r_ls)
    print()


if __name__ == "__main__":
    main()
