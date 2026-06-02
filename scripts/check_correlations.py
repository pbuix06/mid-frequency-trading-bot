"""
Phase 3 Gate 3 check: pairwise alpha return correlations.
Target: |ρ| < 0.3 for each cross-class pair (Playbook §3, GATE 3).

Phase 3 candidate sleeve set, pre-lockbox only:

  TSMOM_SPY        — directional equity trend
  TSMOM_GLD        — directional gold/commodity trend
  TSMOM_TLT        — directional bond trend / diversifier candidate
  LongShortMomentum— dollar-neutral WML factor

Notes:
  - This is a correlation screen, not Phase 4 validation evidence.
  - LowVolAnomaly is implemented but can be highly correlated with SPY trend due market beta.
  - TSMOM_TLT is diversifying but weak on standalone return quality.
  - ShortReversion hit rate ~15% on daily data; edge exists but needs intraday resolution.
  - PairsMeanReversion deferred: needs cointegration test and structural-break checks.
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
from mft.validation.metrics import full_metrics

PIT_DIR = Path(__file__).parents[1] / "data" / "pit"
EQUITY_UNIVERSE = [
    "SPY", "QQQ", "IWM", "MDY", "AAPL", "MSFT", "AMZN", "NVDA",
    "JPM", "BAC", "GS", "WFC", "XOM", "CVX", "JNJ", "PFE", "WMT", "KO", "PG",
]


def _ts(ticker: str, kwargs: dict) -> pd.Series:
    df = load_ticker(ticker, PIT_DIR)
    r = equity_to_returns(run_event_driven(TSMomentum(ticker), df, ticker, **kwargs))
    r.name = f"TSMOM_{ticker}"
    return r


def main() -> None:
    print("\nLoading data and computing returns...")
    kwargs = {
        "slippage_pct": 0.001,
        "commission_pct": 0.001,
        "end_date": LOCKBOX_CUTOFF,
    }

    # Candidate sleeves for the Phase 3 correlation screen.
    r_spy = _ts("SPY", kwargs)
    r_gld = _ts("GLD", kwargs)
    r_tlt = _ts("TLT", kwargs)

    eq_data = {t: load_ticker(t, PIT_DIR) for t in EQUITY_UNIVERSE}
    ls_res = run_research_xs(
        LongShortMomentum(universe=EQUITY_UNIVERSE),
        eq_data,
        commission_pct=0.001,
        slippage_pct=0.001,
        end_date=LOCKBOX_CUTOFF,
    )
    r_ls = ls_res["returns"]
    r_ls.name = "LongShort_Eq"

    # Align
    returns_df = pd.concat([r_spy, r_gld, r_tlt, r_ls], axis=1).dropna()
    corr = returns_df.corr()

    print("\n" + "=" * 65)
    print("  Phase 3 Candidate Sleeve Set — Pairwise Correlations")
    print("  Target: |ρ| < 0.3  (Playbook GATE 3)")
    print("=" * 65)
    print(corr.round(3).to_string())

    print("\n  All pairs:")
    names = list(corr.columns)
    all_pass = True
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho = corr.loc[a, b]
            status = "✓" if abs(rho) < 0.3 else "✗ FAIL"
            if abs(rho) >= 0.3:
                all_pass = False
            print(f"  {a:<20} vs {b:<20}: ρ = {rho:+.3f}  {status}")

    print()
    if all_pass:
        print("  GATE 3: PASSED ✓  All pairs |ρ| < 0.3")
    else:
        print("  GATE 3: NOT PASSED")

    print(f"\n  Per-sleeve performance (individual pre-lockbox windows through {LOCKBOX_CUTOFF.date()}, 1× costs):")
    for series in [r_spy, r_gld, r_tlt, r_ls]:
        m = full_metrics(series)
        window = f"{series.index[0].date()}→{series.index[-1].date()}" if not series.empty else "empty"
        flag = "✓" if m["sharpe"] > 0.5 else ("~" if m["sharpe"] > 0.2 else "✗")
        print(
            f"  {flag} {series.name:<20}: "
            f"{window:<23}  "
            f"Sharpe={m['sharpe']:>7.4f}  "
            f"CAGR={m['cagr']:>7.2%}  "
            f"MaxDD={m['max_drawdown']:>7.2%}"
        )
    print()


if __name__ == "__main__":
    main()
