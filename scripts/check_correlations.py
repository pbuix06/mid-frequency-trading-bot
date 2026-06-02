"""
Phase 3 Gate 3 check: pairwise alpha return correlations.
Target: |ρ| < 0.3 for each cross-class pair (Playbook §3, GATE 3).

Final Phase 3 alpha suite (4 sleeves selected for Phase 4 validation):

  TSMOM_SPY        — directional equity trend, Sharpe 0.68
  TSMOM_GLD        — directional gold/commodity trend, Sharpe 0.65, ρ≈0 with equities
  TSMOM_TLT        — directional bond trend, Sharpe 0.06, ρ=-0.26 with equities (diversifier)
  LongShortMomentum— dollar-neutral WML factor, Sharpe 0.20, ρ<0.17 with all

Notes:
  - LowVolAnomaly (Sharpe 0.78) is implemented but ρ=0.81 with TSMOM_SPY (same market beta);
    kept as research candidate, may replace TSMOM_SPY in Phase 5.
  - TSMOM_TLT and TSMOM_IEF are highly correlated (ρ=0.80); only TLT in final suite.
  - ShortReversion hit rate ~15% on daily data; edge exists but needs intraday resolution.
  - PairsMeanReversion deferred: needs cointegration test (structural drift 2010-2026).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from mft.alphas import LongShortMomentum, TSMomentum
from mft.backtest.event_harness import equity_to_returns, run_event_driven
from mft.backtest.vectorbt_harness import run_research_xs
from mft.data_layer.eodhd_ingest import load_ticker
from mft.validation.metrics import full_metrics

PIT_DIR = Path(__file__).parents[1] / "data" / "pit"
EQUITY_UNIVERSE = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT",
    "GOOGL", "AMZN", "NVDA", "JPM", "XOM",
]


def _ts(ticker: str, kwargs: dict) -> pd.Series:
    df = load_ticker(ticker, PIT_DIR)
    r = equity_to_returns(run_event_driven(TSMomentum(ticker), df, ticker, **kwargs))
    r.name = f"TSMOM_{ticker}"
    return r


def main() -> None:
    print("\nLoading data and computing returns...")
    kwargs = {"slippage_pct": 0.001, "commission_pct": 0.001}

    # Final 4 sleeves
    r_spy = _ts("SPY", kwargs)
    r_gld = _ts("GLD", kwargs)
    r_tlt = _ts("TLT", kwargs)

    eq_data = {t: load_ticker(t, PIT_DIR) for t in EQUITY_UNIVERSE}
    ls_res = run_research_xs(
        LongShortMomentum(universe=EQUITY_UNIVERSE),
        eq_data, commission_pct=0.001,
    )
    r_ls = ls_res["returns"]
    r_ls.name = "LongShort_Eq"

    # Align
    returns_df = pd.concat([r_spy, r_gld, r_tlt, r_ls], axis=1).dropna()
    corr = returns_df.corr()

    print("\n" + "=" * 65)
    print("  Phase 3 Final Suite — Pairwise Correlations")
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

    print("\n  Per-sleeve performance (2010–2026, 1× costs):")
    for series in [r_spy, r_gld, r_tlt, r_ls]:
        m = full_metrics(series)
        flag = "✓" if m["sharpe"] > 0.5 else ("~" if m["sharpe"] > 0.2 else "✗")
        print(
            f"  {flag} {series.name:<20}: "
            f"Sharpe={m['sharpe']:>7.4f}  "
            f"CAGR={m['cagr']:>7.2%}  "
            f"MaxDD={m['max_drawdown']:>7.2%}"
        )
    print()


if __name__ == "__main__":
    main()
