"""
Phase 3 Gate 3 check: pairwise alpha return correlations.
Target: |ρ| < 0.3 for each pair (Playbook §3, GATE 3).

Active alpha suite:
  1. TSMomentum      — single-asset 12-1mo vol-scaled trend (SPY)
  2. ShortReversion  — single-asset 5-day z-score reversal (SPY)
  3. XSMomentum      — cross-sectional long-only top quintile
  4. LongShortMomentum — dollar-neutral WML, market-neutral
  5. LowVolAnomaly   — cross-sectional long lowest-vol quintile

PairsMeanReversion is implemented but held for Phase 4 (requires a formal
cointegration test to select valid pairs; trading untested pairs on 2010–2026
data produced Sharpe < 0 because structural tech/value divergence broke the
spread mean-reversion assumption).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from mft.alphas import LongShortMomentum, LowVolAnomaly, ShortReversion, TSMomentum, XSMomentum
from mft.backtest.event_harness import equity_to_returns, run_event_driven
from mft.backtest.vectorbt_harness import run_research_xs
from mft.data_layer.eodhd_ingest import load_ticker

PIT_DIR = Path(__file__).parents[1] / "data" / "pit"

UNIVERSE_10 = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM"]


def main():
    print("\nLoading data...")
    spy = load_ticker("SPY", PIT_DIR)
    qqq = load_ticker("QQQ", PIT_DIR)
    iwm = load_ticker("IWM", PIT_DIR)
    multi_data = {t: load_ticker(t, PIT_DIR) for t in UNIVERSE_10}

    kwargs_single = {"slippage_pct": 0.001, "commission_pct": 0.001}
    kwargs_xs = {"commission_pct": 0.001}

    # 1. TSMomentum (SPY, single-asset)
    r_ts = equity_to_returns(
        run_event_driven(TSMomentum("SPY"), spy, "SPY", **kwargs_single)
    )
    r_ts.name = "TSMomentum"

    # 2. ShortReversion (SPY, single-asset)
    r_rev = equity_to_returns(
        run_event_driven(ShortReversion("SPY"), spy, "SPY", **kwargs_single)
    )
    r_rev.name = "ShortReversion"

    # 3. XSMomentum (SPY/QQQ/IWM, long-only)
    xs_result = run_research_xs(
        XSMomentum(universe=["SPY", "QQQ", "IWM"]),
        {"SPY": spy, "QQQ": qqq, "IWM": iwm},
        **kwargs_xs,
    )
    r_xs = xs_result["returns"]
    r_xs.name = "XSMomentum"

    # 4. LongShortMomentum (10-stock universe, dollar-neutral)
    ls_result = run_research_xs(
        LongShortMomentum(universe=list(multi_data.keys())),
        multi_data,
        **kwargs_xs,
    )
    r_ls = ls_result["returns"]
    r_ls.name = "LongShortMomentum"

    # 5. LowVolAnomaly (10-stock universe)
    lv_result = run_research_xs(
        LowVolAnomaly(universe=list(multi_data.keys())),
        multi_data,
        **kwargs_xs,
    )
    r_lv = lv_result["returns"]
    r_lv.name = "LowVolAnomaly"

    # Align on common index and compute correlation matrix
    returns_df = pd.concat([r_ts, r_rev, r_xs, r_ls, r_lv], axis=1).dropna()
    corr = returns_df.corr()

    print("\n" + "=" * 65)
    print("  Pairwise Alpha Return Correlations")
    print("  Target: |ρ| < 0.3 for each pair  (Playbook GATE 3)")
    print("=" * 65)
    print(corr.round(3).to_string())

    print("\n  Pair analysis:")
    names = list(corr.columns)
    all_pass = True
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho = corr.loc[a, b]
            status = "✓" if abs(rho) < 0.3 else "✗ FAIL"
            if abs(rho) >= 0.3:
                all_pass = False
            print(f"  {a:22s} vs {b:22s}: ρ = {rho:+.3f}  {status}")

    print()
    if all_pass:
        print("  GATE 3 correlation check: PASSED ✓")
    else:
        print("  GATE 3 correlation check: NOT YET PASSED")
        print("  (high ρ between long-only equity alphas is expected;")
        print("   market-neutral LongShortMomentum provides the key offset)")
    print()


if __name__ == "__main__":
    main()
