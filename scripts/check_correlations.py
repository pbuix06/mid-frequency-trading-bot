"""
Phase 3 Gate 3 check: pairwise alpha signal correlations.
Target: |ρ| < 0.3 between all alpha pairs (Playbook §3, GATE 3).

Computes daily returns for each alpha on SPY and prints the correlation matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from mft.alphas import ShortReversion, TSMomentum, XSMomentum
from mft.backtest.event_harness import equity_to_returns, run_event_driven
from mft.backtest.vectorbt_harness import run_research_xs
from mft.data_layer.eodhd_ingest import load_ticker

PIT_DIR = Path(__file__).parents[1] / "data" / "pit"


def main():
    spy = load_ticker("SPY", PIT_DIR)
    qqq = load_ticker("QQQ", PIT_DIR)
    iwm = load_ticker("IWM", PIT_DIR)

    kwargs = {"slippage_pct": 0.001, "commission_pct": 0.001}

    # Alpha 1: TSMomentum on SPY
    alpha_ts = TSMomentum(symbol="SPY")
    r_ts = equity_to_returns(run_event_driven(alpha_ts, spy, symbol="SPY", **kwargs))
    r_ts.name = "TSMomentum"

    # Alpha 2: ShortReversion on SPY
    alpha_rev = ShortReversion(symbol="SPY")
    r_rev = equity_to_returns(run_event_driven(alpha_rev, spy, symbol="SPY", **kwargs))
    r_rev.name = "ShortReversion"

    # Alpha 3: XSMomentum on SPY/QQQ/IWM (use SPY leg only for comparison)
    alpha_xs = XSMomentum(universe=["SPY", "QQQ", "IWM"])
    xs_result = run_research_xs(alpha_xs, {"SPY": spy, "QQQ": qqq, "IWM": iwm}, **{k: v for k, v in kwargs.items() if k != "slippage_pct"})
    r_xs = xs_result["returns"]
    r_xs.name = "XSMomentum"

    # Align on common index
    returns_df = pd.concat([r_ts, r_rev, r_xs], axis=1).dropna()
    corr = returns_df.corr()

    print("\n" + "="*60)
    print("  Pairwise Alpha Return Correlations")
    print("  Target: |ρ| < 0.3 for each pair")
    print("="*60)
    print(corr.round(3).to_string())

    print("\n  Pair analysis:")
    pairs = [
        ("TSMomentum", "ShortReversion"),
        ("TSMomentum", "XSMomentum"),
        ("ShortReversion", "XSMomentum"),
    ]
    all_pass = True
    for a, b in pairs:
        rho = corr.loc[a, b]
        status = "✓" if abs(rho) < 0.3 else "✗ FAIL"
        print(f"  {a} vs {b}: ρ = {rho:.3f}  {status}")
        if abs(rho) >= 0.3:
            all_pass = False

    print()
    if all_pass:
        print("  GATE 3 correlation check: PASSED")
    else:
        print("  GATE 3 correlation check: FAILED — consider replacing high-correlation alphas")
    print()


if __name__ == "__main__":
    main()
