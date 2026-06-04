"""
Intraday reversal through the gauntlet — does minute resolution rescue the edge?

The daily ShortTermReversal had gross Sharpe 0.39 but went NET-NEGATIVE on daily
bars (76x turnover x daily costs). The minute-frequency question: with minute
resolution and a realistic microstructure cost (you cross the half-spread every
trade), does a reversal edge survive NET?

Runs IntradayReversal per liquid name through the event harness on minute bars,
regular session only, lock-box enforced (2023-07-01). Reports GROSS vs NET
Sharpe and a cost-stress curve. Minute-frequency annualization
(252 x 390 = 98,280 bars/yr).

Honest caveats: IEX free feed (thin), naive per-bar Sharpe ignores minute
autocorrelation (optimistic), holds across the overnight gap (no session
flatten yet). This is a PROTOTYPE signal check, not a validated edge.

Usage:
    python scripts/check_intraday_reversal.py [--window 30] [--threshold 1.5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from mft.alphas import IntradayReversal  # noqa: E402
from mft.backtest.event_harness import equity_to_returns, run_event_driven  # noqa: E402
from mft.data_layer.alpaca_ingest import INTRADAY_LOCKBOX, load_intraday  # noqa: E402
from mft.validation.metrics import full_metrics  # noqa: E402

INTRADAY_DIR = ROOT / "data" / "intraday"
BARS_PER_YEAR = 252 * 390       # minute bars in a regular-session year
# Liquid-name microstructure: ~0.5bp commission + ~1bp half-spread per trade.
BASE_COMMISSION = 0.00005
BASE_SLIPPAGE = 0.0001


def regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 09:30-16:00 America/New_York bars (drop thin pre/post-market)."""
    et = df.index.tz_convert("America/New_York")
    mask = ((et.time >= pd.Timestamp("09:30").time()) & (et.time <= pd.Timestamp("16:00").time()))
    return df[mask]


def run_symbol(symbol: str, window: int, threshold: float, cost_mult: float) -> dict | None:
    try:
        df = load_intraday(symbol, INTRADAY_DIR)
    except FileNotFoundError:
        return None
    df = regular_session(df)
    df = df[df.index <= INTRADAY_LOCKBOX]
    if len(df) < 5000:
        return None
    alpha = IntradayReversal(symbol, window=window, threshold=threshold)
    state = run_event_driven(
        alpha, df, symbol,
        commission_pct=BASE_COMMISSION * cost_mult,
        slippage_pct=BASE_SLIPPAGE * cost_mult,
    )
    r = equity_to_returns(state)
    m = full_metrics(r, periods_per_year=BARS_PER_YEAR)
    return {"symbol": symbol, "sharpe": m["sharpe"], "maxdd": m["max_drawdown"],
            "fills": len(state.fills), "bars": len(df)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--threshold", type=float, default=1.5)
    args = p.parse_args()

    symbols = sorted(q.stem for q in INTRADAY_DIR.glob("*.parquet"))
    if not symbols:
        print("No intraday data. Run scripts/ingest_alpaca.py first.")
        return

    print(f"\nIntraday reversal — {len(symbols)} symbols, regular session, "
          f"lock-box {INTRADAY_LOCKBOX.date()}")
    print(f"window={args.window}m threshold={args.threshold}  "
          f"base cost {(BASE_COMMISSION+BASE_SLIPPAGE)*1e4:.1f}bp/trade\n")
    print(f"  {'symbol':<8} {'NET Sharpe':>11} {'GROSS':>8} {'2x':>7} {'3x':>7} {'fills':>8} {'MaxDD':>8}")
    print("  " + "-" * 62)

    net_sharpes = []
    for s in symbols:
        net = run_symbol(s, args.window, args.threshold, 1.0)
        if net is None:
            continue
        gross = run_symbol(s, args.window, args.threshold, 0.0)["sharpe"]
        s2 = run_symbol(s, args.window, args.threshold, 2.0)["sharpe"]
        s3 = run_symbol(s, args.window, args.threshold, 3.0)["sharpe"]
        net_sharpes.append(net["sharpe"])
        print(f"  {s:<8} {net['sharpe']:>11.3f} {gross:>8.3f} {s2:>7.3f} {s3:>7.3f} "
              f"{net['fills']:>8,} {net['maxdd']:>8.1%}")

    if net_sharpes:
        import numpy as np
        print("\n  " + "-" * 62)
        print(f"  Mean NET Sharpe across names: {np.mean(net_sharpes):.3f}")
        print("  (a reversal book equal-weights these largely-independent names)")
    print()


if __name__ == "__main__":
    main()
