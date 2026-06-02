"""
Survivorship-free cross-sectional study (pre-lockbox, research-only).

Re-measures the cross-sectional equity alphas (LongShortMomentum, XSMomentum) on
a broad, survivorship-FREE liquid universe — including the ~35% of liquid names
that delisted before 2022 — instead of the 19 hand-picked survivors used in the
Phase 3 candidate screen.

Window is ~2010-2022 because the broad 41k ingest starts 2010 (only the
hand-picked research names go back to 2000). Still survivorship-free and far
broader than 19 names. Lock-box (2022-07-01+) is never touched.

Reports per alpha: Sharpe, CAGR, max DD, avg universe size, # delistings
realized, annualized turnover, and a 1x/2x/3x cost-stress curve.

Usage:
    python scripts/check_survivorship.py [--max-names 800] [--min-adv 20000000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from mft.alphas import LongShortMomentum, XSMomentum
from mft.backtest.survivorship_harness import build_panels, run_survivorship_xs
from mft.data_layer.eodhd_ingest import LOCKBOX_CUTOFF
from mft.validation.diagnostics import cost_stress_curve, turnover_from_weights
from mft.validation.metrics import full_metrics

PIT_DIR = Path(__file__).parents[1] / "data" / "pit"


def candidate_pool(max_names: int, min_adv: float) -> list[str]:
    s = pd.read_parquet(PIT_DIR / "_summary.parquet")
    s["start_date"] = pd.to_datetime(s["start_date"], utc=True)
    pool = s[
        (s["start_date"] <= pd.Timestamp("2010-06-01", tz="UTC"))
        & (s["avg_dollar_vol"] >= min_adv)
        & (s["n_bars"] >= 500)
    ].copy()
    # Keep the most liquid `max_names`; PIT eligibility still gates actual membership.
    pool = pool.sort_values("avg_dollar_vol", ascending=False).head(max_names)
    return sorted(pool["ticker"].tolist())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-names", type=int, default=800)
    p.add_argument("--min-adv", type=float, default=20_000_000)
    args = p.parse_args()

    tickers = candidate_pool(args.max_names, args.min_adv)
    print(f"\nSurvivorship-free study — lock-box enforced at {LOCKBOX_CUTOFF.date()}")
    print(f"Candidate pool: {len(tickers)} liquid names (ADV>=${args.min_adv/1e6:.0f}M, data by 2010)")
    print("Building panels (no cross-death forward-fill)...")
    close_df, dvol_df = build_panels(tickers, PIT_DIR, end_date=LOCKBOX_CUTOFF)
    print(f"Panel: {close_df.shape[0]} bars x {close_df.shape[1]} names, "
          f"{close_df.index[0].date()} -> {close_df.index[-1].date()}\n")

    universe = list(close_df.columns)

    specs = [
        ("LongShortMomentum", LongShortMomentum(universe=universe, lookback=252, skip=21, frac=0.20)),
        ("XSMomentum",        XSMomentum(universe=universe, lookback=252, skip=21, top_frac=0.20)),
    ]

    for name, alpha in specs:
        def run_fn(commission_pct, slippage_pct, _alpha=alpha):
            return run_survivorship_xs(
                _alpha, close_df, dvol_df,
                rebalance_freq=21, min_dollar_vol=args.min_adv,
                commission_pct=commission_pct, slippage_pct=slippage_pct,
            ).returns

        res = run_survivorship_xs(
            alpha, close_df, dvol_df,
            rebalance_freq=21, min_dollar_vol=args.min_adv,
            commission_pct=0.001, slippage_pct=0.001,
        )
        m = full_metrics(res.returns)
        turn = turnover_from_weights(res.weights)
        stress = cost_stress_curve(run_fn, multipliers=(1.0, 2.0, 3.0))

        print(f"── {name} ──────────────────────────────────────────────")
        print(f"  avg universe: {res.avg_universe_size:.0f} names   "
              f"delistings realized: {res.n_delistings}")
        print(f"  Sharpe={m['sharpe']:.3f}  CAGR={m['cagr']:.2%}  "
              f"MaxDD={m['max_drawdown']:.2%}  Sortino={m['sortino']:.3f}")
        print(f"  annualized two-way turnover: {turn['annualized']:.1f}x")
        print("  cost stress:")
        for r in stress:
            flag = "✓" if r["survives"] else "✗"
            print(f"    {r['multiplier']:.0f}x ({r['cost_pct']*100:.1f}%/leg): "
                  f"Sharpe={r['sharpe']:>7.3f}  {flag}")
        print()


if __name__ == "__main__":
    main()
