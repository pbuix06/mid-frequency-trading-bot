"""
Value factor (book-to-market) through the full gauntlet — the EDGAR payoff.

Builds a PIT book-to-market panel from EDGAR fundamentals + prices, runs a
dollar-neutral value sleeve (long cheap / short expensive) through the
survivorship-free harness with liquidity-tiered cost stress, then reports its
correlation with the momentum sleeves and its standalone DSR.

The whole point: value is the classic UNCORRELATED-with-momentum family
(Asness 2013). If it's real and ~0-correlated, it's the breadth the
Gate-4-failing momentum book needed.

Window: 2010-2022 (broad survivorship-free universe), lock-box enforced.

Usage:
    python scripts/check_value.py [--min-adv 20000000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_validation as rv  # noqa: E402  (momentum sleeve return builders)

from mft.alphas import CrossSectionalFactor, TSMomentum  # noqa: E402
from mft.backtest.event_harness import equity_to_returns, run_event_driven  # noqa: E402
from mft.backtest.survivorship_harness import build_panels, run_survivorship_xs  # noqa: E402
from mft.data_layer.eodhd_ingest import LOCKBOX_CUTOFF, load_ticker  # noqa: E402
from mft.features.fundamentals import build_bm_panel  # noqa: E402
from mft.validation.metrics import full_metrics  # noqa: E402

PIT_DIR = ROOT / "data" / "pit"
EDGAR_DIR = ROOT / "data" / "edgar"
BROAD_START = pd.Timestamp("2010-01-01", tz="UTC")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-adv", type=float, default=20_000_000)
    p.add_argument("--max-names", type=int, default=2000)
    args = p.parse_args()

    # Universe = liquid pool names that ALSO have EDGAR fundamentals on disk.
    have_fund = {q.stem for q in EDGAR_DIR.glob("*.parquet") if not q.stem.startswith("_")}
    s = pd.read_parquet(PIT_DIR / "_summary.parquet")
    s["start_date"] = pd.to_datetime(s["start_date"], utc=True)
    pool = s[
        (s["start_date"] <= pd.Timestamp("2010-06-01", tz="UTC"))
        & (s["avg_dollar_vol"] >= args.min_adv)
        & (s["n_bars"] >= 500)
    ].sort_values("avg_dollar_vol", ascending=False).head(args.max_names)
    tickers = sorted(set(pool["ticker"]) & have_fund)
    print(f"\nValue study — {len(tickers)} names with prices + EDGAR fundamentals")
    print(f"Lock-box enforced at {LOCKBOX_CUTOFF.date()}")

    close_df, dvol_df = build_panels(tickers, PIT_DIR, end_date=LOCKBOX_CUTOFF)
    close_df = close_df[close_df.index >= BROAD_START]
    dvol_df = dvol_df.reindex(close_df.index)

    print("Building PIT book-to-market panel...")
    bm = build_bm_panel(close_df, EDGAR_DIR, tickers=list(close_df.columns))
    bm = bm.reindex(index=close_df.index, columns=close_df.columns)
    cov = bm.notna().any(axis=0).sum()
    print(f"B/M panel: {bm.shape[0]} bars x {bm.shape[1]} names; {cov} names with any B/M\n")

    universe = list(close_df.columns)
    value = CrossSectionalFactor(universe=universe, signal_panel=bm, frac=0.20, high_is_long=True)

    def run_at(mult):
        return run_survivorship_xs(
            value, close_df, dvol_df, rebalance_freq=21, min_dollar_vol=args.min_adv,
            tiered_cost=True, cost_multiplier=mult,
        )

    res = run_at(1.0)
    m = full_metrics(res.returns)
    print("── Value (book-to-market, dollar-neutral) ──────────────────")
    print(f"  avg universe {res.avg_universe_size:.0f}, delistings {res.n_delistings}")
    print(f"  Sharpe={m['sharpe']:.3f}  CAGR={m['cagr']:.2%}  MaxDD={m['max_drawdown']:.2%}  "
          f"Sortino={m['sortino']:.3f}")
    print("  cost stress (tiered): " + "  ".join(
        f"{mult:.0f}x={full_metrics(run_at(mult).returns)['sharpe']:.3f}" for mult in (1.0, 2.0, 3.0)))

    # Correlation vs momentum sleeves (common 2010-2022 window)
    def ts(t):
        df = load_ticker(t, PIT_DIR)
        r = equity_to_returns(run_event_driven(TSMomentum(t), df, t, end_date=LOCKBOX_CUTOFF))
        return r[r.index >= BROAD_START]
    ls = rv._longshort_returns()
    allr = pd.concat([
        ts("SPY").rename("TSMOM_SPY"), ts("GLD").rename("TSMOM_GLD"),
        ts("TLT").rename("TSMOM_TLT"), ls.rename("LongShort"),
        res.returns.rename("Value"),
    ], axis=1).dropna()
    print("\n  Correlation of Value vs existing sleeves:")
    print(allr.corr()["Value"].round(3).to_string())
    print()


if __name__ == "__main__":
    main()
