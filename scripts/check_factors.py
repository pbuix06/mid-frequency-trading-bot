"""
Phase 4 — does stacking fundamental factors keep climbing toward Gate 4?

Builds value (book-to-market) AND quality (Novy-Marx gross profitability) sleeves
from EDGAR, runs each through the survivorship harness, then assembles the full
book (4 momentum + value + quality) on the common 2010-2022 window and reports
the portfolio DSR. The diagnostic: momentum-only DSR was 0.134, +value lifted it
to ~0.25 — does +quality keep the climb going?

Caveat: the EDGAR universe is survivorship-biased (current-ticker CIK map), so
fundamental-sleeve magnitudes are optimistic. The CORRELATIONS and the DIRECTION
of the portfolio effect are the robust takeaways.

Usage:
    python scripts/check_factors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_validation as rv  # noqa: E402

from mft.alphas import CrossSectionalFactor  # noqa: E402
from mft.backtest.survivorship_harness import build_panels, run_survivorship_xs  # noqa: E402
from mft.data_layer.eodhd_ingest import LOCKBOX_CUTOFF  # noqa: E402
from mft.features.fundamentals import build_bm_panel, build_gross_profitability_panel  # noqa: E402
from mft.validation.dsr import deflated_sharpe_ratio, expected_max_sharpe  # noqa: E402
from mft.validation.metrics import full_metrics  # noqa: E402

PIT = ROOT / "data" / "pit"
ED = ROOT / "data" / "edgar"
START = pd.Timestamp("2010-01-01", tz="UTC")
MIN_ADV = 20_000_000


def _factor_returns(panel_fn, high_is_long, close_df, dvol_df):
    panel = panel_fn(close_df, ED, tickers=list(close_df.columns))
    panel = panel.reindex(index=close_df.index, columns=close_df.columns)
    alpha = CrossSectionalFactor(list(close_df.columns), panel, frac=0.20, high_is_long=high_is_long)
    return run_survivorship_xs(alpha, close_df, dvol_df, rebalance_freq=21,
                               min_dollar_vol=MIN_ADV, tiered_cost=True).returns


def main() -> None:
    have = {q.stem for q in ED.glob("*.parquet") if not q.stem.startswith("_")}
    tickers = sorted(have)
    cdf, ddf = build_panels(tickers, PIT, end_date=LOCKBOX_CUTOFF)
    cdf = cdf[cdf.index >= START]
    ddf = ddf.reindex(cdf.index)

    print(f"\nFactor stack study — {len(cdf.columns)} EDGAR names, 2010-2022\n")
    r_value = _factor_returns(build_bm_panel, True, cdf, ddf).rename("Value")
    r_qual = _factor_returns(build_gross_profitability_panel, True, cdf, ddf).rename("Quality")

    def ts(t):
        s = rv._tsmom_returns(t)
        return s[s.index >= START]

    sleeves = {
        "TSMOM_SPY": ts("SPY"), "TSMOM_GLD": ts("GLD"), "TSMOM_TLT": ts("TLT"),
        "LongShort": rv._longshort_returns(), "Value": r_value, "Quality": r_qual,
    }
    df = pd.concat(sleeves, axis=1).dropna()

    print("  Per-sleeve (common window):")
    for c in df.columns:
        m = full_metrics(df[c])
        print(f"    {c:<11} Sharpe={m['sharpe']:>6.3f}  MaxDD={m['max_drawdown']:>7.2%}")

    print("\n  Correlation of the two NEW factors:")
    print("    Value vs Quality:", round(df["Value"].corr(df["Quality"]), 3))
    print("    Quality vs momentum:",
          {k: round(df["Quality"].corr(df[k]), 2) for k in ["TSMOM_SPY", "TSMOM_GLD", "TSMOM_TLT", "LongShort"]})

    n_trials = rv.TrialLog(ROOT / "trials" / "trials.csv").count()
    srstd = rv._trial_sharpe_std_daily()
    bar = expected_max_sharpe(n_trials, sr_std=srstd) * np.sqrt(252)

    def pdsr(p):
        return deflated_sharpe_ratio(float(p.mean() / p.std()), p.dropna().values,
                                     n_trials=n_trials,
                                     sr_benchmark=expected_max_sharpe(n_trials, sr_std=srstd))

    def book(cols):
        sub = df[cols]
        w = (1 / sub.std()) / (1 / sub.std()).sum()
        return (sub * w).sum(axis=1)

    mom = ["TSMOM_SPY", "TSMOM_GLD", "TSMOM_TLT", "LongShort"]
    print(f"\n  Portfolio DSR trajectory (inverse-vol, deflated bar {bar:.2f}, N={n_trials}):")
    for label, cols in [
        ("momentum only", mom),
        ("+ value", mom + ["Value"]),
        ("+ value + quality", mom + ["Value", "Quality"]),
    ]:
        p = book(cols)
        m = full_metrics(p)
        print(f"    {label:<20} Sharpe={m['sharpe']:>5.2f}  MaxDD={m['max_drawdown']:>7.2%}  "
              f"DSR={pdsr(p):.3f}")
    print()


if __name__ == "__main__":
    main()
