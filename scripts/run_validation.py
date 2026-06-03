"""
Phase 4 — Honest Validation, step 1: the Deflated Sharpe Ratio gate.

The "one canonical engine" worry dissolves: the 4 sleeves are two different
strategy TYPES (single-asset directional trend vs cross-sectional dollar-neutral),
so they cannot share one engine. The common currency is the daily RETURN SERIES.
DSR and CPCV operate on returns, not engine internals. This script produces each
sleeve's canonical, lock-box-safe return series on its maximal honest window with
its appropriate cost treatment, then runs the primary statistical kill test.

DSR (Bailey & López de Prado 2014) asks: given that we ran N trials, is this
Sharpe higher than the best we'd expect from luck alone? It deflates for trial
count, non-normality (skew/kurtosis), and sample length. Reject anything not
significant at 95%.

Per-sleeve windows are maximal (more data = more power for standalone
significance); the COMBINED portfolio is validated later on the common 2010-2022
window where all four overlap.

Usage:
    python scripts/run_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from mft.alphas import LongShortMomentum, TSMomentum
from mft.backtest.event_harness import equity_to_returns, run_event_driven
from mft.backtest.survivorship_harness import build_panels, run_survivorship_xs
from mft.data_layer.eodhd_ingest import LOCKBOX_CUTOFF, load_ticker
from mft.monitoring.trial_log import TrialLog
from mft.validation.dsr import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_backtest_length,
)
from mft.validation.metrics import full_metrics

PIT_DIR = Path(__file__).parents[1] / "data" / "pit"
TRIALS_FILE = Path(__file__).parents[1] / "trials" / "trials.csv"
BROAD_START = pd.Timestamp("2010-01-01", tz="UTC")


def _tsmom_returns(ticker: str) -> pd.Series:
    df = load_ticker(ticker, PIT_DIR)
    return equity_to_returns(
        run_event_driven(TSMomentum(ticker), df, ticker, end_date=LOCKBOX_CUTOFF)
    )


def _longshort_returns() -> pd.Series:
    s = pd.read_parquet(PIT_DIR / "_summary.parquet")
    s["start_date"] = pd.to_datetime(s["start_date"], utc=True)
    pool = s[
        (s["start_date"] <= pd.Timestamp("2010-06-01", tz="UTC"))
        & (s["avg_dollar_vol"] >= 20_000_000)
        & (s["n_bars"] >= 500)
    ].sort_values("avg_dollar_vol", ascending=False).head(800)
    cdf, ddf = build_panels(sorted(pool["ticker"]), PIT_DIR, end_date=LOCKBOX_CUTOFF)
    cdf = cdf[cdf.index >= BROAD_START]
    ddf = ddf.reindex(cdf.index)
    alpha = LongShortMomentum(universe=list(cdf.columns), lookback=252, skip=21, frac=0.20)
    return run_survivorship_xs(
        alpha, cdf, ddf, rebalance_freq=21, min_dollar_vol=20_000_000, tiered_cost=True
    ).returns


def _t_stat(sr_ann: float, n_days: int) -> float:
    """Approx t-stat of an annualized Sharpe over n daily observations."""
    return sr_ann * np.sqrt(n_days / 252.0)


def _trial_sharpe_std_daily() -> float:
    """
    Empirical std of the DAILY Sharpe across all logged trials — the V input to
    the deflated benchmark (Bailey-LdP). The trial log stores ANNUALIZED Sharpe,
    so convert by /sqrt(252). This is 'how much Sharpe varies across the things
    we tried' — the dispersion that makes a high max likely by luck.
    """
    import csv
    vals = []
    with open(TRIALS_FILE) as f:
        for row in csv.DictReader(f):
            s = row.get("is_sharpe", "").strip()
            if s:
                try:
                    v = float(s)
                    if np.isfinite(v):
                        vals.append(v)
                except ValueError:
                    pass
    ann = np.std(vals, ddof=1) if len(vals) > 1 else 1.0
    return float(ann / np.sqrt(252.0))


def _dsr_row(name, r, n_trials, sr_std_daily, minbtl):
    r = r.dropna()
    n = len(r)
    sr_ann = full_metrics(r)["sharpe"]
    sr_daily = float(r.mean() / r.std())            # per-bar Sharpe
    bench_daily = expected_max_sharpe(n_trials, sr_std=sr_std_daily)
    dsr = deflated_sharpe_ratio(sr_daily, r.values, n_trials=n_trials,
                                sr_benchmark=bench_daily)
    enough = n >= minbtl
    passes = (dsr >= 0.95) and enough
    return {
        "name": name, "n": n, "sr_ann": sr_ann, "t": _t_stat(sr_ann, n),
        "bench_ann": bench_daily * np.sqrt(252.0), "dsr": dsr,
        "enough": enough, "pass": passes,
        "window": f"{r.index[0].date()}→{r.index[-1].date()}",
    }


def main() -> None:
    n_trials = TrialLog(TRIALS_FILE).count()
    sr_std_daily = _trial_sharpe_std_daily()
    minbtl = min_backtest_length(n_trials)
    bench_ann = expected_max_sharpe(n_trials, sr_std=sr_std_daily) * np.sqrt(252.0)

    print("\n" + "=" * 80)
    print("  PHASE 4 — STEP 1: Deflated Sharpe Ratio gate")
    print(f"  N = {n_trials} trials  |  cross-trial Sharpe dispersion ≈ "
          f"{sr_std_daily * np.sqrt(252.0):.2f} ann.")
    print(f"  Deflated benchmark to beat ≈ {bench_ann:.2f} annualized Sharpe "
          f"(luck's expected max over {n_trials} trials)")
    print(f"  MinBTL(N) = {minbtl} bars ({minbtl/252:.1f} yrs). Reject if DSR < 0.95.")
    print("=" * 80)

    sleeves = {
        "TSMOM_SPY": _tsmom_returns("SPY"),
        "TSMOM_GLD": _tsmom_returns("GLD"),
        "TSMOM_TLT": _tsmom_returns("TLT"),
        "LongShort":  _longshort_returns(),
    }

    print(f"\n  {'Sleeve':<11} {'window':<20} {'Sharpe':>7} {'t':>5} "
          f"{'bar':>5} {'DSR':>6} {'verdict':>8}")
    print("  " + "-" * 70)
    rows = [_dsr_row(n, r, n_trials, sr_std_daily, minbtl) for n, r in sleeves.items()]
    for x in rows:
        print(f"  {x['name']:<11} {x['window']:<20} {x['sr_ann']:>7.3f} {x['t']:>5.2f} "
              f"{x['bench_ann']:>5.2f} {x['dsr']:>6.3f} "
              f"{'PASS' if x['pass'] else 'reject':>8}")

    survivors = [x["name"] for x in rows if x["pass"]]
    print("\n  " + "-" * 70)
    print(f"  DSR survivors at 95%: {survivors if survivors else 'NONE'}")
    print("\n  Read:")
    print("  - 'bar' is the deflated benchmark Sharpe a sleeve must beat given N trials.")
    print("    With N this large, the bar is high — modest Sharpes are expected to fail.")
    print("  - This is the playbook's reality: most candidates die at Gate 4.")
    print("  - Next: CPCV distribution (not a point estimate) + portfolio-level DSR on")
    print("    the common 2010-2022 window, where diversification may lift the book")
    print("    Sharpe above any single sleeve.")
    print()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
