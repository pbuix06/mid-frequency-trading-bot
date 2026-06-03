"""
Phase 4 — Honest Validation, step 2: portfolio-level DSR + CPCV distribution.

Step 1 (run_validation.py) showed no SINGLE sleeve clears the deflated Sharpe
bar. This step asks the real question: does COMBINING the four weakly-correlated
sleeves lift the book above the bar, and is that edge STABLE (a distribution of
Sharpes across resampled sub-periods, not one flattering point estimate)?

Everything is on the common 2010-2022 window (where all four overlap) and
lock-box-safe. Portfolio weights:
  - equal-weight  : zero estimation, zero look-ahead (the honest baseline).
  - inverse-vol   : risk-balanced; uses FULL-SAMPLE vol, a mild look-ahead we
                    flag — proper causal vol-targeting is Phase 5.

CPCV here: our alphas are parameter-free, so there is nothing to fit; CPCV
degenerates to computing the strategy Sharpe on every C(N,k) combination of
held-out sub-period groups -> a distribution. We report median, [5,95] band,
and the fraction of paths with positive Sharpe (stability).

Usage:
    python scripts/run_cpcv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import run_validation as rv  # noqa: E402  (local sibling script with the sleeve builders)

from mft.monitoring.trial_log import TrialLog  # noqa: E402
from mft.validation.cpcv import cpcv_splits  # noqa: E402
from mft.validation.dsr import deflated_sharpe_ratio, expected_max_sharpe  # noqa: E402
from mft.validation.metrics import full_metrics, sharpe  # noqa: E402

COMMON_START = pd.Timestamp("2010-01-01", tz="UTC")
TRIALS_FILE = Path(__file__).parents[1] / "trials" / "trials.csv"


def common_window_returns() -> pd.DataFrame:
    """Daily returns of all 4 sleeves aligned on the common 2010-2022 window."""
    sleeves = {
        "TSMOM_SPY": rv._tsmom_returns("SPY"),
        "TSMOM_GLD": rv._tsmom_returns("GLD"),
        "TSMOM_TLT": rv._tsmom_returns("TLT"),
        "LongShort": rv._longshort_returns(),
    }
    df = pd.concat(sleeves, axis=1)
    df = df[df.index >= COMMON_START].dropna()
    return df


def cpcv_sharpe_distribution(returns: pd.Series, n_groups: int = 8, n_test: int = 2) -> np.ndarray:
    """Sharpe on each C(n_groups, n_test) held-out sub-period combination."""
    r = returns.dropna().values
    splits = cpcv_splits(len(r), n_groups=n_groups, n_test_groups=n_test)
    out = []
    for _train, test in splits:
        seg = r[np.sort(test)]
        if len(seg) > 20:
            out.append(sharpe(pd.Series(seg)))
    return np.array(out)


def _portfolio_dsr(port: pd.Series, n_trials: int, sr_std_daily: float) -> float:
    sr_daily = float(port.mean() / port.std())
    bench = expected_max_sharpe(n_trials, sr_std=sr_std_daily)
    return deflated_sharpe_ratio(sr_daily, port.dropna().values, n_trials=n_trials,
                                 sr_benchmark=bench)


def _summarize(label: str, r: pd.Series, n_trials, sr_std_daily) -> None:
    m = full_metrics(r)
    dist = cpcv_sharpe_distribution(r)
    dsr = _portfolio_dsr(r, n_trials, sr_std_daily)
    print(f"  {label:<16} Sharpe={m['sharpe']:>6.3f}  MaxDD={m['max_drawdown']:>7.2%}  "
          f"DSR={dsr:>5.3f}")
    print(f"  {'':<16} CPCV: median={np.median(dist):>6.3f}  "
          f"[5,95]=[{np.percentile(dist,5):>6.3f},{np.percentile(dist,95):>6.3f}]  "
          f"frac+={np.mean(dist>0):.0%}  (n={len(dist)} paths)")


def main() -> None:
    n_trials = TrialLog(TRIALS_FILE).count()
    sr_std_daily = rv._trial_sharpe_std_daily()
    bench_ann = expected_max_sharpe(n_trials, sr_std=sr_std_daily) * np.sqrt(252)

    df = common_window_returns()
    print("\n" + "=" * 78)
    print("  PHASE 4 — STEP 2: portfolio DSR + CPCV distribution")
    print(f"  Common window {df.index[0].date()} → {df.index[-1].date()}  "
          f"({len(df)} bars)")
    print(f"  Deflated bar ≈ {bench_ann:.2f} ann. Sharpe (N={n_trials}). DSR pass = 0.95.")
    print("=" * 78)

    print("\n  Per-sleeve on the COMMON window (shorter than step 1 -> lower power):")
    for c in df.columns:
        _summarize(c, df[c], n_trials, sr_std_daily)

    # Portfolios
    eq_w = pd.Series(1.0 / df.shape[1], index=df.columns)
    inv_vol = (1.0 / df.std())
    iv_w = inv_vol / inv_vol.sum()
    port_eq = (df * eq_w).sum(axis=1)
    port_iv = (df * iv_w).sum(axis=1)

    print("\n  Combined portfolios:")
    _summarize("equal-weight", port_eq, n_trials, sr_std_daily)
    _summarize("inverse-vol*", port_iv, n_trials, sr_std_daily)
    print("  * inverse-vol uses full-sample vol (mild look-ahead); causal vol-")
    print("    targeting is Phase 5. Weights:", {k: round(v, 2) for k, v in iv_w.items()})

    print("\n  Read:")
    print("  - If the portfolio DSR clears 0.95, diversification earned a real edge")
    print("    no single sleeve had. If not, this suite has not earned live capital")
    print("    on this data — the honest call is better data, not more mining.")
    print("  - CPCV frac+ near 100% = the edge is stable across sub-periods;")
    print("    a wide [5,95] band straddling 0 = one period carries it.")
    print()


if __name__ == "__main__":
    main()
