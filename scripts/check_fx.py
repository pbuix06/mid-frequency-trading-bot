"""
Cross-asset FX momentum sleeve — clean uncorrelated breadth (no bias landmines).

Unlike the EDGAR fundamental factors (survivorship-data ceiling), FX has no
delisting and no fundamental-data dependency, so the number is trustworthy.
Builds a 15-currency "foreign value in USD" panel (handling the XXXUSD vs USDXXX
convention), runs cross-sectional momentum (long strong / short weak currency,
dollar-neutral) via the validated vectorbt harness, and reports its correlation
with the equity book and its lift to the portfolio DSR. Lock-box enforced.

Usage:
    python scripts/check_fx.py
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

from mft.alphas import LongShortMomentum  # noqa: E402
from mft.backtest.vectorbt_harness import run_research_xs  # noqa: E402
from mft.data_layer.eodhd_ingest import LOCKBOX_CUTOFF, load_ticker  # noqa: E402
from mft.validation.dsr import deflated_sharpe_ratio, expected_max_sharpe  # noqa: E402
from mft.validation.metrics import full_metrics  # noqa: E402

PIT = ROOT / "data" / "pit"
START = pd.Timestamp("2010-01-01", tz="UTC")
# pair -> (currency, invert): foreign-in-USD = price (XXXUSD) or 1/price (USDXXX)
PAIRS = {
    "EURUSD": ("EUR", 0), "GBPUSD": ("GBP", 0), "AUDUSD": ("AUD", 0), "NZDUSD": ("NZD", 0),
    "USDJPY": ("JPY", 1), "USDCHF": ("CHF", 1), "USDCAD": ("CAD", 1), "USDSEK": ("SEK", 1),
    "USDNOK": ("NOK", 1), "USDMXN": ("MXN", 1), "USDZAR": ("ZAR", 1), "USDTRY": ("TRY", 1),
    "USDSGD": ("SGD", 1), "USDPLN": ("PLN", 1), "USDHUF": ("HUF", 1),
}


def fx_currency_panels() -> dict[str, pd.DataFrame]:
    """foreign-currency-in-USD OHLCV (close-only) for each currency."""
    out = {}
    for pair, (ccy, inv) in PAIRS.items():
        c = load_ticker(pair, PIT)["close"]
        v = (1.0 / c) if inv else c
        out[ccy] = pd.DataFrame({"open": v, "high": v, "low": v, "close": v, "volume": 1.0}, index=c.index)
    return out


def fx_momentum_returns(cost: float = 0.0005) -> pd.Series:
    multi = fx_currency_panels()
    r = run_research_xs(
        LongShortMomentum(list(multi), lookback=252, skip=21, frac=0.20),
        multi, rebalance_freq=21, commission_pct=cost, slippage_pct=cost,
        end_date=LOCKBOX_CUTOFF,
    )["returns"]
    return r[r.index >= START]


def main() -> None:
    fx = fx_momentum_returns()
    m = full_metrics(fx)
    print(f"\nFX cross-sectional momentum ({len(PAIRS)} currencies, 2010-2022):")
    print(f"  Sharpe={m['sharpe']:.3f}  CAGR={m['cagr']:.2%}  MaxDD={m['max_drawdown']:.2%}")
    print("  cost stress: " + "  ".join(
        f"{k}x={full_metrics(fx_momentum_returns(0.0005*k))['sharpe']:.3f}" for k in (1, 2, 3)))

    def ts(t):
        r = rv._tsmom_returns(t)
        return r[r.index >= START]

    sl = {"SPY": ts("SPY"), "GLD": ts("GLD"), "TLT": ts("TLT"),
          "LS": rv._longshort_returns(), "FX": fx}
    df = pd.concat(sl, axis=1).dropna()
    print("\n  Correlation of FX vs equity book:")
    print(df.corr()["FX"].round(3).to_string())

    n = rv.TrialLog(ROOT / "trials" / "trials.csv").count()
    srstd = rv._trial_sharpe_std_daily()
    bar = expected_max_sharpe(n, sr_std=srstd) * np.sqrt(252)

    def pdsr(p):
        return deflated_sharpe_ratio(float(p.mean() / p.std()), p.dropna().values,
                                     n_trials=n, sr_benchmark=expected_max_sharpe(n, sr_std=srstd))

    def book(cols):
        w = (1 / df[cols].std()) / (1 / df[cols].std()).sum()
        return (df[cols] * w).sum(axis=1)

    mom = ["SPY", "GLD", "TLT", "LS"]
    print(f"\n  Clean cross-asset book DSR (deflated bar {bar:.2f}, N={n}):")
    for lbl, cols in [("momentum only", mom), ("+ FX", mom + ["FX"])]:
        p = book(cols)
        mm = full_metrics(p)
        print(f"    {lbl:<16} Sharpe={mm['sharpe']:.2f}  MaxDD={mm['max_drawdown']:.2%}  DSR={pdsr(p):.3f}")
    print()


if __name__ == "__main__":
    main()
