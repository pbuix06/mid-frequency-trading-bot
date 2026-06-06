"""
Afternoon-continuation sleeve through the PRODUCTION path, REALISTIC execution.

The disciplined response to ORB's death (RESEARCH_LOG §11e): an intraday-momentum
edge whose alpha lives over a multi-hour hold and enters at a SCHEDULED time, so it
should be execution-ROBUST where ORB was fragile.

    ParquetIntradayProvider -> IntradayAfternoonContinuation (AlphaBase)
      -> simulate_intraday (next-bar fills, slippage, EOD flat) -> equal-weight book

Reports, all lock-box-safe (< 2023-07-01):
  - per-name realistic NET Sharpe, high-beta book, all-14 book;
  - cost stress 1x/2x/3x;
  - decision-time plateau (11:00 / 12:00 / 13:00) — robustness, not best-of;
  - OPTIMISTIC vs REALISTIC haircut — the key contrast with ORB (ORB lost ~0.82;
    a robust edge should lose almost nothing).

Built realistic from the start. One trial logged separately. No parameter hunt.

Usage:
    python scripts/run_intraday_continuation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from mft.alphas.intraday_continuation import IntradayAfternoonContinuation  # noqa: E402
from mft.data_layer.alpaca_ingest import INTRADAY_LOCKBOX  # noqa: E402
from mft.data_layer.intraday_provider import (  # noqa: E402
    ParquetIntradayProvider,
    enforce_intraday_lockbox,
)
from mft.execution.intraday_sim import IntradayExecConfig, simulate_intraday  # noqa: E402
from mft.validation.metrics import full_metrics  # noqa: E402

INTRADAY_DIR = ROOT / "data" / "intraday"
HIGH_BETA = ["TSLA", "NVDA", "META", "AMZN", "XOM", "AAPL", "GOOGL"]
ALL_14 = ["AAPL", "AMZN", "GOOGL", "IWM", "JNJ", "JPM", "KO",
          "META", "MSFT", "NVDA", "SPY", "TSLA", "WMT", "XOM"]


def _realistic_daily(bars, alpha, cfg) -> tuple[pd.Series, int]:
    target = alpha.target_series(bars)
    res = simulate_intraday(bars, target, cfg)
    et_date = res.returns.index.tz_convert("America/New_York").date
    daily = res.returns.groupby(et_date).sum()
    daily.index = pd.to_datetime(daily.index)
    return daily, res.n_fills


def _optimistic_daily(bars, decision_time="12:00") -> pd.Series:
    """Per-session OPTIMISTIC fill: enter at the decision-time price, hold to close.
    Measures the SAME signal (sign of morning return) under a best-case fill."""
    et = bars.index.tz_convert("America/New_York")
    dec_t = pd.Timestamp(decision_time).time()
    df = bars.copy()
    df["d"] = et.date
    df["t"] = et.time
    rows = {}
    for day, g in df.groupby("d", sort=True):
        g = g.sort_index()
        morning = g[g["t"] < dec_t]
        after = g[g["t"] >= dec_t]
        if len(morning) < 30 or len(after) < 2:
            continue
        sess_open = float(g["open"].iloc[0])
        dec_px = float(morning["close"].iloc[-1])
        close_px = float(after["close"].iloc[-1])
        if sess_open <= 0 or dec_px <= 0:
            continue
        direction = np.sign(dec_px / sess_open - 1.0)
        rows[pd.Timestamp(day)] = direction * (close_px / dec_px - 1.0) - 2 * 1.5e-4
    return pd.Series(rows).sort_index()


def _book(provider, names, alpha_factory, slippage_bps) -> tuple[pd.Series, dict]:
    cfg = IntradayExecConfig(
        slippage_bps=slippage_bps, assumed_spread_bps=0.0,
        max_participation=1.0, flatten_at_close=True,
    )
    cols, stats = {}, {}
    for sym in names:
        bars = enforce_intraday_lockbox(provider.get_bars(sym))
        daily, nf = _realistic_daily(bars, alpha_factory(sym), cfg)
        cols[sym] = daily
        stats[sym] = (full_metrics(daily, periods_per_year=252)["sharpe"], nf, len(daily))
    return pd.concat(cols, axis=1).mean(axis=1).dropna(), stats


def main() -> None:
    if not list(INTRADAY_DIR.glob("*.parquet")):
        print("No intraday data. Run scripts/ingest_alpaca.py first.")
        return
    provider = ParquetIntradayProvider(INTRADAY_DIR, source_tag="alpaca-iex")

    def mk(sym, dt="12:00"):
        return IntradayAfternoonContinuation(sym, decision_time=dt)

    print(f"\nAFTERNOON CONTINUATION — realistic execution, lock-box {INTRADAY_LOCKBOX.date()}")
    print("Decide at 12:00 ET from the morning return, hold to close. Next-bar taker fills.\n")

    hb, stats = _book(provider, HIGH_BETA, mk, slippage_bps=1.5)
    print(f"  {'sym':<7}{'NET Sharpe':>11}{'fills':>9}{'days':>7}")
    for sym, (shp, nf, nd) in stats.items():
        print(f"  {sym:<7}{shp:>11.3f}{nf:>9}{nd:>7}")

    m = full_metrics(hb, periods_per_year=252)
    t = m["sharpe"] * (len(hb) / 252) ** 0.5
    print(f"\n  HIGH-beta BOOK (realistic): Sharpe={m['sharpe']:.3f}  CAGR={m['cagr']:.2%}  "
          f"MaxDD={m['max_drawdown']:.2%}  t~{t:.2f} over {len(hb)/252:.1f}yr")
    hb2, _ = _book(provider, HIGH_BETA, mk, 3.0)
    hb3, _ = _book(provider, HIGH_BETA, mk, 4.5)
    print(f"    cost 2x Sharpe={full_metrics(hb2, periods_per_year=252)['sharpe']:.3f}   "
          f"3x Sharpe={full_metrics(hb3, periods_per_year=252)['sharpe']:.3f}")

    all14, _ = _book(provider, ALL_14, mk, 1.5)
    ma = full_metrics(all14, periods_per_year=252)
    print(f"  ALL-14 BOOK (realistic):   Sharpe={ma['sharpe']:.3f}  CAGR={ma['cagr']:.2%}  "
          f"MaxDD={ma['max_drawdown']:.2%}")

    print("\n  Decision-time plateau (high-beta book, realistic):")
    for dt in ("11:00", "12:00", "13:00"):
        b, _ = _book(provider, HIGH_BETA, lambda s, d=dt: mk(s, d), 1.5)
        print(f"    decide {dt}  Sharpe={full_metrics(b, periods_per_year=252)['sharpe']:.3f}")

    # Execution haircut: optimistic (fill at decision price) vs realistic (next-bar).
    # For a robust edge this gap is SMALL (ORB lost ~0.82 here).
    opt_cols = {s: _optimistic_daily(enforce_intraday_lockbox(provider.get_bars(s))) for s in HIGH_BETA}
    opt_book = pd.concat(opt_cols, axis=1).mean(axis=1).dropna()
    print("\n  Execution haircut (high-beta book):")
    print(f"    OPTIMISTIC (fill at decision price) Sharpe={full_metrics(opt_book, periods_per_year=252)['sharpe']:.3f}")
    print(f"    REALISTIC  (next-bar taker)         Sharpe={m['sharpe']:.3f}")
    print()


if __name__ == "__main__":
    main()
