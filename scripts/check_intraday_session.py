"""
Longer-hold intraday (Path B) through the gauntlet — taker-viable?

Gap-fade and opening-range breakout on minute bars, one decision/day, held
open->close. ~1 round-trip/day so the per-trade move can clear the spread.
Per name: NET Sharpe + cost stress (1x/2x/3x of ~3bp round-trip), lock-box
enforced (2023-07-01). A book equal-weights the (largely independent) names.

Usage:
    python scripts/check_intraday_session.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from mft.backtest.intraday_session import (  # noqa: E402
    BASE_COST_PER_TRADE,
    gap_fade_returns,
    opening_range_break_returns,
)
from mft.data_layer.alpaca_ingest import INTRADAY_LOCKBOX, load_intraday  # noqa: E402
from mft.features.intraday import daily_session_features  # noqa: E402
from mft.validation.metrics import full_metrics  # noqa: E402

INTRADAY_DIR = ROOT / "data" / "intraday"


def _study(name: str, strat_fn, gap_threshold=None) -> None:
    rows = {}
    feats = {}
    for sym in sorted(q.stem for q in INTRADAY_DIR.glob("*.parquet")):
        df = load_intraday(sym, INTRADAY_DIR)
        feat = daily_session_features(df)
        feat = feat[feat.index <= INTRADAY_LOCKBOX]
        if len(feat) < 250:
            continue
        feats[sym] = feat
        kw = {"threshold": gap_threshold} if gap_threshold is not None else {}
        r = strat_fn(feat, cost_per_trade=BASE_COST_PER_TRADE, **kw)
        rows[sym] = r

    print(f"\n── {name} ──────────────────────────────────────────")
    print(f"  {'sym':<6} {'NET Sharpe':>11} {'GROSS':>7} {'2x':>7} {'3x':>7} {'CAGR':>8} {'days':>6}")
    net_book = []
    for sym, r in rows.items():
        m = full_metrics(r, periods_per_year=252)
        kw = {"threshold": gap_threshold} if gap_threshold is not None else {}
        g = full_metrics(strat_fn(feats[sym], cost_per_trade=0.0, **kw), periods_per_year=252)["sharpe"]
        s2 = full_metrics(strat_fn(feats[sym], cost_per_trade=BASE_COST_PER_TRADE * 2, **kw), periods_per_year=252)["sharpe"]
        s3 = full_metrics(strat_fn(feats[sym], cost_per_trade=BASE_COST_PER_TRADE * 3, **kw), periods_per_year=252)["sharpe"]
        net_book.append(r)
        print(f"  {sym:<6} {m['sharpe']:>11.3f} {g:>7.2f} {s2:>7.2f} {s3:>7.2f} {m['cagr']:>8.2%} {len(r):>6}")

    if net_book:
        book = pd.concat(net_book, axis=1).mean(axis=1).dropna()   # equal-weight names
        bm = full_metrics(book, periods_per_year=252)
        print(f"  {'BOOK':<6} {bm['sharpe']:>11.3f} {'':>7} {'':>7} {'':>7} {bm['cagr']:>8.2%} {len(book):>6}"
              f"   MaxDD={bm['max_drawdown']:.1%}")


def main() -> None:
    if not list(INTRADAY_DIR.glob("*.parquet")):
        print("No intraday data. Run scripts/ingest_alpaca.py first.")
        return
    print(f"Longer-hold intraday — lock-box {INTRADAY_LOCKBOX.date()}, "
          f"~{2*BASE_COST_PER_TRADE*1e4:.0f}bp round-trip")
    _study("Gap-fade (all gaps)", gap_fade_returns, gap_threshold=0.0)
    _study("Gap-fade (>0.5% gaps)", gap_fade_returns, gap_threshold=0.005)
    _study("Opening-range breakout", opening_range_break_returns)
    print()


if __name__ == "__main__":
    main()
