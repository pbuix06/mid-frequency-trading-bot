"""
Refine + pressure-test the opening-range breakout (Path B) — disciplined.

3 years of free intraday data = a TINY trial budget, so this is deliberately
minimal and economically motivated, NOT a parameter hunt:
  1. Ex-ante high-beta universe: rank names by realized intraday vol (a
     CHARACTERISTIC, not the strategy outcome) and take the top half. Intraday
     trend should live in high-vol names.
  2. Parameter PLATEAU not spike: opening range 15 / 30 / 60 min — is the edge
     stable across neighbours, or a single lucky setting?
  3. CPCV: Sharpe distribution across purged sub-period combinations, not a point.
  4. Cost stress on the book.

Lock-box (2023-07-01) enforced throughout. Honest expectation: 3 years cannot
make this significant — the question is whether it STRENGTHENS (plateau + stable
CPCV) or falls apart.

Usage:
    python scripts/refine_intraday_breakout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from mft.backtest.intraday_session import (  # noqa: E402
    BASE_COST_PER_TRADE,
    opening_range_break_returns,
)
from mft.data_layer.alpaca_ingest import INTRADAY_LOCKBOX, load_intraday  # noqa: E402
from mft.features.intraday import daily_session_features  # noqa: E402
from mft.validation.cpcv import cpcv_splits, purge_embargo  # noqa: E402
from mft.validation.metrics import full_metrics, sharpe  # noqa: E402

INTRADAY_DIR = ROOT / "data" / "intraday"


def _features(or_minutes: int) -> dict[str, pd.DataFrame]:
    out = {}
    for sym in sorted(q.stem for q in INTRADAY_DIR.glob("*.parquet")):
        f = daily_session_features(load_intraday(sym, INTRADAY_DIR), or_minutes=or_minutes)
        f = f[f.index <= INTRADAY_LOCKBOX]
        if len(f) >= 250:
            out[sym] = f
    return out


def _high_beta(feats: dict[str, pd.DataFrame], top: int) -> list[str]:
    """Rank by realized intraday vol (ex-ante characteristic), take the top half."""
    vol = {s: f["intraday_ret"].std() for s, f in feats.items()}
    return [s for s, _ in sorted(vol.items(), key=lambda kv: -kv[1])[:top]]


def _book(feats, names, or_minutes, cost) -> pd.Series:
    rs = [opening_range_break_returns(feats[s], cost_per_trade=cost) for s in names]
    return pd.concat(rs, axis=1).mean(axis=1).dropna()


def main() -> None:
    feats30 = _features(30)
    n = len(feats30)
    hb = _high_beta(feats30, top=n // 2)
    lb = [s for s in feats30 if s not in hb]
    print(f"\nUniverse {n} names. Ex-ante HIGH-beta (top intraday vol): {hb}")
    print(f"                       LOW-beta: {lb}\n")

    # 1) high-beta vs low-beta book (economic conditioning, not Sharpe-picking)
    for label, names in [("HIGH-beta book", hb), ("LOW-beta book", lb)]:
        b = _book(feats30, names, 30, BASE_COST_PER_TRADE)
        m = full_metrics(b, periods_per_year=252)
        cs = [full_metrics(_book(feats30, names, 30, BASE_COST_PER_TRADE * k),
                           periods_per_year=252)["sharpe"] for k in (2, 3)]
        print(f"  {label:<16} NET Sharpe={m['sharpe']:>6.3f}  CAGR={m['cagr']:>7.2%}  "
              f"MaxDD={m['max_drawdown']:>7.2%}  cost 2x/3x={cs[0]:.2f}/{cs[1]:.2f}")

    # 2) parameter PLATEAU: opening range 15 / 30 / 60 on the high-beta book
    print("\n  Parameter plateau (high-beta book, opening-range minutes):")
    for orm in (15, 30, 60):
        f = _features(orm)
        names = [s for s in hb if s in f]
        b = _book(f, names, orm, BASE_COST_PER_TRADE)
        print(f"    or={orm:>2}m  NET Sharpe={full_metrics(b, periods_per_year=252)['sharpe']:.3f}")

    # 3) CPCV on the high-beta book (or=30)
    book = _book(feats30, hb, 30, BASE_COST_PER_TRADE)
    vals = book.values
    splits = cpcv_splits(n=len(vals), n_groups=6, n_test_groups=2)
    path_sharpes = []
    for train_idx, test_idx in splits:
        _ = purge_embargo(train_idx, test_idx, embargo_bars=2)   # discipline (no fit here)
        seg = vals[test_idx]
        if len(seg) > 20:
            path_sharpes.append(sharpe(pd.Series(seg), periods_per_year=252))
    ps = np.array(path_sharpes)
    print(f"\n  CPCV (high-beta book, {len(ps)} paths): "
          f"median Sharpe={np.median(ps):.3f}  frac>0={np.mean(ps > 0):.0%}  "
          f"min={ps.min():.2f} max={ps.max():.2f}")
    print(f"\n  Honest: book t-stat ~ {full_metrics(book, periods_per_year=252)['sharpe']*np.sqrt(len(book)/252):.2f} "
          f"over {len(book)/252:.1f}yr — {'NOT ' if full_metrics(book, periods_per_year=252)['sharpe']*np.sqrt(len(book)/252) < 1.65 else ''}significant.")
    print()


if __name__ == "__main__":
    main()
