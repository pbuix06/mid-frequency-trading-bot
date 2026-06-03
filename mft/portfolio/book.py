"""
Book construction — combine the surviving sleeves into ONE netted target book.

A sleeve is an alpha that emits per-instrument target weights (e.g. TSMomentum
emits {SPY: 0.8}; LongShortMomentum emits {AAPL: +0.02, XOM: -0.02, ...}). The
book:

  1. allocates CAPITAL across sleeves by inverse volatility of their historical
     returns (equal-risk-contribution, robust for weakly-correlated sleeves), and
  2. NETS the sleeves into a single per-instrument target weight:
        book[i] = Sum_s  alloc[s] * sleeve_target[s].get(i, 0)

Dollar-neutral sleeves (Sum of weights ~ 0) stay neutral after scaling; long-only
trend sleeves keep their sign. The netted book is what the RiskManager clips and
the execution layer trades — one position per instrument, not one per sleeve.

This is the Stage-5 "combine survivors into one coherent book" step. It does NOT
decide whether the sleeves are worth trading (that is Gate 4); it assembles
whatever set you hand it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_vol_alloc(
    sleeve_returns: pd.DataFrame,
    vol_window: int | None = None,
) -> dict[str, float]:
    """
    Capital allocation across sleeves by inverse realized volatility.

    Args:
        sleeve_returns: DataFrame, columns = sleeve names, rows = daily returns.
        vol_window: if set, use the trailing `vol_window` rows; else full sample.

    Returns:
        {sleeve: weight}, weights >= 0 summing to 1. A lower-vol sleeve gets a
        larger share so each contributes comparable risk.
    """
    r = sleeve_returns.tail(vol_window) if vol_window else sleeve_returns
    vol = r.std()
    inv = 1.0 / (vol + 1e-12)
    inv = inv.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    total = inv.sum()
    if total <= 0:
        n = len(sleeve_returns.columns)
        return {s: 1.0 / n for s in sleeve_returns.columns}
    w = inv / total
    return {s: float(w[s]) for s in sleeve_returns.columns}


def net_book(
    sleeve_targets: dict[str, dict[str, float]],
    sleeve_alloc: dict[str, float],
) -> dict[str, float]:
    """
    Net the sleeves' per-instrument target weights into one book.

    Args:
        sleeve_targets: {sleeve: {instrument: weight}} — each sleeve's current
                        target weights (from its compute_signal).
        sleeve_alloc:   {sleeve: capital_fraction} from inverse_vol_alloc.

    Returns:
        {instrument: net_weight}. Instruments that net to ~0 are dropped.
    """
    book: dict[str, float] = {}
    for sleeve, targets in sleeve_targets.items():
        a = sleeve_alloc.get(sleeve, 0.0)
        if a == 0.0:
            continue
        for inst, w in targets.items():
            book[inst] = book.get(inst, 0.0) + a * w
    return {i: w for i, w in book.items() if abs(w) > 1e-9}


def book_exposure(book: dict[str, float]) -> dict[str, float]:
    """Gross (sum |w|), net (sum w), and long/short leg exposures of a book."""
    longs = sum(w for w in book.values() if w > 0)
    shorts = sum(w for w in book.values() if w < 0)
    return {
        "gross": float(sum(abs(w) for w in book.values())),
        "net": float(sum(book.values())),
        "long": float(longs),
        "short": float(shorts),
    }
