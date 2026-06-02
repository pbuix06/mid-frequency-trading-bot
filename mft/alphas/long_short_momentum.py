"""
Long-Short Cross-Sectional Momentum (WML — Winners Minus Losers).

Economic rationale: recent winners continue to outperform recent losers over
1–12 months; a dollar-neutral long/short implementation removes market beta
entirely, isolating the pure momentum factor premium (Jegadeesh & Titman,
JF 1993). Unlike long-only XSMomentum, the returns are market-neutral and
weakly correlated with directional trend alphas — the key diversification
benefit of this sleeve.

Signal: rank by (lookback - skip) month return; long top `frac` fraction,
short bottom `frac` fraction; equal-weight within each leg.
Dollar-neutral: total long exposure = total short exposure = 50% of equity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.alphas.base import AlphaBase


class LongShortMomentum(AlphaBase):
    """
    Dollar-neutral cross-sectional momentum (WML factor).

    The `window` passed to compute_signal must be a DataFrame whose columns
    are ticker symbols and rows are daily close prices.

    Parameters:
        universe:    Ticker symbols in the ranked universe.
        lookback:    Total lookback for return computation (default 252).
        skip:        Recent bars to skip to avoid short-term reversal (default 21).
        frac:        Fraction to go long and short (default 0.20 = top/bottom quintile).
    """

    def __init__(
        self,
        universe: list[str],
        lookback: int = 252,
        skip: int = 21,
        frac: float = 0.20,
    ):
        if not universe:
            raise ValueError("universe must not be empty")
        self.universe = list(universe)
        self._lookback = lookback
        self.skip = skip
        self.frac = frac

    @property
    def lookback(self) -> int:
        return self._lookback

    def compute_signal(self, window: pd.DataFrame) -> dict[str, float]:
        """
        window: DataFrame with columns = ticker symbols, rows = daily closes.
        Returns: +0.5/n_long for top quintile, -0.5/n_short for bottom,
                  0.0 for the rest. Dollar-neutral: sum of all weights ≈ 0.
        """
        available = [s for s in self.universe if s in window.columns]
        if len(available) < 4:
            return {s: 0.0 for s in self.universe}

        closes = window[available]
        if len(closes) < self._lookback + 1:
            return {s: 0.0 for s in self.universe}

        # (lookback - skip) month return: from bar[-lookback] to bar[-skip]
        past_price = closes.iloc[-self._lookback]
        lagged_price = closes.shift(self.skip).iloc[-1]

        ret = (lagged_price / past_price - 1).dropna()
        if ret.empty:
            return {s: 0.0 for s in self.universe}

        n_each = max(1, int(len(ret) * self.frac))
        long_syms = set(ret.nlargest(n_each).index)
        short_syms = set(ret.nsmallest(n_each).index)

        # Equal-weight within each leg; total long = total short = 0.5 × equity
        long_w = 0.5 / n_each
        short_w = -0.5 / n_each

        signals = {s: 0.0 for s in self.universe}
        for sym in long_syms:
            signals[sym] = long_w
        for sym in short_syms:
            signals[sym] = short_w
        return signals

    def __repr__(self) -> str:
        return (
            f"LongShortMomentum(n={len(self.universe)}, lookback={self._lookback}, "
            f"skip={self.skip}, frac={self.frac})"
        )
