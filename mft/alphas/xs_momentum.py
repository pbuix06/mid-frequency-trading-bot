"""
Cross-Sectional Momentum (XSMomentum).

Economic rationale: recent winners continue to outperform recent losers over
the next 1–12 months; documented across 40+ countries and asset classes
(Jegadeesh & Titman, JF 1993; Asness, Moskowitz & Pedersen, JF 2013).

Design: takes a multi-symbol close-price DataFrame (columns = ticker symbols).
Returns long (1.0) for the top quintile, flat (0.0) for the rest.
The signal is computed cross-sectionally — every symbol is ranked against all
others at each bar.

Single-asset harnesses cannot run this alpha; use run_research_xs() in the
vectorbt harness. Parity across event and NautilusTrader harnesses requires
multi-asset harness support (Phase 3+).
"""

from __future__ import annotations

import pandas as pd

from mft.alphas.base import AlphaBase


class XSMomentum(AlphaBase):
    """
    Cross-sectional momentum: long top quintile by (lookback - skip) return.

    The `window` passed to compute_signal must be a DataFrame whose COLUMNS
    are ticker symbols and whose rows are daily closes.

    Parameters:
        universe:    List of ticker symbols in the universe.
        lookback:    Bars for return computation (default 252 ≈ 12 months).
        skip:        Recent bars to exclude (default 21 ≈ 1 month).
        top_frac:    Fraction to go long (default 0.20 = top quintile).
    """

    def __init__(
        self,
        universe: list[str],
        lookback: int = 252,
        skip: int = 21,
        top_frac: float = 0.20,
    ):
        if not universe:
            raise ValueError("universe must contain at least one symbol")
        self.universe = list(universe)
        self._lookback = lookback
        self.skip = skip
        self.top_frac = top_frac

    @property
    def lookback(self) -> int:
        return self._lookback

    def compute_signal(self, window: pd.DataFrame) -> dict[str, float]:
        """
        window: DataFrame with columns = ticker symbols, rows = daily closes.
        Returns: dict mapping each symbol to 1.0 (long) or 0.0 (flat).
        """
        # Only use symbols present in the window
        available = [s for s in self.universe if s in window.columns]
        if len(available) < 2:
            return {s: 0.0 for s in self.universe}

        closes = window[available]
        if len(closes) < self._lookback + 1:
            return {s: 0.0 for s in self.universe}

        # (lookback - skip) month return, skip most recent month.
        # Use shift so the denominator is exactly t-lookback, matching TSMomentum.
        past_price = closes.shift(self._lookback).iloc[-1]
        lagged_price = closes.shift(self.skip).iloc[-1]
        ret_series = (lagged_price / past_price - 1).dropna()
        if ret_series.empty:
            return {s: 0.0 for s in self.universe}

        # Rank and select top quintile; equal weight summing to 1.0 (no leverage)
        n_long = max(1, int(len(ret_series) * self.top_frac))
        top_symbols = set(ret_series.nlargest(n_long).index)
        long_w = 1.0 / n_long

        signals = {s: 0.0 for s in self.universe}
        for sym in top_symbols:
            signals[sym] = long_w
        return signals

    def __repr__(self) -> str:
        return (
            f"XSMomentum(n={len(self.universe)}, lookback={self._lookback}, "
            f"skip={self.skip}, top_frac={self.top_frac})"
        )
