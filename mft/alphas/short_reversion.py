"""
Short-Horizon Mean Reversion.

Economic rationale: short-term overreaction creates 1-week reversal in liquid
equities; buying extreme recent losers earns a liquidity-provision premium
(Jegadeesh, JF 1990). Slippage is often favourable because orders go against
the crowd.

Signal: long when the rolling z-score of price is below -threshold (oversold),
flat otherwise. Long-only; pairs naturally with trend alphas (negative
correlation reduces portfolio volatility).
"""

from __future__ import annotations

import pandas as pd

from mft.alphas.base import AlphaBase
from mft.features.momentum import z_score


class ShortReversion(AlphaBase):
    """
    Z-score mean reversion on a short rolling window.

    Parameters:
        window:    Rolling window in bars (default 5 ≈ 1 week).
        threshold: Enter long when z-score < -threshold (default 1.0).
    """

    def __init__(self, symbol: str, window: int = 5, threshold: float = 1.0):
        self.symbol = symbol
        self._window = window
        self.threshold = threshold

    @property
    def lookback(self) -> int:
        # Need enough bars for the rolling z-score to be valid at the last row
        return self._window * 4

    def compute_signal(self, window: pd.DataFrame) -> dict[str, float]:
        close = window["close"]
        z = z_score(close, self._window).iloc[-1]
        if pd.isna(z):
            return {self.symbol: 0.0}
        return {self.symbol: 1.0 if z < -self.threshold else 0.0}

    def __repr__(self) -> str:
        return (
            f"ShortReversion(symbol={self.symbol!r}, "
            f"window={self._window}, threshold={self.threshold})"
        )
