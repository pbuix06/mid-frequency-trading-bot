"""
SMA Crossover — hello-world alpha for Gate 0 parity test.

Economic rationale: price momentum is persistent; a fast MA crossing above
a slow MA signals that recent momentum is positive.

This is NOT a production alpha. Its purpose is to prove that the same
compute_signal() produces identical equity curves in vectorbt and the
event-driven harness before you trust either engine.
"""

from __future__ import annotations

import pandas as pd

from .base import AlphaBase


class SMACrossover(AlphaBase):
    """
    Long when fast SMA > slow SMA; flat otherwise.
    Single-asset, long-only.
    """

    def __init__(self, symbol: str, fast: int = 20, slow: int = 50):
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        self.symbol = symbol
        self.fast = fast
        self.slow = slow

    @property
    def lookback(self) -> int:
        return self.slow

    def compute_signal(self, window: pd.DataFrame) -> dict[str, float]:
        close = window["close"]
        fast_ma = close.rolling(self.fast).mean().iloc[-1]
        slow_ma = close.rolling(self.slow).mean().iloc[-1]

        if pd.isna(fast_ma) or pd.isna(slow_ma):
            return {self.symbol: 0.0}

        return {self.symbol: 1.0 if fast_ma > slow_ma else 0.0}

    def __repr__(self) -> str:
        return f"SMACrossover(symbol={self.symbol!r}, fast={self.fast}, slow={self.slow})"
