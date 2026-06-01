"""
AlphaBase — the one code path guarantee.

This interface is called identically by:
  - mft/backtest/vectorbt_harness.py  (research)
  - mft/backtest/event_harness.py     (validation / live)

A strategy that works in backtest and breaks live because the live system
doesn't behave like the simulator is the single most common cause of death.
This interface prevents it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class AlphaBase(ABC):
    """
    Pure, stateless signal generator.

    Invariants (enforced by the look-ahead unit test):
      - Pure: no side effects, no mutable state, no global reads
      - PIT: uses only data present in `window` (index <= window.index[-1])
      - Deterministic: same window -> same result
    """

    @property
    @abstractmethod
    def lookback(self) -> int:
        """Minimum bars of history required before a valid signal can be produced."""
        ...

    @abstractmethod
    def compute_signal(self, window: pd.DataFrame) -> dict[str, float]:
        """
        Given a point-in-time data window, return target weights.

        Args:
            window: OHLCV DataFrame, DatetimeIndex in UTC.
                    Shape: (lookback+1, ≥5). Latest bar is the LAST row.
                    Columns must include: open, high, low, close, volume.

        Returns:
            dict mapping symbol -> target weight in [-1.0, 1.0].
            Positive = long, negative = short, 0.0 = flat.
            Return {} to leave current positions unchanged.

        Must NOT:
            - Read from disk, network, or global state
            - Mutate instance variables (use local variables only)
            - Access data beyond window.index[-1]
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lookback={self.lookback})"
