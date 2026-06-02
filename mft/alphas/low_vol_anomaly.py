"""
Low-Volatility Anomaly (cross-sectional).

Economic rationale: low-beta/low-volatility stocks earn higher risk-adjusted
returns than CAPM predicts; institutions cannot use sufficient leverage to
exploit the low-vol premium, creating a persistent mispricing. Managers
benchmark against the index and therefore overweight high-vol stocks, pushing
their prices up and suppressing future returns (Baker, Bradley & Wurgler,
FAJ 2011; Ang, Hodrick, Xing & Zhang, JF 2006).

Long-only: buy the lowest-volatility quintile; rebalance monthly.
Captures a different risk premium than momentum — correlation with momentum
is low because low-vol stocks are often in defensive sectors (utilities,
staples) that move counter-cyclically to high-momentum tech/growth stocks.

Signal: rank universe by rolling 20-day realized volatility, long bottom
`bottom_frac` fraction (default 20%).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.alphas.base import AlphaBase


class LowVolAnomaly(AlphaBase):
    """
    Cross-sectional low-volatility anomaly.

    The `window` passed to compute_signal must be a DataFrame whose columns
    are ticker symbols and whose rows are daily close prices.

    Parameters:
        universe:    List of ticker symbols to consider.
        vol_window:  Rolling window for realized volatility (default 20 bars).
        lookback:    Minimum history before signal activates (default 63).
        bottom_frac: Fraction of universe to go long (default 0.20 = bottom quintile).
    """

    def __init__(
        self,
        universe: list[str],
        vol_window: int = 20,
        lookback: int = 63,
        bottom_frac: float = 0.20,
    ):
        if not universe:
            raise ValueError("universe must not be empty")
        self.universe = list(universe)
        self.vol_window = vol_window
        self._lookback = lookback
        self.bottom_frac = bottom_frac

    @property
    def lookback(self) -> int:
        return self._lookback

    def compute_signal(self, window: pd.DataFrame) -> dict[str, float]:
        """
        window: DataFrame with columns = ticker symbols, rows = daily closes.
        Returns: 1.0 for lowest-volatility quintile symbols, 0.0 for others.
        """
        available = [s for s in self.universe if s in window.columns]
        if len(available) < 2:
            return {s: 0.0 for s in self.universe}

        closes = window[available]
        if len(closes) < self.vol_window + 1:
            return {s: 0.0 for s in self.universe}

        # Realized vol from the last vol_window bars of log returns
        log_ret = np.log(closes / closes.shift(1)).iloc[-self.vol_window :]
        realized_vol = log_ret.std() * np.sqrt(252)

        valid = realized_vol.dropna()
        if valid.empty:
            return {s: 0.0 for s in self.universe}

        n_long = max(1, int(len(valid) * self.bottom_frac))
        low_vol_set = set(valid.nsmallest(n_long).index)

        signals = {s: 0.0 for s in self.universe}
        for sym in low_vol_set:
            signals[sym] = 1.0
        return signals

    def __repr__(self) -> str:
        return (
            f"LowVolAnomaly(n={len(self.universe)}, vol_window={self.vol_window}, "
            f"bottom_frac={self.bottom_frac})"
        )
