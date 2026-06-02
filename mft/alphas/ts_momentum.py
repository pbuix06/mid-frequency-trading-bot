"""
Time-Series Momentum (TSMOM).

Economic rationale: past 12-month returns positively predict the next month's
return across asset classes; persistence documented in 58 liquid instruments
(Moskowitz, Ooi & Pedersen, JFE 2012).

Signal: long when the (lookback - skip) month return is positive, sized by
target_vol / realized_vol so each position contributes equal risk regardless
of the volatility regime. Flat otherwise (long-only implementation).
"""

from __future__ import annotations

import pandas as pd

from mft.alphas.base import AlphaBase
from mft.features.momentum import rolling_volatility, time_series_momentum


class TSMomentum(AlphaBase):
    """
    Volatility-scaled time-series momentum.

    Parameters:
        lookback:   Total lookback in bars (default 252 ≈ 12 months).
        skip:       Bars to skip at the recent end (default 21 ≈ 1 month).
                    Avoids contamination from short-term reversal.
        vol_window: Rolling window for realized volatility (default 63 ≈ 3 months).
        target_vol: Annualised vol target for position sizing (default 0.15).
    """

    def __init__(
        self,
        symbol: str,
        lookback: int = 252,
        skip: int = 21,
        vol_window: int = 63,
        target_vol: float = 0.15,
    ):
        self.symbol = symbol
        self._lookback = lookback
        self.skip = skip
        self.vol_window = vol_window
        self.target_vol = target_vol

    @property
    def lookback(self) -> int:
        return self._lookback

    def compute_signal(self, window: pd.DataFrame) -> dict[str, float]:
        close = window["close"]

        mom = time_series_momentum(close, self._lookback, self.skip).iloc[-1]
        if pd.isna(mom):
            return {self.symbol: 0.0}

        direction = 1.0 if mom > 0 else 0.0  # long-only

        # Vol-scale: size down when vol is high, up when vol is low (capped at 1)
        realized_vol = rolling_volatility(close, self.vol_window).iloc[-1]
        if pd.isna(realized_vol) or realized_vol <= 0:
            return {self.symbol: direction}

        scale = min(self.target_vol / realized_vol, 1.0)
        return {self.symbol: round(direction * scale, 6)}

    def __repr__(self) -> str:
        return (
            f"TSMomentum(symbol={self.symbol!r}, lookback={self._lookback}, "
            f"skip={self.skip}, vol_window={self.vol_window})"
        )
