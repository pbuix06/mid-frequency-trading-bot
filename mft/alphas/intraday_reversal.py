"""
Intraday Reversal — short-horizon mean reversion on minute bars.

Economic rationale: over minutes, order-flow imbalance and liquidity demand push
price away from fair value; it reverts. Buying what just fell / shorting what
just spiked earns a liquidity-provision premium, and slippage is often favourable
because you trade AGAINST the move (Nagel 2012, "Evaporating Liquidity";
Lehmann 1990; Lo & MacKinlay 1990).

This is the minute-frequency cousin of the daily ShortTermReversal that had real
GROSS edge (0.39) but died on daily bars to turnover/cost. The whole question of
the intraday pivot: does minute resolution + realistic microstructure cost let a
reversal edge survive NET?

Signal: z-score of price vs a short rolling mean. Fade extremes —
  z >  threshold  (price stretched UP)   -> SHORT  (-1)
  z < -threshold  (price stretched DOWN) -> LONG   (+1)
  else flat.
Single-name, signed, stateless (the harness manages position state).
"""

from __future__ import annotations

import pandas as pd

from mft.alphas.base import AlphaBase


class IntradayReversal(AlphaBase):
    """
    Minute-bar mean reversion.

    Parameters:
        symbol:    ticker this sleeve trades.
        window:    rolling window in BARS (minutes) for the mean/std (default 30).
        threshold: enter when |z| > threshold (default 1.5).
        skip:      bars to skip at the recent end to avoid 1-bar microstructure
                   bounce (default 1) — the honest version that doesn't trade on
                   the bid-ask flicker we cannot capture live.
    """

    def __init__(self, symbol: str, window: int = 30, threshold: float = 1.5, skip: int = 1):
        self.symbol = symbol
        self._window = window
        self.threshold = threshold
        self.skip = skip

    @property
    def lookback(self) -> int:
        # enough bars for the rolling stat at the skipped point
        return self._window + self.skip + 1

    def compute_signal(self, window: pd.DataFrame) -> dict[str, float]:
        close = window["close"]
        if len(close) < self._window + self.skip + 1:
            return {self.symbol: 0.0}

        # Reference price = price `skip` bars ago (avoid the last-bar bounce).
        ref = close.shift(self.skip)
        roll = ref.rolling(self._window)
        mu = roll.mean().iloc[-1]
        sd = roll.std().iloc[-1]
        px = ref.iloc[-1]

        if pd.isna(mu) or pd.isna(sd) or sd <= 0 or pd.isna(px):
            return {self.symbol: 0.0}

        z = (px - mu) / sd
        if z > self.threshold:
            return {self.symbol: -1.0}     # stretched up -> fade short
        if z < -self.threshold:
            return {self.symbol: 1.0}      # stretched down -> fade long
        return {self.symbol: 0.0}

    def __repr__(self) -> str:
        return (
            f"IntradayReversal(symbol={self.symbol!r}, window={self._window}, "
            f"threshold={self.threshold}, skip={self.skip})"
        )
