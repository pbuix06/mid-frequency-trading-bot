"""
Short-Horizon Cross-Sectional Reversal (contrarian).

Economic rationale: short-horizon overreaction reverts — recent losers bounce,
recent winners give back — so a dollar-neutral book that buys the biggest
1-week losers and shorts the biggest winners earns a reversal / liquidity-
provision premium (Lehmann, QJE 1990; Lo & MacKinlay, RFS 1990). Distinct from
the momentum sleeves: opposite SIGN and a much shorter HORIZON, so it should be
weakly/negatively correlated with them — genuine edge-type diversification.

Microstructure honesty: the most recent day's return is contaminated by
bid-ask bounce, which manufactures spurious "reversal" you cannot capture after
costs. We therefore SKIP the most recent `skip` day(s) and rank on the formation
window ending at t-skip. This is the conservative test the playbook implies when
it warns short-horizon reversal "needs intraday resolution" on daily bars.

Cross-sectional: `window` passed to compute_signal has columns = symbols, rows =
daily closes. Dollar-neutral: total long = total short = 50% of equity.
"""

from __future__ import annotations

import pandas as pd

from mft.alphas.base import AlphaBase


class ShortTermReversal(AlphaBase):
    """
    Dollar-neutral short-horizon reversal (contrarian).

    Parameters:
        universe:  Ticker symbols in the ranked universe.
        window:    Formation horizon in bars for the reversal signal (default 5 ≈ 1wk).
        skip:      Recent bars excluded to avoid bid-ask-bounce artifacts (default 1).
        frac:      Fraction to go long and short (default 0.20 = bottom/top quintile).
        lookback:  History/eligibility requirement in bars (default 63 ≈ 1 quarter),
                   kept larger than the signal window so the universe gate stays
                   meaningful even though the signal only uses the recent window.
    """

    def __init__(
        self,
        universe: list[str],
        window: int = 5,
        skip: int = 1,
        frac: float = 0.20,
        lookback: int = 63,
    ):
        if not universe:
            raise ValueError("universe must not be empty")
        self.universe = list(universe)
        self.window = window
        self.skip = skip
        self.frac = frac
        self._lookback = lookback

    @property
    def lookback(self) -> int:
        return self._lookback

    def compute_signal(self, window: pd.DataFrame) -> dict[str, float]:
        """
        window: DataFrame, columns = ticker symbols, rows = daily closes.
        Returns: dollar-neutral weights — long recent losers, short recent winners.
        """
        available = [s for s in self.universe if s in window.columns]
        if len(available) < 4:
            return {s: 0.0 for s in self.universe}

        closes = window[available]
        if len(closes) < self.window + self.skip + 1:
            return {s: 0.0 for s in self.universe}

        # Formation return over [t-window-skip, t-skip], excluding the most recent
        # `skip` bar(s) so bid-ask bounce on the last close can't drive the signal.
        recent_price = closes.shift(self.skip).iloc[-1]
        past_price = closes.shift(self.window + self.skip).iloc[-1]
        formation_ret = (recent_price / past_price - 1).dropna()
        if formation_ret.empty:
            return {s: 0.0 for s in self.universe}

        n_each = max(1, int(len(formation_ret) * self.frac))
        # CONTRARIAN: long the LOSERS (smallest formation return),
        #             short the WINNERS (largest). Opposite of momentum.
        long_syms = set(formation_ret.nsmallest(n_each).index)
        short_syms = set(formation_ret.nlargest(n_each).index)

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
            f"ShortTermReversal(n={len(self.universe)}, window={self.window}, "
            f"skip={self.skip}, frac={self.frac})"
        )
