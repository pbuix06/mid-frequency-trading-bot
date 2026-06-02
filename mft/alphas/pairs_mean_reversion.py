"""
Pairs Mean Reversion (StatArb).

Economic rationale: economically related assets share common risk factors;
deviations in their log-price ratio are stationary and mean-revert because
the fundamental linkage between them is permanent (Bamberger & Tartaglia,
Morgan Stanley 1986; Engle & Granger, Econometrica 1987).

Dollar-neutral by construction: long asset1 / short asset2 or vice versa.
This removes market beta so the P&L is uncorrelated with directional alphas.

Signal: z-score of the OLS-estimated spread log(asset1) − β·log(asset2).
Enter when |z| > threshold; exit when |z| < exit_threshold.

Hard stop: spread z-score > stop_z exits immediately (spread can break;
'spreads break so stops are essential' — playbook §3.2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.alphas.base import AlphaBase


class PairsMeanReversion(AlphaBase):
    """
    Cointegration-based stat-arb on two assets.

    The `window` passed to compute_signal must be a DataFrame whose columns
    include both asset1 and asset2 as close prices.

    Parameters:
        asset1, asset2: Ticker strings for the two legs.
        lookback:       Bars used to estimate hedge ratio and spread stats (252).
        z_entry:        Enter when |z-score| > z_entry (default 1.5).
        z_exit:         Close position when |z-score| < z_exit (default 0.5).
        z_stop:         Hard stop when |z-score| > z_stop (default 3.5).
    """

    def __init__(
        self,
        asset1: str,
        asset2: str,
        lookback: int = 252,
        z_entry: float = 1.5,
        z_exit: float = 0.5,
        z_stop: float = 3.5,
    ):
        self.asset1 = asset1
        self.asset2 = asset2
        self._lookback = lookback
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop

    @property
    def lookback(self) -> int:
        return self._lookback

    def compute_signal(self, window: pd.DataFrame) -> dict[str, float]:
        """
        window: DataFrame with columns [asset1, asset2] = close prices.
        Returns:
            - dollar-neutral weights {asset1: ±0.5, asset2: ∓0.5} to enter/reverse
            - explicit zeros to exit
            - {} to hold the current pair position
        """
        flat = {self.asset1: 0.0, self.asset2: 0.0}

        if self.asset1 not in window.columns or self.asset2 not in window.columns:
            return flat

        p1 = window[self.asset1].replace(0, np.nan).dropna()
        p2 = window[self.asset2].replace(0, np.nan).dropna()
        idx = p1.index.intersection(p2.index)

        if len(idx) < self._lookback + 1:
            return flat

        idx = idx[-(self._lookback + 1):]
        log1 = np.log(p1.loc[idx].values)
        log2 = np.log(p2.loc[idx].values)

        # OLS: log1 = alpha + beta * log2 + epsilon  (all lookback — PIT safe)
        X = np.column_stack([np.ones(len(log2)), log2])
        coeffs, *_ = np.linalg.lstsq(X, log1, rcond=None)
        alpha_coef, beta = coeffs

        spread = log1 - (alpha_coef + beta * log2)
        mu, sigma = spread.mean(), spread.std()

        if sigma < 1e-10:
            return flat

        z = (spread[-1] - mu) / sigma

        # Hard stop — spread has broken, exit
        if abs(z) > self.z_stop:
            return flat

        # Entry / exit / hold. The harness treats {} as "leave current
        # positions unchanged", preserving the entry/exit band without alpha state.
        if z < -self.z_entry:
            # asset1 cheap vs asset2 → buy asset1, sell asset2
            return {self.asset1: 0.5, self.asset2: -0.5}
        elif z > self.z_entry:
            # asset1 expensive vs asset2 → sell asset1, buy asset2
            return {self.asset1: -0.5, self.asset2: 0.5}
        elif abs(z) < self.z_exit:
            # Spread reverted — close position
            return flat

        return {}

    def __repr__(self) -> str:
        return (
            f"PairsMeanReversion({self.asset1!r}/{self.asset2!r}, "
            f"lookback={self._lookback}, z_entry={self.z_entry})"
        )
