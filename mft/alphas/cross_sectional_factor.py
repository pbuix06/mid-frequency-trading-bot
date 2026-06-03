"""
Cross-Sectional Factor — dollar-neutral long/short on a precomputed signal panel.

A generic ranking sleeve: given a PIT-aligned signal panel (dates × tickers) —
e.g. book-to-market for value, gross-profitability for quality — go long the top
`frac` of the cross-section and short the bottom `frac`, equal-weight, dollar-
neutral. The same object drives value, quality, or any fundamental factor; only
the panel changes.

PIT safety: the alpha reads ONLY the panel row at the current bar
(window.index[-1]). The panel itself must already be point-in-time (fundamentals
forward-filled by SEC `filed` date, market cap from that day's price), so no
future information enters. Verified by test_cross_sectional_factor.

Economic rationale depends on the panel:
  - book-to-market (high=long): value premium; cheap stocks outperform
    (Fama & French 1992) — classically UNCORRELATED with momentum
    (Asness, Moskowitz & Pedersen, "Value and Momentum Everywhere", 2013).
  - gross profitability (high=long): quality premium (Novy-Marx 2013).
"""

from __future__ import annotations

import pandas as pd

from mft.alphas.base import AlphaBase


class CrossSectionalFactor(AlphaBase):
    """
    Dollar-neutral long/short ranking on a precomputed PIT signal panel.

    Parameters:
        universe:     Ticker symbols in the ranked universe.
        signal_panel: DataFrame, index = dates (UTC), columns = tickers, values =
                      the PIT factor signal (already point-in-time aligned).
        frac:         Fraction long and short (default 0.20 = top/bottom quintile).
        high_is_long: True → long high-signal names (e.g. high book-to-market =
                      cheap = long). False → long low-signal names.
        lookback:     History/eligibility requirement in bars (default 252); kept
                      large so the survivorship harness's liquidity gate stays
                      meaningful even though the signal is a single-date lookup.
    """

    def __init__(
        self,
        universe: list[str],
        signal_panel: pd.DataFrame,
        frac: float = 0.20,
        high_is_long: bool = True,
        lookback: int = 252,
    ):
        if not universe:
            raise ValueError("universe must not be empty")
        self.universe = list(universe)
        self._panel = signal_panel
        self.frac = frac
        self.high_is_long = high_is_long
        self._lookback = lookback

    @property
    def lookback(self) -> int:
        return self._lookback

    def compute_signal(self, window: pd.DataFrame) -> dict[str, float]:
        flat = {s: 0.0 for s in self.universe}
        date = window.index[-1]
        if date not in self._panel.index:
            return flat

        row = self._panel.loc[date]
        # Only names that are both tradeable now (in the price window) and have a
        # known (non-NaN) factor value as of this date.
        available = [s for s in window.columns if s in row.index and pd.notna(row[s])]
        if len(available) < 4:
            return flat

        vals = row[available].astype(float)
        n_each = max(1, int(len(vals) * self.frac))
        if self.high_is_long:
            long_syms = set(vals.nlargest(n_each).index)
            short_syms = set(vals.nsmallest(n_each).index)
        else:
            long_syms = set(vals.nsmallest(n_each).index)
            short_syms = set(vals.nlargest(n_each).index)

        long_w, short_w = 0.5 / n_each, -0.5 / n_each
        sig = dict(flat)
        for s in long_syms:
            sig[s] = long_w
        for s in short_syms:
            sig[s] = short_w
        return sig

    def __repr__(self) -> str:
        side = "high=long" if self.high_is_long else "low=long"
        return f"CrossSectionalFactor(n={len(self.universe)}, frac={self.frac}, {side})"
