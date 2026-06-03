"""Tests for the generic cross-sectional factor sleeve."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.alphas.cross_sectional_factor import CrossSectionalFactor


def _prices(symbols, n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for s in symbols},
        index=idx,
    )


def test_dollar_neutral_and_correct_side():
    syms = ["A", "B", "C", "D", "E"]
    prices = _prices(syms)
    # Static factor: A highest, E lowest
    panel = pd.DataFrame(
        {s: np.full(len(prices), v) for s, v in zip(syms, [5, 4, 3, 2, 1])},
        index=prices.index,
    )
    alpha = CrossSectionalFactor(universe=syms, signal_panel=panel, frac=0.20, high_is_long=True)
    sig = alpha.compute_signal(prices.iloc[-alpha.lookback - 1:])
    assert abs(sum(sig.values())) < 1e-9          # dollar-neutral
    assert sig["A"] > 0 and sig["E"] < 0          # high=long, low=short


def test_low_is_long_inverts_side():
    syms = ["A", "B", "C", "D", "E"]
    prices = _prices(syms)
    panel = pd.DataFrame(
        {s: np.full(len(prices), v) for s, v in zip(syms, [5, 4, 3, 2, 1])},
        index=prices.index,
    )
    alpha = CrossSectionalFactor(universe=syms, signal_panel=panel, frac=0.20, high_is_long=False)
    sig = alpha.compute_signal(prices.iloc[-alpha.lookback - 1:])
    assert sig["A"] < 0 and sig["E"] > 0          # inverted


def test_flat_when_date_absent_from_panel():
    syms = ["A", "B", "C", "D", "E"]
    prices = _prices(syms)
    panel = pd.DataFrame(  # panel covers a DIFFERENT date range
        {s: [1.0, 2.0] for s in syms},
        index=pd.date_range("2000-01-01", periods=2, freq="B", tz="UTC"),
    )
    alpha = CrossSectionalFactor(universe=syms, signal_panel=panel)
    sig = alpha.compute_signal(prices.iloc[-alpha.lookback - 1:])
    assert all(v == 0.0 for v in sig.values())


def test_no_look_ahead_only_reads_current_row():
    """Poisoning panel rows AFTER the current bar must not change the signal."""
    syms = ["A", "B", "C", "D", "E"]
    prices = _prices(syms)
    panel = pd.DataFrame(
        {s: np.linspace(1, 2, len(prices)) + i for i, s in enumerate(syms)},
        index=prices.index,
    )
    alpha = CrossSectionalFactor(universe=syms, signal_panel=panel)
    window = prices.iloc[100 - alpha.lookback if 100 >= alpha.lookback else 0: 101]
    # use a window whose last date is prices.index[100]
    window = prices.iloc[: 101]
    sig_clean = alpha.compute_signal(window)

    poisoned_panel = panel.copy()
    poisoned_panel.iloc[101:] = 1e9
    alpha_p = CrossSectionalFactor(universe=syms, signal_panel=poisoned_panel)
    sig_after = alpha_p.compute_signal(window)
    assert sig_clean == sig_after


def test_nan_factor_names_excluded():
    syms = ["A", "B", "C", "D", "E", "F"]
    prices = _prices(syms)
    panel = pd.DataFrame(
        {s: np.full(len(prices), v) for s, v in zip(syms, [5, 4, 3, 2, 1, np.nan])},
        index=prices.index,
    )
    alpha = CrossSectionalFactor(universe=syms, signal_panel=panel, frac=0.34)
    sig = alpha.compute_signal(prices.iloc[-alpha.lookback - 1:])
    assert sig["F"] == 0.0                          # NaN factor -> never selected
    assert abs(sum(sig.values())) < 1e-9
