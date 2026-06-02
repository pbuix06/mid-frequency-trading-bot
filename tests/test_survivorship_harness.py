"""
Tests for the survivorship-free cross-sectional harness.

Focus: the things that make it *survivorship-free* — point-in-time eligibility
and delisting liquidation — since those are what the fixed-universe harness
gets wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.alphas.long_short_momentum import LongShortMomentum
from mft.backtest.survivorship_harness import (
    eligible_as_of,
    run_survivorship_xs,
)


def _panel(n=400, seed=0):
    """
    6-name panel where 'DEAD' delists at bar 250 (NaN thereafter).
    DEAD is a dominant, low-noise uptrend so a momentum book reliably HOLDS it
    long right up to the delisting — that is what exercises the liquidation path.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="B", tz="UTC")
    closes, dvols = {}, {}
    # Five ordinary noisy names
    for k, nm in enumerate(["A", "B", "C", "D", "E"]):
        px = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.015, n)))
        closes[nm] = pd.Series(px, index=idx)
    # DEAD: strong persistent winner, minimal noise -> stays in the long book
    dead = 100 * np.exp(np.cumsum(np.full(n, 0.004) + rng.normal(0, 0.002, n)))
    s = pd.Series(dead, index=idx)
    s.iloc[250:] = np.nan  # delists at bar 250 while still a top winner
    closes["DEAD"] = s
    for nm, c in closes.items():
        dvols[nm] = c * 5_000_000  # ample liquidity
    return pd.DataFrame(closes), pd.DataFrame(dvols)


def test_eligible_excludes_delisted_after_death():
    close_df, dvol_df = _panel()
    elig_before = eligible_as_of(close_df, dvol_df, 240, lookback=60, min_dollar_vol=1.0)
    elig_after = eligible_as_of(close_df, dvol_df, 300, lookback=60, min_dollar_vol=1.0)
    assert "DEAD" in elig_before, "DEAD should be eligible while still trading"
    assert "DEAD" not in elig_after, "DEAD must drop out once it stops trading"


def test_eligible_requires_enough_history():
    close_df, dvol_df = _panel()
    # At a bar earlier than lookback, nothing qualifies
    assert eligible_as_of(close_df, dvol_df, 30, lookback=60, min_dollar_vol=1.0) == []


def test_eligible_filters_on_liquidity():
    close_df, dvol_df = _panel()
    # Make 'A' illiquid; it should drop out under a high threshold
    dvol_df = dvol_df.copy()
    dvol_df["A"] = 1.0
    elig = eligible_as_of(close_df, dvol_df, 300, lookback=60, min_dollar_vol=1_000.0)
    assert "A" not in elig


def test_delisting_is_liquidated_and_equity_finite():
    close_df, dvol_df = _panel()
    alpha = LongShortMomentum(universe=list(close_df.columns), lookback=60, skip=5, frac=0.34)
    res = run_survivorship_xs(
        alpha, close_df, dvol_df,
        rebalance_freq=21, min_dollar_vol=1.0,
        commission_pct=0.0, slippage_pct=0.0,
    )
    # Equity must stay finite across the delisting (no NaN/inf leakage)
    assert np.isfinite(res.final_equity)
    assert res.equity.notna().all()
    assert res.n_delistings >= 1, "DEAD delisting should be detected and realized"
    assert res.avg_universe_size > 0


def test_no_lookahead_in_eligibility():
    """Eligibility at bar i must not depend on bars after i."""
    close_df, dvol_df = _panel()
    base = eligible_as_of(close_df, dvol_df, 200, lookback=60, min_dollar_vol=1.0)
    poisoned_c = close_df.copy()
    poisoned_c.iloc[201:] = 1e9
    poisoned_d = dvol_df.copy()
    poisoned_d.iloc[201:] = 1e15
    after = eligible_as_of(poisoned_c, poisoned_d, 200, lookback=60, min_dollar_vol=1.0)
    assert base == after
