"""Tests for the transaction-cost model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.alphas.long_short_momentum import LongShortMomentum
from mft.backtest.survivorship_harness import run_survivorship_xs
from mft.execution.costs import (
    DEFAULT_COST_PCT,
    liquidity_tiered_cost,
)


def test_tiers_are_monotone_decreasing_in_liquidity():
    """A more liquid name must never cost more than a less liquid one."""
    mega = liquidity_tiered_cost(2_000_000_000)
    large = liquidity_tiered_cost(300_000_000)
    small = liquidity_tiered_cost(50_000_000)
    micro = liquidity_tiered_cost(5_000_000)
    assert mega < large < small < micro
    assert micro == 0.0080
    assert mega == 0.0010


def test_multiplier_scales_linearly():
    base = liquidity_tiered_cost(50_000_000)
    assert liquidity_tiered_cost(50_000_000, multiplier=3.0) == base * 3.0


def test_default_cost_constant():
    assert DEFAULT_COST_PCT == 0.002


def _panel(n=400, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="B", tz="UTC")
    closes, dvols = {}, {}
    for k, nm in enumerate(["A", "B", "C", "D", "E", "F"]):
        px = 100 * np.exp(np.cumsum(rng.normal(0.0002 + k * 0.0001, 0.015, n)))
        closes[nm] = pd.Series(px, index=idx)
        # ~$30M average daily dollar volume -> small tier (40 bps) > flat (20 bps)
        dvols[nm] = pd.Series(30_000_000.0, index=idx)
    return pd.DataFrame(closes), pd.DataFrame(dvols)


def test_tiered_cost_drags_more_than_flat_on_illiquid_book():
    """
    On a book of mid/small names (40 bps tiered vs 20 bps flat), tiered costs
    must reduce realized return at least as much as the flat default.
    """
    close_df, dvol_df = _panel()
    alpha = LongShortMomentum(universe=list(close_df.columns), lookback=60, skip=5, frac=0.34)
    flat = run_survivorship_xs(close_df=close_df, dvol_df=dvol_df, alpha=alpha,
                               rebalance_freq=21, min_dollar_vol=1.0, tiered_cost=False)
    tiered = run_survivorship_xs(close_df=close_df, dvol_df=dvol_df, alpha=alpha,
                                 rebalance_freq=21, min_dollar_vol=1.0, tiered_cost=True)
    assert tiered.final_equity <= flat.final_equity + 1e-6


def test_cost_multiplier_reduces_equity():
    close_df, dvol_df = _panel()
    alpha = LongShortMomentum(universe=list(close_df.columns), lookback=60, skip=5, frac=0.34)
    one = run_survivorship_xs(close_df=close_df, dvol_df=dvol_df, alpha=alpha,
                              rebalance_freq=21, min_dollar_vol=1.0,
                              tiered_cost=True, cost_multiplier=1.0)
    three = run_survivorship_xs(close_df=close_df, dvol_df=dvol_df, alpha=alpha,
                                rebalance_freq=21, min_dollar_vol=1.0,
                                tiered_cost=True, cost_multiplier=3.0)
    assert three.final_equity < one.final_equity
