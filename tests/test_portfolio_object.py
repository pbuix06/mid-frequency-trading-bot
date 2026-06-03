"""Tests for the Portfolio object — alloc + netting + risk in one callable."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.portfolio.portfolio import Portfolio
from mft.risk.limits import RiskLimits, RiskManager


def _hist(seed=0) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "trend": rng.normal(0.0003, 0.008, 300),
        "ls":    rng.normal(0.0002, 0.012, 300),
    }, index=idx)


def _pf(**limit_kw) -> Portfolio:
    risk = RiskManager(RiskLimits(**limit_kw), init_equity=100_000.0)
    return Portfolio(_hist(), risk)


def test_construct_nets_and_allocates():
    pf = _pf(max_position_pct=1.0, max_gross_exposure=10.0)
    res = pf.construct({"trend": {"SPY": 1.0}, "ls": {"SPY": -0.5, "AAPL": 0.5}}, 100_000)
    # SPY = alloc[trend]*1.0 + alloc[ls]*-0.5 ; allocations sum to 1
    assert abs(sum(pf.alloc.values()) - 1.0) < 1e-9
    expected_spy = pf.alloc["trend"] * 1.0 + pf.alloc["ls"] * -0.5
    assert abs(res.approved["SPY"] - expected_spy) < 1e-9


def test_unknown_sleeve_ignored():
    pf = _pf(max_position_pct=1.0, max_gross_exposure=10.0)
    res = pf.construct({"trend": {"SPY": 1.0}, "ghost": {"SPY": 9.9}}, 100_000)
    # 'ghost' has no allocation -> contributes nothing
    assert abs(res.approved["SPY"] - pf.alloc["trend"]) < 1e-9


def test_risk_clip_applied_in_construct():
    pf = _pf(max_position_pct=0.10, max_gross_exposure=10.0)
    res = pf.construct({"trend": {"SPY": 1.0}}, 100_000)   # raw SPY ~ alloc<=1
    assert abs(res.approved["SPY"]) <= 0.10 + 1e-9
    if abs(res.raw["SPY"]) > 0.10:
        assert res.violations


def test_kill_switch_blocks_construct():
    pf = _pf(max_position_pct=1.0, max_gross_exposure=10.0, max_drawdown_pct=0.10)
    pf.mark_to_market(100_000)
    pf.mark_to_market(80_000)            # -20% -> halt
    assert pf.halted
    res = pf.construct({"trend": {"SPY": 1.0}}, 80_000)
    assert res.approved == {}
    assert any("HALTED" in v for v in res.violations)


def test_exposure_reported():
    pf = _pf(max_position_pct=1.0, max_gross_exposure=10.0)
    res = pf.construct({"ls": {"A": 0.5, "B": -0.5}}, 100_000)
    assert abs(res.exposure["net"]) < 1e-9          # dollar-neutral sleeve
    assert res.exposure["gross"] > 0
