"""
Stage 6 paper-harness tests — the full loop with mock sleeves + synthetic prices.
Proves: rebalancing toward the target book, equity tracking, kill-switch halt,
and crash-recovery from persisted state. No real data/alphas needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.execution.adapter import SimulatedAdapter
from mft.execution.paper_trader import CallableSleeve, PaperTrader
from mft.portfolio.portfolio import Portfolio
from mft.risk.limits import RiskLimits, RiskManager


def _portfolio(**limit_kw) -> Portfolio:
    idx = pd.date_range("2020-01-01", periods=100, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    hist = pd.DataFrame({"trend": rng.normal(0, 0.01, 100), "ls": rng.normal(0, 0.01, 100)}, index=idx)
    risk = RiskManager(RiskLimits(**limit_kw), init_equity=100_000.0)
    return Portfolio(hist, risk)


def _trader(state_path=None, **limit_kw):
    pf = _portfolio(**limit_kw)
    sleeves = [
        CallableSleeve("trend", lambda ts: {"SPY": 1.0}),
        CallableSleeve("ls", lambda ts: {"SPY": -0.5, "AAPL": 0.5}),
    ]
    adapter = SimulatedAdapter(init_cash=100_000.0, commission_pct=0.0, slippage_pct=0.0)
    return PaperTrader(sleeves, pf, adapter, state_path=state_path)


def test_cycle_moves_toward_target_book():
    pt = _trader(max_position_pct=1.0, max_gross_exposure=10.0)
    rep = pt.run_cycle(pd.Timestamp("2020-06-01", tz="UTC"), {"SPY": 100.0, "AAPL": 200.0})
    assert not rep.halted
    # Book should hold both names; orders were placed from a flat start.
    assert "SPY" in rep.book and "AAPL" in rep.book
    pos = pt.adapter.get_positions()
    assert pos  # non-empty after first rebalance


def test_equity_roughly_conserved_zero_cost_flat_prices():
    pt = _trader(max_position_pct=1.0, max_gross_exposure=10.0)
    prices = {"SPY": 100.0, "AAPL": 200.0}
    pt.run_cycle(pd.Timestamp("2020-06-01", tz="UTC"), prices)
    pt.run_cycle(pd.Timestamp("2020-06-02", tz="UTC"), prices)   # same prices
    eq = pt._equity(prices)
    assert abs(eq - 100_000.0) < 1.0   # zero cost + unchanged prices -> equity flat


def test_kill_switch_halts_and_stops_trading():
    pt = _trader(max_position_pct=1.0, max_gross_exposure=10.0, max_drawdown_pct=0.05)
    pt.run_cycle(pd.Timestamp("2020-06-01", tz="UTC"), {"SPY": 100.0, "AAPL": 200.0})
    # Crash SPY (held long via trend sleeve) to trip drawdown next cycle
    rep = pt.run_cycle(pd.Timestamp("2020-06-02", tz="UTC"), {"SPY": 50.0, "AAPL": 200.0})
    if not rep.halted:
        rep = pt.run_cycle(pd.Timestamp("2020-06-03", tz="UTC"), {"SPY": 40.0, "AAPL": 200.0})
    assert pt.portfolio.halted
    assert rep.orders == {}   # no trading while halted


def test_crash_recovery_restores_positions(tmp_path):
    path = tmp_path / "paper_state.json"
    pt = _trader(state_path=path, max_position_pct=1.0, max_gross_exposure=10.0)
    pt.run_cycle(pd.Timestamp("2020-06-01", tz="UTC"), {"SPY": 100.0, "AAPL": 200.0})
    saved_pos = pt.adapter.get_positions()
    saved_cash = pt.adapter.get_cash()

    # New trader from the same state file -> adapter restored
    pt2 = _trader(state_path=path, max_position_pct=1.0, max_gross_exposure=10.0)
    assert pt2.adapter.get_positions() == saved_pos
    assert abs(pt2.adapter.get_cash() - saved_cash) < 1e-6


def test_returns_series_after_multiple_cycles():
    pt = _trader(max_position_pct=1.0, max_gross_exposure=10.0)
    for i, px in enumerate([100, 101, 102, 101]):
        pt.run_cycle(pd.Timestamp("2020-06-01", tz="UTC") + pd.Timedelta(days=i),
                     {"SPY": float(px), "AAPL": 200.0})
    assert len(pt.returns) == 3   # 4 equity points -> 3 returns
