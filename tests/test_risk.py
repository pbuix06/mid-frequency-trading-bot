"""
Risk module tests — the kill switches must be proven BEFORE any paper/live run.
The playbook: 'risk limits implemented and unit-tested' is a Stage-5 gate.
"""

from __future__ import annotations

from mft.risk.limits import RiskLimits, RiskManager


def _mgr(**kw) -> RiskManager:
    return RiskManager(RiskLimits(**kw), init_equity=100_000.0)


def test_position_cap_clips():
    m = _mgr(max_position_pct=0.25)
    approved, viol = m.check_pre_trade({"AAPL": 0.40, "MSFT": -0.50}, 100_000)
    assert approved["AAPL"] == 0.25
    assert approved["MSFT"] == -0.25
    assert len(viol) == 2


def test_gross_exposure_scaled():
    m = _mgr(max_position_pct=1.0, max_gross_exposure=1.0)
    approved, viol = m.check_pre_trade({"A": 0.8, "B": -0.8}, 100_000)  # gross 1.6
    gross = sum(abs(w) for w in approved.values())
    assert abs(gross - 1.0) < 1e-9
    assert any("Gross" in v for v in viol)


def test_drawdown_kill_switch_halts():
    m = _mgr(max_drawdown_pct=0.10)
    m.update_post_trade(100_000)             # peak
    alerts = m.update_post_trade(88_000)     # -12% drawdown
    assert m.state.is_halted
    assert any("drawdown" in a.lower() for a in alerts)
    # Once halted, pre-trade returns no orders
    approved, viol = m.check_pre_trade({"A": 0.1}, 88_000)
    assert approved == {}
    assert "HALTED" in viol[0]


def test_daily_loss_kill_switch():
    m = _mgr(daily_loss_limit_pct=0.02, max_drawdown_pct=0.50)
    m.reset_daily(100_000)
    alerts = m.update_post_trade(97_500)     # -2.5% on the day
    assert m.state.is_halted
    assert any("daily loss" in a.lower() for a in alerts)


def test_manual_resume_required():
    m = _mgr(max_drawdown_pct=0.10)
    m.update_post_trade(100_000)
    m.update_post_trade(80_000)              # halt
    assert m.state.is_halted
    m.manual_resume()
    assert not m.state.is_halted
    approved, _ = m.check_pre_trade({"A": 0.1}, 80_000)
    assert approved == {"A": 0.1}


def test_no_false_halt_within_limits():
    m = _mgr(max_drawdown_pct=0.10, daily_loss_limit_pct=0.05)
    m.reset_daily(100_000)
    m.update_post_trade(100_000)
    alerts = m.update_post_trade(98_000)     # -2%, within both limits
    assert not m.state.is_halted
    assert alerts == []
