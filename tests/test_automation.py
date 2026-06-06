"""
Research-automation framework: governance gate (no live), simulated-fill ledger, and the
deviation / false-positive guard. No real orders exist anywhere to test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mft.automation import registry as REG
from mft.automation.monitor import DataMonitor, DeviationMonitor, RiskCostMonitor
from mft.automation.paper_engine import PaperConfig, run_paper
from mft.automation.registry import StrategySpec


# ── governance ────────────────────────────────────────────────────────────────
def test_no_live_approved():
    assert REG.LIVE_TRADING_APPROVED is False
    REG.assert_no_live_approved()
    assert "live" not in REG.VALID_STATUS
    assert all(s.status in REG.VALID_STATUS for s in REG.all_specs().values())


def test_live_status_is_rejected_by_spec():
    with pytest.raises(ValueError):
        StrategySpec(name="x", asset_class="crypto", family="momentum", universe=("A",),
                     params={}, expected={}, status="live", trial_ids="")


# ── paper engine: simulated fills only, full ledger ───────────────────────────
def _panel(n=20, syms=("A", "B", "C", "D"), seed=0):
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({s: 100 + np.cumsum(rng.normal(0, 0.3, n)) for s in syms}, index=idx)


def _spec():
    return StrategySpec(name="t_mom", asset_class="crypto", family="momentum", universe=("A", "B", "C", "D"),
                        params={"lookback_bars": 2, "hold_bars": 2, "top_n": 1},
                        expected={"ic": -0.04, "gross_bps": 1.0, "net_bps_5": -12.0, "verdict": "REJECTED"},
                        status="paper_watch", trial_ids="Tx")


def test_paper_ledger_records_simulated_trades():
    close = _panel()
    reg = {"trend": pd.Series("testregime", index=close.index)}
    res = run_paper(_spec(), close, reg, PaperConfig(cost_bps_per_side=5.0, mode="taker"))
    led = res.ledger
    assert not led.empty
    for col in ("ts", "symbol", "side", "signal", "entry_px", "exit_px", "cost_bps", "regime", "net_bps", "reason"):
        assert col in led.columns
    # every fill is simulated (taker x2) and net = gross + funding - cost
    assert (led["fill"] == "taker x2").all()
    assert led["cost_bps"].round(6).eq(2 * 5.0).all()             # round-trip 2 x 5 bps
    assert np.allclose(led["net_bps"], led["gross_bps"] + led["funding_bps"] - led["cost_bps"])
    assert set(led["side"]) <= {"long", "short"} and (led["regime"] == "testregime").all()


def test_maker_optimistic_cheaper_than_taker():
    close = _panel(seed=3)
    reg = {"trend": pd.Series("r", index=close.index)}
    t = run_paper(_spec(), close, reg, PaperConfig(mode="taker", cost_bps_per_side=5.0)).ledger["cost_bps"].mean()
    m = run_paper(_spec(), close, reg, PaperConfig(mode="maker_optimistic")).ledger["cost_bps"].mean()
    assert m < t   # capturing spread (best case) costs less than crossing it


# ── deviation / false-positive guard ──────────────────────────────────────────
def test_false_positive_flag_when_paper_beats_rejected_backtest():
    spec = _spec()
    warns = DeviationMonitor().compare(spec, {"n_trades": 100, "mean_net_bps": +6.0, "ic": -0.04})
    assert any(w.code == "SUSPECTED_FALSE_POSITIVE" for w in warns)


def test_stays_rejected_when_paper_negative():
    warns = DeviationMonitor().compare(_spec(), {"n_trades": 100, "mean_net_bps": -11.0, "ic": -0.04})
    assert any(w.code == "STAYS_REJECTED" for w in warns)
    assert not any(w.code == "SUSPECTED_FALSE_POSITIVE" for w in warns)


def test_ic_sign_flip_flagged():
    warns = DeviationMonitor().compare(_spec(), {"n_trades": 100, "mean_net_bps": -11.0, "ic": +0.05})
    assert any(w.code == "IC_SIGN_FLIP" for w in warns)


# ── risk / cost ───────────────────────────────────────────────────────────────
def test_cost_floor_enforced():
    rc = RiskCostMonitor()
    assert any(w.code == "COST_TOO_LOW" for w in rc.check_cost_assumption("crypto", 1.0))
    assert rc.check_cost_assumption("crypto", 5.0) == []


def test_data_monitor_flags_dirty():
    idx = pd.date_range("2025-01-01", periods=30, freq="1min", tz="UTC")
    c = pd.Series(100 + np.arange(30) * 0.1, index=idx)
    bars = pd.DataFrame({"open": c, "high": c + 0.1, "low": c - 0.1, "close": c, "volume": 1.0}, index=idx)
    _, warns = DataMonitor().validate(bars)
    assert warns == []                                   # clean
    bars.iloc[5, bars.columns.get_loc("low")] = 999      # low > high
    _, warns2 = DataMonitor().validate(bars)
    assert any(w.code == "DATA_DIRTY" for w in warns2)
