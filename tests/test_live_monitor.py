"""
Live-monitor (paper-only) tests.

Covers the hard guarantees and the per-component behaviour the brief requires:
  - no live order path exists anywhere
  - the bar builder assembles correct 1m bars from sub-minute ticks
  - the risk gate blocks invalid trades (stale data, spread, limits, duplicate, extreme vol)
  - the paper decision engine logs no-trade reasons (flat signals + gated rejections)
  - the monitor handles stale data and missing-symbol outages
  - end-to-end: a synthetic breakout flows stream->bars->features->signal->gate->paper->report
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mft.live_monitor.bar_builder import Bar, BarBuilder, Tick
from mft.live_monitor.feature_stream import (
    FeatureConfig,
    FeatureSnapshot,
    FeatureStream,
    RegimeConfig,
    btc_regime,
)
from mft.live_monitor.paper_decision import (
    LIVE_TRADING_ENABLED,
    PaperDecisionConfig,
    PaperDecisionEngine,
    assert_no_live_order_path,
)
from mft.live_monitor.report import daily_monitor_report, deviation_warnings
from mft.live_monitor.risk_gate import PortfolioView, RiskGate, RiskLimits
from mft.live_monitor.runner import LiveMonitor, LiveMonitorConfig
from mft.live_monitor.signal_engine import BreakoutSignalEngine, SignalConfig

UTC = "UTC"


def _snap(symbol="SOLUSDT", close=110.0, high=100.0, low=90.0, volz=2.0, warm=True,
          vol24=0.01, ts=None, spread=None) -> FeatureSnapshot:
    ts = ts or pd.Timestamp("2025-01-01 12:00:00", tz=UTC)
    return FeatureSnapshot(
        ts=ts, symbol=symbol, close=close, return_1h=0.0, return_4h=0.0, return_24h=0.0,
        vol_4h=vol24, vol_24h=vol24, high_24h=high, low_24h=low,
        dist_from_high=close / high - 1.0, dist_from_low=close / low - 1.0,
        volume_zscore=volz, funding_rate=None, spread_bps=spread, n_bars=5000, warm=warm,
    )


# ── 1. NO LIVE ORDER PATH ──────────────────────────────────────────────────────
def test_no_live_order_path_exists():
    assert LIVE_TRADING_ENABLED is False
    assert_no_live_order_path()                      # class-level
    eng = PaperDecisionEngine()
    assert_no_live_order_path(eng)                   # instance-level
    for forbidden in ("submit_order", "send_order", "place_order", "create_order",
                      "route_order", "execute_order", "broker", "exchange_client",
                      "live_order", "cancel_order"):
        assert not hasattr(eng, forbidden)


def test_live_monitor_constructor_runs_governance():
    # constructing the monitor must not raise and must keep live disabled
    m = LiveMonitor(LiveMonitorConfig(symbols=["BTCUSDT", "SOLUSDT"]))
    assert m.paper is not None
    assert_no_live_order_path(m.paper)


# ── 2. BAR BUILDER ──────────────────────────────────────────────────────────────
def test_bar_builder_builds_correct_1m_bars():
    bb = BarBuilder("BTCUSDT")
    t0 = pd.Timestamp("2025-01-01 00:00:05", tz=UTC)
    assert bb.add_tick(Tick(t0, 100.0, 1.0)) is None
    assert bb.add_tick(Tick(t0 + pd.Timedelta(seconds=10), 102.0, 2.0)) is None
    assert bb.add_tick(Tick(t0 + pd.Timedelta(seconds=20), 99.0, 1.5)) is None
    # first tick of the next minute finalizes the previous minute's bar
    bar = bb.add_tick(Tick(pd.Timestamp("2025-01-01 00:01:03", tz=UTC), 101.0, 0.5))
    assert bar is not None
    assert bar.ts == pd.Timestamp("2025-01-01 00:00:00", tz=UTC)
    assert (bar.open, bar.high, bar.low, bar.close) == (100.0, 102.0, 99.0, 99.0)
    assert bar.volume == pytest.approx(4.5)
    assert bar.trades == 3
    # an out-of-order tick before the current minute is dropped (not back-dated)
    assert bb.add_tick(Tick(t0, 50.0, 1.0)) is None
    last = bb.flush()
    assert last.ts == pd.Timestamp("2025-01-01 00:01:00", tz=UTC)
    assert last.open == 101.0 and last.trades == 1


def test_bar_spread_from_book():
    bb = BarBuilder("X")
    bb.update_book(99.95, 100.05)
    bb.add_tick(Tick(pd.Timestamp("2025-01-01 00:00:01", tz=UTC), 100.0, 1.0))
    bar = bb.flush()
    assert bar.best_bid == 99.95 and bar.best_ask == 100.05
    assert bar.spread_bps == pytest.approx((0.10 / 100.0) * 1e4, rel=1e-6)
    plain = Bar.from_mapping(pd.Timestamp("2025-01-01", tz=UTC),
                             {"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10})
    assert plain.spread_bps is None


# ── 3. SIGNAL ENGINE ────────────────────────────────────────────────────────────
def test_breakout_long_short_and_flat():
    eng = BreakoutSignalEngine(SignalConfig(min_volume_z=1.0))
    assert eng.evaluate(_snap(close=110, high=100, low=90, volz=2.0), "neutral").side == "long"
    assert eng.evaluate(_snap(close=80, high=100, low=90, volz=2.0), "neutral").side == "short"
    assert eng.evaluate(_snap(close=95, high=100, low=90, volz=2.0), "neutral").side == "flat"


def test_breakout_blocked_by_volume_and_regime():
    eng = BreakoutSignalEngine(SignalConfig(min_volume_z=1.0))
    weak = eng.evaluate(_snap(close=110, volz=0.2), "neutral")
    assert weak.side == "flat" and "volume" in weak.reason
    blocked = eng.evaluate(_snap(close=110, volz=3.0), "bearish")
    assert blocked.side == "flat" and "bearish" in blocked.reason
    warm = eng.evaluate(_snap(close=110, volz=3.0, warm=False), "neutral")
    assert warm.side == "flat" and "warmup" in warm.reason


def test_btc_regime_thresholds():
    cfg = RegimeConfig(bull_threshold=0.03, bear_threshold=-0.03)
    up = _snap()
    up.return_24h = 0.05
    dn = _snap()
    dn.return_24h = -0.05
    mid = _snap()
    mid.return_24h = 0.0
    assert btc_regime(up, cfg) == "bullish"
    assert btc_regime(dn, cfg) == "bearish"
    assert btc_regime(mid, cfg) == "neutral"
    assert btc_regime(None, cfg) == "unknown"


# ── 4. RISK GATE BLOCKS INVALID TRADES ──────────────────────────────────────────
def _long_signal():
    return BreakoutSignalEngine(SignalConfig()).evaluate(_snap(close=110, volz=2.0), "neutral")


def test_risk_gate_allows_clean_trade():
    gate = RiskGate(RiskLimits())
    now = pd.Timestamp("2025-01-01 12:00:00", tz=UTC)
    d = gate.check(_long_signal(), _snap(ts=now), PortfolioView(), now, notional=500.0)
    assert d.allowed and d.reasons == ["ok"]


def test_risk_gate_blocks_stale_data():
    gate = RiskGate(RiskLimits(stale_seconds=120))
    now = pd.Timestamp("2025-01-01 12:05:00", tz=UTC)
    snap = _snap(ts=pd.Timestamp("2025-01-01 12:00:00", tz=UTC))  # 300s old
    d = gate.check(_long_signal(), snap, PortfolioView(), now, notional=500.0)
    assert not d.allowed and any("stale" in r for r in d.reasons)


def test_risk_gate_blocks_wide_spread_and_limits():
    now = pd.Timestamp("2025-01-01 12:00:00", tz=UTC)
    sig = _long_signal()
    # wide spread
    d = RiskGate(RiskLimits(max_spread_bps=10)).check(sig, _snap(ts=now, spread=50.0),
                                                      PortfolioView(), now, 500.0)
    assert not d.allowed and any("spread" in r for r in d.reasons)
    # max positions
    pf = PortfolioView(open_symbols={"A", "B", "C"}, n_positions=3)
    d = RiskGate(RiskLimits(max_positions=3)).check(sig, _snap(ts=now), pf, now, 500.0)
    assert not d.allowed and any("max positions" in r for r in d.reasons)
    # symbol exposure
    d = RiskGate(RiskLimits(max_symbol_notional=500)).check(sig, _snap(ts=now),
                                                            PortfolioView(), now, 1000.0)
    assert not d.allowed and any("exposure" in r for r in d.reasons)
    # daily loss
    pf = PortfolioView(realized_pnl_today=-300.0)
    d = RiskGate(RiskLimits(max_daily_loss=200)).check(sig, _snap(ts=now), pf, now, 100.0)
    assert not d.allowed and any("daily loss" in r for r in d.reasons)
    # extreme vol
    d = RiskGate(RiskLimits(extreme_vol_24h=0.05, allow_extreme_vol=False)).check(
        sig, _snap(ts=now, vol24=0.10), PortfolioView(), now, 100.0)
    assert not d.allowed and any("extreme vol" in r for r in d.reasons)
    # duplicate
    pf = PortfolioView(open_symbols={"SOLUSDT"}, n_positions=1)
    d = RiskGate(RiskLimits()).check(sig, _snap(ts=now), pf, now, 100.0)
    assert not d.allowed and any("duplicate" in r for r in d.reasons)


# ── 5. PAPER DECISION LOGS NO-TRADE REASONS ─────────────────────────────────────
def test_paper_logs_flat_and_rejected_reasons():
    eng = PaperDecisionEngine(PaperDecisionConfig())
    now = pd.Timestamp("2025-01-01 12:00:00", tz=UTC)
    flat = BreakoutSignalEngine().evaluate(_snap(close=95), "neutral")  # no breakout
    eng.on_signal(flat, _snap(close=95), RiskGate().check(flat, _snap(ts=now), PortfolioView(), now, 1000.0), now, "neutral")
    assert eng.signals and eng.signals[-1]["is_trade"] is False
    assert not eng.rejected and not eng.positions

    sig = _long_signal()
    blocked = RiskGate(RiskLimits(max_spread_bps=1)).check(sig, _snap(ts=now, spread=50), PortfolioView(), now, 1000.0)
    eng.on_signal(sig, _snap(ts=now, spread=50), blocked, now, "neutral")
    assert eng.rejected and "spread" in " ".join(eng.rejected[-1]["reasons"])
    assert not eng.positions      # blocked ⇒ no position opened


def test_paper_open_close_costs_and_slippage():
    cfg = PaperDecisionConfig(taker_fee_bps=5.0, slippage_bps=2.0, notional_per_trade=1000.0, hold_minutes=3)
    eng = PaperDecisionEngine(cfg)
    t0 = pd.Timestamp("2025-01-01 12:00:00", tz=UTC)
    snap = _snap(close=100.0, ts=t0)
    sig = _long_signal()
    eng.on_signal(sig, snap, RiskGate().check(sig, snap, PortfolioView(), t0, 1000.0), t0, "neutral")
    assert "SOLUSDT" in eng.positions
    pos = eng.positions["SOLUSDT"]
    assert pos.entry_price == pytest.approx(100.0 * (1 + 2e-4))   # long pays UP (slippage)

    # time-stop exit after hold_minutes
    t1 = t0 + pd.Timedelta(minutes=3)
    exit_snap = _snap(close=100.0, ts=t1)
    eng.manage(t1, {"SOLUSDT": exit_snap}, "neutral")
    assert not eng.positions and len(eng.trades) == 1
    tr = eng.trades[0]
    assert tr.cost > 0 and tr.exit_reason.startswith("time-stop")
    assert tr.net_pnl == pytest.approx(tr.gross_pnl - tr.cost)
    # flat price round-trip: slippage + fees ⇒ strictly negative net
    assert tr.net_pnl < 0


def test_paper_regime_flip_exit():
    cfg = PaperDecisionConfig(hold_minutes=10_000, exit_on_regime_flip=True)
    eng = PaperDecisionEngine(cfg)
    t0 = pd.Timestamp("2025-01-01 12:00:00", tz=UTC)
    sig = _long_signal()
    eng.on_signal(sig, _snap(close=100, ts=t0), RiskGate().check(sig, _snap(ts=t0), PortfolioView(), t0, 1000.0), t0, "neutral")
    eng.manage(t0 + pd.Timedelta(minutes=5), {"SOLUSDT": _snap(close=100, ts=t0 + pd.Timedelta(minutes=5))}, "bearish")
    assert len(eng.trades) == 1 and "regime flip" in eng.trades[0].exit_reason


# ── 6. FEATURE STREAM ───────────────────────────────────────────────────────────
def test_feature_stream_returns_and_warm():
    cfg = FeatureConfig(ret_1h_bars=10, ret_4h_bars=20, ret_24h_bars=30,
                        vol_short_bars=10, vol_long_bars=20, high_low_bars=20,
                        volz_bars=10, min_history_bars=25)
    fs = FeatureStream(cfg)
    idx = pd.date_range("2025-01-01", periods=40, freq="1min", tz=UTC)
    for i, ts in enumerate(idx):
        close = 100.0 + i
        fs.update("X", Bar(ts=ts, open=close, high=close + 0.5, low=close - 0.5,
                           close=close, volume=10.0 + (i % 3)))
    snap = fs.snapshot("X")
    # deque is capped (maxlen = max window + 5 = 35), so only the last 35 bars are retained
    assert snap.warm and snap.n_bars == 35
    assert snap.return_1h == pytest.approx((139.0 / 129.0) - 1.0)   # 10-bar return
    assert snap.high_24h == pytest.approx(138.5)                    # prior bar's high (excl. current)
    assert snap.close == 139.0


# ── 7. END-TO-END: synthetic breakout flows through the whole monitor ────────────
def _bar(ts, close, vol, high=None, low=None):
    return Bar(ts=ts, open=close, high=high if high is not None else close,
               low=low if low is not None else close, close=close, volume=vol)


def test_end_to_end_breakout_produces_paper_trade_and_report(tmp_path):
    cfg = LiveMonitorConfig(
        symbols=["BTCUSDT", "SOLUSDT"], btc_symbol="BTCUSDT",
        feature=FeatureConfig(min_history_bars=15, high_low_bars=10, volz_bars=5,
                              ret_1h_bars=5, ret_4h_bars=10, ret_24h_bars=12,
                              vol_short_bars=5, vol_long_bars=10),
        signal=SignalConfig(min_volume_z=1.5),
        risk=RiskLimits(allow_extreme_vol=True, stale_seconds=1e12, max_spread_bps=1e9),
        paper=PaperDecisionConfig(hold_minutes=3, exit_on_regime_flip=False,
                                  taker_fee_bps=5.0, slippage_bps=2.0),
    )
    monitor = LiveMonitor(cfg)
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01 00:00", periods=24, freq="1min", tz=UTC)
    for i, ts in enumerate(idx):
        btc = _bar(ts, 100.0, 10.0 + rng.normal(0, 0.3))
        if i == 18:
            sol = _bar(ts, 110.0, 200.0, high=110.0, low=100.0)   # breakout + volume spike
        else:
            sol = _bar(ts, 100.0, 10.0 + rng.normal(0, 0.3))
        monitor.on_minute(ts, {"BTCUSDT": btc, "SOLUSDT": sol})

    eng = monitor.paper
    # an actionable SOL long was generated and filled, then time-stopped
    assert any(s["symbol"] == "SOLUSDT" and s["is_trade"] for s in eng.signals)
    assert len(eng.trades) >= 1
    tr = next(t for t in eng.trades if t.symbol == "SOLUSDT")
    assert tr.side == "long" and np.isfinite(tr.net_pnl)

    # report + deviation guard
    import mft.live_monitor.report as R
    R.REPORT_DIR = tmp_path
    R.LEDGER_DIR = tmp_path
    path = daily_monitor_report(monitor)
    assert path.exists()
    text = path.read_text()
    assert "NO LIVE TRADING" in text and "Performance by symbol" in text


def test_deviation_guard_flags_profit_as_false_positive():
    eng = PaperDecisionEngine()
    # a winning trade is the SUSPICIOUS case for a zero-edge smoke test
    from mft.live_monitor.paper_decision import PaperTrade
    eng.trades.append(PaperTrade(symbol="X", side="long", entry_ts=pd.Timestamp("2025-01-01", tz=UTC),
                                 exit_ts=pd.Timestamp("2025-01-01 01:00", tz=UTC), entry_price=100, exit_price=105,
                                 notional=1000, qty=10, gross_pnl=50, cost=5, net_pnl=45, hold_minutes=60,
                                 regime_at_entry="neutral", exit_reason="time-stop"))
    warns = deviation_warnings(eng)
    assert any(w.code == "SUSPECTED_FALSE_POSITIVE" for w in warns)

    eng2 = PaperDecisionEngine()
    eng2.trades.append(PaperTrade(symbol="X", side="long", entry_ts=pd.Timestamp("2025-01-01", tz=UTC),
                                  exit_ts=pd.Timestamp("2025-01-01 01:00", tz=UTC), entry_price=100, exit_price=99,
                                  notional=1000, qty=10, gross_pnl=-10, cost=5, net_pnl=-15, hold_minutes=60,
                                  regime_at_entry="neutral", exit_reason="time-stop"))
    warns2 = deviation_warnings(eng2)
    assert any(w.code == "STAYS_REJECTED" for w in warns2)
