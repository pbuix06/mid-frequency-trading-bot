"""
LiveMonitor — the orchestrator that wires the live-monitor pieces into one loop.

Per closed minute frame `(now, {symbol: Bar})`:
  1. update the rolling feature window for every symbol in the frame
  2. recompute the (coarse, causal) BTC regime
  3. manage open paper positions (time-stop / regime-flip exits)
  4. for each symbol: snapshot -> breakout signal -> risk gate -> paper decision
  5. log data outages (subscribed symbols missing from the frame)

High-frequency MONITORING, low-frequency TRADING: most minutes produce no trade. Nothing
here can place a real order — the only "execution" is the simulated paper book. Live
trading is not approved; `assert_no_live_order_path()` runs in the constructor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mft.automation.registry import assert_no_live_approved
from mft.live_monitor.feature_stream import (
    FeatureConfig,
    FeatureSnapshot,
    FeatureStream,
    RegimeConfig,
    btc_regime,
)
from mft.live_monitor.paper_decision import (
    PaperDecisionConfig,
    PaperDecisionEngine,
    assert_no_live_order_path,
)
from mft.live_monitor.risk_gate import RiskGate, RiskLimits
from mft.live_monitor.signal_engine import BreakoutSignalEngine, SignalConfig
from mft.live_monitor.websocket_client import MarketDataStream


@dataclass
class LiveMonitorConfig:
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    btc_symbol: str = "BTCUSDT"
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskLimits = field(default_factory=RiskLimits)
    paper: PaperDecisionConfig = field(default_factory=PaperDecisionConfig)


@dataclass
class MinuteResult:
    now: pd.Timestamp
    regime: str
    n_symbols: int
    n_trades_open: int
    signals_fired: list[str]           # symbols that produced an actionable signal
    outages: list[str]                 # subscribed symbols missing this minute


class LiveMonitor:
    def __init__(self, cfg: LiveMonitorConfig | None = None,
                 funding: dict[str, float] | None = None):
        self.cfg = cfg or LiveMonitorConfig()
        assert_no_live_approved()                 # registry-level governance
        self.features = FeatureStream(self.cfg.feature)
        self.signals = BreakoutSignalEngine(self.cfg.signal)
        self.gate = RiskGate(self.cfg.risk)
        self.paper = PaperDecisionEngine(self.cfg.paper)
        assert_no_live_order_path(self.paper)     # no order path on the paper book
        self.funding = funding or {}              # optional static funding map (perp)
        self.outage_log: list[dict] = []
        self.minute_log: list[MinuteResult] = []
        self.last_regime: str = "unknown"

    # ── one closed-minute step ──
    def on_minute(self, now: pd.Timestamp, frame: dict[str, object]) -> MinuteResult:
        now = pd.Timestamp(now)
        for sym, bar in frame.items():
            self.features.update(sym, bar)

        btc_snap = self.features.snapshot(self.cfg.btc_symbol)
        regime = btc_regime(btc_snap, self.cfg.regime)
        self.last_regime = regime

        # snapshot every tracked symbol once (reused by exits + entries)
        snaps: dict[str, FeatureSnapshot] = {}
        for sym in self.cfg.symbols:
            snap = self.features.snapshot(sym, funding_rate=self.funding.get(sym))
            if snap is not None:
                snaps[sym] = snap

        self.paper.manage(now, snaps, regime)

        fired: list[str] = []
        for sym in frame:                          # evaluate only symbols that printed this minute
            snap = snaps.get(sym)
            if snap is None:
                continue
            sig = self.signals.evaluate(snap, regime)
            pf = self.paper.portfolio_view()
            decision = self.gate.check(sig, snap, pf, now, self.cfg.paper.notional_per_trade)
            self.paper.on_signal(sig, snap, decision, now, regime)
            if sig.is_trade:
                fired.append(sym)

        outages = [s for s in self.cfg.symbols if s not in frame]
        if outages:
            self.outage_log.append({"ts": now, "missing": outages})

        res = MinuteResult(now=now, regime=regime, n_symbols=len(frame),
                           n_trades_open=len(self.paper.positions),
                           signals_fired=fired, outages=outages)
        self.minute_log.append(res)
        return res

    # ── drive from any MarketDataStream ──
    def run(self, stream: MarketDataStream, on_step=None) -> LiveMonitor:
        for now, frame in stream.minute_frames():
            res = self.on_minute(now, frame)
            if on_step is not None:
                on_step(res)
        return self
