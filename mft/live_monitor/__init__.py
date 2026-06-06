"""
Real-time crypto monitoring + paper-decision system — PAPER / OBSERVATION ONLY.

High-frequency MONITORING (seconds / 1-minute bars), low-frequency paper TRADING (most
minutes produce no trade). This module observes the live crypto tape, computes the same
past-only features the research engine uses, runs ONE pre-declared smoke-test trigger
(24h breakout), gates it through risk limits, and records simulated fills to a ledger.

It does NOT change the project's research verdict (no production-ready alpha found), does
NOT revive any rejected strategy, and does NOT enable live trading.

Hard guarantees (enforced in code + tests/test_live_monitor.py):
  - No real-order path exists anywhere: `paper_decision.LIVE_TRADING_ENABLED` is False and
    `assert_no_live_order_path()` rejects any order-routing method on the paper engine.
  - The registry governance gate (`mft.automation.registry.assert_no_live_approved`) runs in
    the LiveMonitor constructor; `LIVE_TRADING_APPROVED` stays False.
  - Every "trade" is a hypothetical simulated fill (configurable taker fee + slippage),
    logged with its signal, costs, regime, and exit reason; every rejected signal is logged
    with the exact risk reasons.
  - The breakout trigger has ZERO expected edge; a profitable paper run is flagged a
    SUSPECTED FALSE POSITIVE by report.py, never treated as a discovery.

See docs/PAPER_MONITORING_ONLY.md and docs/LIVE_MONITORING_PLAN.md.
"""

from __future__ import annotations

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
from mft.live_monitor.risk_gate import RiskDecision, RiskGate, RiskLimits
from mft.live_monitor.runner import LiveMonitor, LiveMonitorConfig, MinuteResult
from mft.live_monitor.signal_engine import BreakoutSignalEngine, Signal, SignalConfig
from mft.live_monitor.websocket_client import (
    BinanceRestPoller,
    BinanceWebSocketClient,
    ReplayStream,
)

__all__ = [
    "Bar", "BarBuilder", "Tick",
    "FeatureConfig", "FeatureSnapshot", "FeatureStream", "RegimeConfig", "btc_regime",
    "LIVE_TRADING_ENABLED", "PaperDecisionConfig", "PaperDecisionEngine", "assert_no_live_order_path",
    "RiskDecision", "RiskGate", "RiskLimits",
    "LiveMonitor", "LiveMonitorConfig", "MinuteResult",
    "BreakoutSignalEngine", "Signal", "SignalConfig",
    "BinanceRestPoller", "BinanceWebSocketClient", "ReplayStream",
]
