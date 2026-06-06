"""
Paper decision engine — SIMULATED FILLS ONLY. There is no broker/exchange adapter here.

Translates gated signals into hypothetical positions, marks them with configurable taker
fee + slippage, times them out, and records EVERYTHING: every entry, every exit, and every
rejected signal with its reason. There is intentionally NO method that submits, sends,
places, or routes a real order — `assert_no_live_order_path()` enforces that, and the test
suite asserts it. `LIVE_TRADING_ENABLED` is False and nothing flips it.

This mirrors mft.automation.paper_engine's philosophy (simulated ledger, false-positive
guard) but for the event-driven live-monitor path instead of the vectorised research path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mft.live_monitor.feature_stream import FeatureSnapshot
from mft.live_monitor.risk_gate import PortfolioView, RiskDecision
from mft.live_monitor.signal_engine import Signal

LIVE_TRADING_ENABLED = False  # hard governance: never True in this project

# method-name fragments that would imply a real order path; must never appear on the engine
_FORBIDDEN_ORDER_METHODS = (
    "submit_order", "send_order", "place_order", "create_order", "route_order",
    "execute_order", "broker", "exchange_client", "live_order", "cancel_order",
)


@dataclass
class PaperDecisionConfig:
    taker_fee_bps: float = 5.0
    slippage_bps: float = 2.0
    notional_per_trade: float = 1_000.0
    hold_minutes: int = 240            # smoke-test time-stop (4h)
    exit_on_regime_flip: bool = True   # close a long if BTC turns bearish (and vice-versa)


@dataclass
class PaperPosition:
    symbol: str
    side: str                          # 'long' | 'short'
    entry_ts: pd.Timestamp
    entry_price: float                 # slippage-adjusted fill
    notional: float
    qty: float
    entry_fee: float
    regime_at_entry: str
    entry_reason: str


@dataclass
class PaperTrade:
    symbol: str
    side: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    notional: float
    qty: float
    gross_pnl: float
    cost: float
    net_pnl: float
    hold_minutes: float
    regime_at_entry: str
    exit_reason: str


@dataclass
class PaperBookState:
    positions: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    signals: list = field(default_factory=list)


class PaperDecisionEngine:
    """Event-driven paper book. Simulated fills only; no real-order path exists."""

    def __init__(self, cfg: PaperDecisionConfig | None = None):
        if LIVE_TRADING_ENABLED:
            raise RuntimeError("LIVE_TRADING_ENABLED must be False — paper only.")
        self.cfg = cfg or PaperDecisionConfig()
        self.positions: dict[str, PaperPosition] = {}
        self.trades: list[PaperTrade] = []
        self.rejected: list[dict] = []
        self.signals: list[dict] = []
        self._day: object | None = None
        self._realized_today: float = 0.0
        self.realized_total: float = 0.0

    # ── portfolio view for the risk gate ──
    def portfolio_view(self) -> PortfolioView:
        return PortfolioView(
            open_symbols=set(self.positions),
            n_positions=len(self.positions),
            symbol_notional={s: p.notional for s, p in self.positions.items()},
            realized_pnl_today=self._realized_today,
        )

    # ── fills (SIMULATED — slippage moves the fill against us) ──
    def _fill_price(self, ref_price: float, side: str, entering: bool) -> float:
        buy = (side == "long") == entering        # long-entry & short-exit buy; the others sell
        slip = self.cfg.slippage_bps * 1e-4
        return ref_price * (1.0 + slip) if buy else ref_price * (1.0 - slip)

    def _roll_day(self, now: pd.Timestamp) -> None:
        d = pd.Timestamp(now).date()
        if self._day != d:
            self._day, self._realized_today = d, 0.0

    # ── decision step (called per minute, per symbol) ──
    def on_signal(self, signal: Signal, snap: FeatureSnapshot, gate: RiskDecision,
                  now: pd.Timestamp, regime: str) -> None:
        self._roll_day(now)
        self.signals.append({"ts": now, "symbol": signal.symbol, "side": signal.side,
                             "is_trade": signal.is_trade, "reason": signal.reason,
                             "btc_regime": regime})
        if not signal.is_trade:
            return
        if not gate.allowed:
            self.rejected.append({"ts": now, "symbol": signal.symbol, "side": signal.side,
                                  "signal_reason": signal.reason, "reasons": list(gate.reasons),
                                  "btc_regime": regime})
            return
        self._open(signal, snap, now, regime)

    def _open(self, signal: Signal, snap: FeatureSnapshot, now: pd.Timestamp, regime: str) -> None:
        price = self._fill_price(snap.close, signal.side, entering=True)
        notional = self.cfg.notional_per_trade
        qty = notional / price
        fee = notional * self.cfg.taker_fee_bps * 1e-4
        self.positions[signal.symbol] = PaperPosition(
            symbol=signal.symbol, side=signal.side, entry_ts=pd.Timestamp(now),
            entry_price=price, notional=notional, qty=qty, entry_fee=fee,
            regime_at_entry=regime, entry_reason=signal.reason,
        )

    def _close(self, pos: PaperPosition, ref_price: float, now: pd.Timestamp, reason: str) -> None:
        exit_price = self._fill_price(ref_price, pos.side, entering=False)
        exit_notional = pos.qty * exit_price
        exit_fee = exit_notional * self.cfg.taker_fee_bps * 1e-4
        if pos.side == "long":
            gross = pos.qty * (exit_price - pos.entry_price)
        else:
            gross = pos.qty * (pos.entry_price - exit_price)
        cost = pos.entry_fee + exit_fee
        net = gross - cost
        hold = (pd.Timestamp(now) - pos.entry_ts).total_seconds() / 60.0
        self.trades.append(PaperTrade(
            symbol=pos.symbol, side=pos.side, entry_ts=pos.entry_ts, exit_ts=pd.Timestamp(now),
            entry_price=pos.entry_price, exit_price=exit_price, notional=pos.notional, qty=pos.qty,
            gross_pnl=gross, cost=cost, net_pnl=net, hold_minutes=hold,
            regime_at_entry=pos.regime_at_entry, exit_reason=reason,
        ))
        self._realized_today += net
        self.realized_total += net
        del self.positions[pos.symbol]

    # ── manage open positions each minute (time-stop / regime-flip exits) ──
    def manage(self, now: pd.Timestamp, snapshots: dict[str, FeatureSnapshot], regime: str) -> None:
        self._roll_day(now)
        for sym in list(self.positions):
            pos = self.positions[sym]
            snap = snapshots.get(sym)
            ref = snap.close if snap is not None else pos.entry_price
            held = (pd.Timestamp(now) - pos.entry_ts).total_seconds() / 60.0
            if held >= self.cfg.hold_minutes:
                self._close(pos, ref, now, f"time-stop {self.cfg.hold_minutes}m")
            elif self.cfg.exit_on_regime_flip and (
                (pos.side == "long" and regime == "bearish")
                or (pos.side == "short" and regime == "bullish")
            ):
                self._close(pos, ref, now, f"regime flip to {regime}")

    def unrealized_pnl(self, snapshots: dict[str, FeatureSnapshot]) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            snap = snapshots.get(sym)
            if snap is None:
                continue
            if pos.side == "long":
                total += pos.qty * (snap.close - pos.entry_price)
            else:
                total += pos.qty * (pos.entry_price - snap.close)
        return total


def assert_no_live_order_path(engine: object | None = None) -> None:
    """Hard governance check: no live flag, no order-routing method on the paper engine."""
    if LIVE_TRADING_ENABLED:
        raise RuntimeError("LIVE_TRADING_ENABLED must be False — no live deployment is approved.")
    target = engine if engine is not None else PaperDecisionEngine
    present = {m for m in _FORBIDDEN_ORDER_METHODS if hasattr(target, m)}
    if present:
        raise RuntimeError(f"forbidden live-order method(s) present: {sorted(present)}")
