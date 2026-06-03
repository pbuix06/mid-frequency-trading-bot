"""
Paper-trading harness (Stage 6) — run the book through the LIVE code path.

The loop, once per cycle (daily, after close):
  1. ask each sleeve for its current target weights,
  2. drive the Portfolio (alloc -> net -> RiskManager) to get the approved book,
  3. translate target weights -> share deltas vs current broker positions, and
     submit them through the ExecutionAdapter (SimulatedAdapter now; the IB
     adapter later — same interface, that's the point),
  4. mark to market and run the post-trade kill-switch check,
  5. persist state so a crash/restart resumes cleanly.

This proves EXECUTION, data flow, timing, risk, and state under realistic
conditions with zero capital. It is NOT edge validation — the book has not
cleared Gate 4; this is plumbing validation only.

A "sleeve" here is anything with `.name` and `.compute_targets(as_of) ->
{instrument: weight}`. `CallableSleeve` wraps a plain function; real-alpha
sleeve wrappers (building each alpha's data window) are a thin follow-on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd

from mft.execution.adapter import ExecutionAdapter, Order, SimulatedAdapter
from mft.execution.state import StateStore, TradingState
from mft.portfolio.portfolio import Portfolio


class Sleeve(Protocol):
    name: str
    def compute_targets(self, as_of: pd.Timestamp) -> dict[str, float]: ...


@dataclass
class CallableSleeve:
    """Wrap a plain `as_of -> {instrument: weight}` function as a sleeve."""
    name: str
    fn: Callable[[pd.Timestamp], dict[str, float]]

    def compute_targets(self, as_of: pd.Timestamp) -> dict[str, float]:
        return self.fn(as_of)


@dataclass
class CycleReport:
    ts: pd.Timestamp
    equity: float
    book: dict[str, float] = field(default_factory=dict)       # approved target weights
    orders: dict[str, float] = field(default_factory=dict)     # symbol -> share delta
    violations: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    halted: bool = False


class PaperTrader:
    """
    Drives the Portfolio through the ExecutionAdapter on a schedule.

    Args:
        sleeves:    list of objects with .name + .compute_targets(as_of).
        portfolio:  a configured Portfolio (alloc + risk).
        adapter:    ExecutionAdapter (SimulatedAdapter for paper-sim). The broker
                    is the source of truth for cash/positions (one-code-path).
        state_path: if set, TradingState is persisted here after every cycle and
                    restored on construction (crash recovery).
        min_trade_value: skip orders smaller than this notional (avoid churn).
    """

    def __init__(
        self,
        sleeves: list[Sleeve],
        portfolio: Portfolio,
        adapter: ExecutionAdapter,
        state_path: Path | None = None,
        min_trade_value: float = 1.0,
    ):
        self.sleeves = sleeves
        self.portfolio = portfolio
        self.adapter = adapter
        self.state_path = state_path
        self.min_trade_value = min_trade_value
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        self._store = StateStore()

        if state_path is not None and self._store.exists(state_path):
            st = self._store.load(state_path)
            if isinstance(adapter, SimulatedAdapter):
                adapter.set_state(st.cash, st.positions)

    # ── one scheduled cycle ───────────────────────────────────────────────────

    def run_cycle(self, as_of: pd.Timestamp, prices: dict[str, float]) -> CycleReport:
        """
        Run one rebalance cycle at `as_of` given current `prices` (symbol->price).
        Orders fill at those prices (with the adapter's slippage/commission).
        """
        positions = self.adapter.get_positions()
        equity = self._equity(prices)

        # Post-trade risk check first: if a prior cycle's loss tripped the switch,
        # or this mark-to-market does, we halt and place no orders.
        alerts = self.portfolio.mark_to_market(equity, as_of)
        if self.portfolio.halted:
            self.equity_curve.append((as_of, equity))
            self._persist(as_of)
            return CycleReport(ts=as_of, equity=equity, alerts=alerts, halted=True)

        sleeve_targets = {s.name: s.compute_targets(as_of) for s in self.sleeves}
        res = self.portfolio.construct(sleeve_targets, equity)

        orders = self._rebalance(res.approved, positions, prices, as_of)

        equity = self._equity(prices)
        self.equity_curve.append((as_of, equity))
        self._persist(as_of)
        return CycleReport(
            ts=as_of, equity=equity, book=res.approved, orders=orders,
            violations=res.violations, alerts=alerts, halted=False,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _equity(self, prices: dict[str, float]) -> float:
        cash = self.adapter.get_cash()
        pos = self.adapter.get_positions()
        return float(cash + sum(q * prices.get(s, 0.0) for s, q in pos.items()))

    def _rebalance(self, book, positions, prices, as_of) -> dict[str, float]:
        """Trade from current positions to the target book at `prices`."""
        equity = self._equity(prices)
        orders: dict[str, float] = {}
        symbols = set(book) | set(positions)
        for s in symbols:
            px = prices.get(s, 0.0)
            if px <= 0:
                continue
            target_shares = (book.get(s, 0.0) * equity) / px
            delta = target_shares - positions.get(s, 0.0)
            if abs(delta * px) < self.min_trade_value:
                continue
            self.adapter.submit_order(Order(symbol=s, qty=delta, order_type="market"))
            if isinstance(self.adapter, SimulatedAdapter):
                bar = pd.Series({"open": px, "high": px, "low": px, "close": px}, name=as_of)
                self.adapter.process_bar(bar, s)
            orders[s] = delta
        return orders

    def _persist(self, as_of: pd.Timestamp) -> None:
        if self.state_path is None:
            return
        st = TradingState(
            cash=self.adapter.get_cash(),
            positions=self.adapter.get_positions(),
            last_bar_ts=as_of,
        )
        self._store.save(st, self.state_path)

    @property
    def returns(self) -> pd.Series:
        if len(self.equity_curve) < 2:
            return pd.Series(dtype=float, name="returns")
        ts, vals = zip(*self.equity_curve)
        return pd.Series(list(vals), index=list(ts), name="equity").pct_change().dropna()
