"""
Execution adapter interface.

The sim and live adapters implement the same interface.
Swapping from backtesting to paper to live changes ONLY the adapter —
strategy logic is untouched. This is the one code path guarantee.

Phase 0: SimulatedAdapter only.
Phase 2: NautilusTrader adapter.
Phase 7: Interactive Brokers adapter via NautilusTrader.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mft.execution.costs import DEFAULT_COMMISSION_PCT, DEFAULT_SLIPPAGE_PCT


@dataclass
class Order:
    symbol: str
    qty: float           # positive = buy, negative = sell
    order_type: str      # "market" | "limit"
    limit_price: float | None = None
    client_id: str | None = None


@dataclass
class Fill:
    order: Order
    fill_price: float
    fill_qty: float
    commission: float
    timestamp: pd.Timestamp


class ExecutionAdapter(ABC):
    """
    Identical interface for simulation and live trading.
    Swap the concrete class; never change strategy code.
    """

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit order. Returns broker/sim order ID."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if successfully cancelled."""
        ...

    @abstractmethod
    def get_positions(self) -> dict[str, float]:
        """Current positions: symbol → quantity (+ long, - short)."""
        ...

    @abstractmethod
    def get_cash(self) -> float:
        """Available cash."""
        ...


class SimulatedAdapter(ExecutionAdapter):
    """
    Simulated execution. Orders fill at next-bar open with slippage.
    Used by the event-driven harness.
    """

    def __init__(
        self,
        init_cash: float = 100_000,
        commission_pct: float = DEFAULT_COMMISSION_PCT,
        slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    ):
        self._cash = init_cash
        self._positions: dict[str, float] = {}
        self._commission_pct = commission_pct
        self._slippage_pct = slippage_pct
        self._pending: list[tuple[str, Order]] = []
        self._fills: list[Fill] = []
        self._order_counter = 0

    def submit_order(self, order: Order) -> str:
        self._order_counter += 1
        oid = f"SIM-{self._order_counter:06d}"
        self._pending.append((oid, order))
        return oid

    def cancel_order(self, order_id: str) -> bool:
        before = len(self._pending)
        self._pending = [(oid, o) for oid, o in self._pending if oid != order_id]
        return len(self._pending) < before

    def get_positions(self) -> dict[str, float]:
        return {s: q for s, q in self._positions.items() if abs(q) > 1e-8}

    def get_cash(self) -> float:
        return self._cash

    def set_state(self, cash: float, positions: dict[str, float]) -> None:
        """Restore broker state after a crash/restart (paper-sim source of truth)."""
        self._cash = float(cash)
        self._positions = {s: float(q) for s, q in positions.items()}

    def process_bar(self, bar: pd.Series, symbol: str) -> list[Fill]:
        """Fill pending orders for this symbol at bar open with slippage."""
        fills = []
        remaining = []

        for oid, order in self._pending:
            if order.symbol != symbol:
                remaining.append((oid, order))
                continue

            direction = np.sign(order.qty) if order.qty != 0 else 0
            exec_price = bar["open"] * (1 + self._slippage_pct * direction)
            notional = order.qty * exec_price
            commission = abs(notional) * self._commission_pct

            fill = Fill(
                order=order,
                fill_price=exec_price,
                fill_qty=order.qty,
                commission=commission,
                timestamp=bar.name,
            )
            self._cash -= notional + commission
            self._positions[symbol] = self._positions.get(symbol, 0.0) + order.qty
            fills.append(fill)
            self._fills.append(fill)

        self._pending = remaining
        return fills
