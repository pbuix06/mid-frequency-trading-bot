"""
Trading state persistence for crash recovery.

State that must survive a process restart:
  - cash: current available cash
  - positions: open positions by symbol (symbol → qty)
  - last_signals: last computed signal per symbol (prevents spurious re-entry on restart)
  - last_bar_ts: timestamp of last fully processed bar (determines resume point)

Write is atomic (write to .tmp → os.replace) so a crash mid-write never
produces a corrupt state file. The worst case is the pre-crash version
survives intact.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class TradingState:
    """Minimal persistent state needed to resume trading after a crash."""

    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    last_signals: dict[str, float] = field(default_factory=dict)
    last_bar_ts: pd.Timestamp | None = None


class StateStore:
    """
    Atomic JSON state persistence.

    Usage:
        store = StateStore()
        store.save(state, Path("run/state.json"))
        state = store.load(Path("run/state.json"))
    """

    def save(self, state: TradingState, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cash": state.cash,
            "positions": state.positions,
            "last_signals": state.last_signals,
            "last_bar_ts": state.last_bar_ts.isoformat() if state.last_bar_ts is not None else None,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, path)  # atomic on POSIX; prevents partial-write corruption

    def load(self, path: Path) -> TradingState:
        payload = json.loads(Path(path).read_text())
        raw_ts = payload.get("last_bar_ts")
        return TradingState(
            cash=float(payload["cash"]),
            positions={k: float(v) for k, v in payload.get("positions", {}).items()},
            last_signals={k: float(v) for k, v in payload.get("last_signals", {}).items()},
            last_bar_ts=pd.Timestamp(raw_ts) if raw_ts is not None else None,
        )

    def exists(self, path: Path) -> bool:
        return Path(path).exists()
