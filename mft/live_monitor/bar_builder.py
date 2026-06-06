"""
Tick/stream -> 1-minute bar aggregation.

The seconds-level core: a websocket (or any sub-minute) source pushes individual
trades/quotes; `BarBuilder` rolls them into closed 1-minute OHLCV bars and emits a bar
the instant the clock crosses into a new minute. A REST source that already returns
*closed* 1m klines can skip the builder and use `Bar.from_mapping(...)` directly.

NO orders, NO network here — pure aggregation, fully deterministic and offline-testable.
All timestamps are UTC; bar `ts` is the minute OPEN (floored), matching the project's
left-labelled bar convention (see mft.research.crypto_panel.resample_crypto_bars).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def floor_to_minute(ts) -> pd.Timestamp:
    """UTC minute-floor of any timestamp-like (naive assumed UTC)."""
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    return t.floor("min")


@dataclass
class Tick:
    """One sub-minute observation: a trade print (and/or a book update)."""
    ts: pd.Timestamp
    price: float
    size: float = 0.0


@dataclass
class Bar:
    """A closed 1-minute OHLCV bar, optionally stamped with last top-of-book."""
    ts: pd.Timestamp                  # minute OPEN, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int = 0
    best_bid: float | None = None
    best_ask: float | None = None

    @property
    def mid(self) -> float:
        if self.best_bid and self.best_ask and self.best_bid > 0 and self.best_ask > 0:
            return 0.5 * (self.best_bid + self.best_ask)
        return self.close

    @property
    def spread_bps(self) -> float | None:
        """Top-of-book spread in bps, or None if no book was captured."""
        if self.best_bid and self.best_ask and self.best_bid > 0 and self.best_ask > 0:
            return (self.best_ask - self.best_bid) / self.mid * 1e4
        return None

    @classmethod
    def from_mapping(cls, ts, row, best_bid: float | None = None,
                     best_ask: float | None = None) -> Bar:
        """Wrap an already-closed 1m OHLCV row (e.g. a REST kline / a DataFrame row)."""
        return cls(
            ts=floor_to_minute(ts),
            open=float(row["open"]), high=float(row["high"]),
            low=float(row["low"]), close=float(row["close"]),
            volume=float(row["volume"]),
            trades=int(row["trades"]) if row.get("trades") == row.get("trades")
            and row.get("trades") is not None else 0,
            best_bid=best_bid, best_ask=best_ask,
        )


class BarBuilder:
    """
    Streaming tick -> 1m bar aggregator for ONE symbol.

    Feed it ticks in (roughly) time order with `add_tick`; it returns a finished `Bar`
    exactly when a tick lands in a later minute than the one being built, else None.
    Out-of-order ticks that predate the current minute are ignored (not back-dated).
    `update_book` stamps the in-progress bar with the latest best bid/ask.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._cur: dict | None = None
        self._bid: float | None = None
        self._ask: float | None = None

    def update_book(self, best_bid: float, best_ask: float) -> None:
        self._bid, self._ask = best_bid, best_ask
        if self._cur is not None:
            self._cur["best_bid"], self._cur["best_ask"] = best_bid, best_ask

    def add_tick(self, tick: Tick) -> Bar | None:
        minute = floor_to_minute(tick.ts)
        if self._cur is None:
            self._start(minute, tick)
            return None
        if minute < self._cur["ts"]:
            return None                       # stale tick before the open bar: drop
        if minute > self._cur["ts"]:
            done = self._finalize()
            self._start(minute, tick)         # gaps (empty minutes) simply skipped
            return done
        c = self._cur                          # same minute: extend OHLCV
        c["high"] = max(c["high"], tick.price)
        c["low"] = min(c["low"], tick.price)
        c["close"] = tick.price
        c["volume"] += tick.size
        c["trades"] += 1
        return None

    def flush(self) -> Bar | None:
        """Finalize and return the in-progress bar (e.g. end of a replay). Clears state."""
        if self._cur is None:
            return None
        done = self._finalize()
        self._cur = None
        return done

    def current_partial(self) -> Bar | None:
        return self._finalize() if self._cur is not None else None

    # ── internals ──
    def _start(self, minute: pd.Timestamp, tick: Tick) -> None:
        self._cur = {"ts": minute, "open": tick.price, "high": tick.price,
                     "low": tick.price, "close": tick.price, "volume": tick.size,
                     "trades": 1, "best_bid": self._bid, "best_ask": self._ask}

    def _finalize(self) -> Bar:
        c = self._cur
        return Bar(ts=c["ts"], open=c["open"], high=c["high"], low=c["low"],
                   close=c["close"], volume=c["volume"], trades=c["trades"],
                   best_bid=c["best_bid"], best_ask=c["best_ask"])
