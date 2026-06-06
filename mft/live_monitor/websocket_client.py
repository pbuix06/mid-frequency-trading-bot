"""
Market-data stream sources — READ-ONLY. No order endpoints are ever touched.

A source yields closed 1-minute frames `(now, {symbol: Bar})` that the LiveMonitor consumes.
Three implementations, in order of test-ability:

  - ReplayStream         : deterministic, offline. Replays stored/synthetic 1m bars minute
                           by minute. Used by tests and by `--source replay`. The safe default.
  - BinanceRestPoller    : near-live. Polls Binance public REST for the latest CLOSED 1m
                           kline per symbol (+ bookTicker for spread). Uses `requests` only.
  - BinanceWebSocketClient: seconds-level. Subscribes to public kline/bookTicker streams and
                           assembles 1m bars via BarBuilder. Lazily imports `websocket-client`
                           (optional dep); if absent it points you at the REST poller.

ALL three are market-data consumers. None can place, route, or cancel an order — this module
imports no trading endpoint. Live trading is not approved (see mft.automation.registry).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import pandas as pd

from mft.live_monitor.bar_builder import Bar, BarBuilder, Tick, floor_to_minute

SPOT_WS = "wss://stream.binance.com:9443/stream"
SPOT_REST = "https://api.binance.com/api/v3"


class MarketDataStream(ABC):
    """A source of closed 1-minute bar frames. Read-only by construction."""

    @abstractmethod
    def minute_frames(self) -> Iterator[tuple[pd.Timestamp, dict[str, Bar]]]:
        """Yield (frame_close_time_utc, {symbol: closed Bar}) in chronological order."""
        ...


class ReplayStream(MarketDataStream):
    """
    Replay stored 1m OHLCV frames (one DataFrame per symbol) minute by minute.

    Each input frame must have a UTC DatetimeIndex and open/high/low/close/volume columns
    (the schema of mft.research.crypto_panel / a Binance kline dump). Optionally a `spread_bps`
    column per symbol can be supplied via `spreads` to exercise the spread risk check offline.
    """

    def __init__(self, bars_by_symbol: dict[str, pd.DataFrame],
                 spreads_by_symbol: dict[str, pd.Series] | None = None):
        self.bars = {s: df.sort_index() for s, df in bars_by_symbol.items()}
        self.spreads = spreads_by_symbol or {}
        idx = pd.DatetimeIndex([])
        for df in self.bars.values():
            idx = idx.union(df.index)
        self.timeline = idx.sort_values()

    def minute_frames(self) -> Iterator[tuple[pd.Timestamp, dict[str, Bar]]]:
        for ts in self.timeline:
            frame: dict[str, Bar] = {}
            for sym, df in self.bars.items():
                if ts not in df.index:
                    continue
                row = df.loc[ts]
                bb = ba = None
                sp = self.spreads.get(sym)
                if sp is not None and ts in sp.index and pd.notna(sp.loc[ts]):
                    half = float(sp.loc[ts]) * 1e-4 * 0.5 * float(row["close"])
                    bb, ba = float(row["close"]) - half, float(row["close"]) + half
                frame[sym] = Bar.from_mapping(ts, row, best_bid=bb, best_ask=ba)
            if frame:
                yield floor_to_minute(ts), frame


class BinanceRestPoller(MarketDataStream):
    """
    Near-live source: poll Binance public REST for the latest CLOSED 1m kline per symbol.

    Network only — no key, market data only. Emits one frame per completed minute. Bounded by
    `max_minutes` (None = run until interrupted). `bookticker=True` attaches top-of-book spread.
    This is the simplest real source; for seconds-level updates use BinanceWebSocketClient.
    """

    def __init__(self, symbols: list[str], poll_seconds: float = 5.0,
                 max_minutes: int | None = None, bookticker: bool = True,
                 base: str = SPOT_REST):
        import requests
        self.symbols = list(symbols)
        self.poll_seconds = poll_seconds
        self.max_minutes = max_minutes
        self.bookticker = bookticker
        self.base = base
        self._s = requests.Session()
        self._s.headers.update({"User-Agent": "mft-live-monitor/1.0 (read-only)"})

    def _latest_closed_bar(self, symbol: str) -> Bar | None:
        rows = self._s.get(f"{self.base}/klines",
                           params={"symbol": symbol, "interval": "1m", "limit": 2},
                           timeout=15).json()
        if not isinstance(rows, list) or len(rows) < 2:
            return None
        k = rows[-2]  # rows[-1] is the still-forming bar; rows[-2] is the last CLOSED one
        ts = pd.to_datetime(int(k[0]), unit="ms", utc=True)
        bb = ba = None
        if self.bookticker:
            bt = self._s.get(f"{self.base}/ticker/bookTicker",
                             params={"symbol": symbol}, timeout=15).json()
            if isinstance(bt, dict) and "bidPrice" in bt:
                bb, ba = float(bt["bidPrice"]), float(bt["askPrice"])
        return Bar(ts=floor_to_minute(ts), open=float(k[1]), high=float(k[2]),
                   low=float(k[3]), close=float(k[4]), volume=float(k[5]),
                   trades=int(k[8]), best_bid=bb, best_ask=ba)

    def minute_frames(self) -> Iterator[tuple[pd.Timestamp, dict[str, Bar]]]:
        import time
        last_emitted: pd.Timestamp | None = None
        emitted = 0
        while self.max_minutes is None or emitted < self.max_minutes:
            frame: dict[str, Bar] = {}
            newest: pd.Timestamp | None = None
            for sym in self.symbols:
                bar = self._latest_closed_bar(sym)
                if bar is None:
                    continue
                frame[sym] = bar
                newest = bar.ts if newest is None else max(newest, bar.ts)
            if frame and newest is not None and newest != last_emitted:
                last_emitted = newest
                emitted += 1
                yield newest + pd.Timedelta(minutes=1), frame
            time.sleep(self.poll_seconds)


class BinanceWebSocketClient(MarketDataStream):
    """
    Seconds-level source: public kline_1m (+ optional bookTicker) websocket -> 1m bars.

    Lazily imports `websocket-client` (NOT a project dependency). If it is not installed this
    raises with a clear pointer to BinanceRestPoller. Market data only — the URL is a public
    market stream; there is no authenticated/order socket anywhere in this class.
    """

    def __init__(self, symbols: list[str], with_bookticker: bool = True,
                 max_minutes: int | None = None, url: str = SPOT_WS):
        self.symbols = list(symbols)
        self.with_bookticker = with_bookticker
        self.max_minutes = max_minutes
        self.url = url
        self._builders = {s: BarBuilder(s) for s in symbols}

    def _streams(self) -> str:
        parts = [f"{s.lower()}@kline_1m" for s in self.symbols]
        if self.with_bookticker:
            parts += [f"{s.lower()}@bookTicker" for s in self.symbols]
        return "/".join(parts)

    def minute_frames(self) -> Iterator[tuple[pd.Timestamp, dict[str, Bar]]]:
        try:
            import json

            import websocket  # type: ignore  # optional dep: `pip install websocket-client`
        except ImportError as e:  # pragma: no cover - exercised only without the optional dep
            raise ImportError(
                "BinanceWebSocketClient needs `websocket-client` (pip install websocket-client). "
                "For a dependency-free near-live feed use BinanceRestPoller instead."
            ) from e

        ws = websocket.create_connection(f"{self.url}?streams={self._streams()}", timeout=30)
        emitted = 0
        try:
            while self.max_minutes is None or emitted < self.max_minutes:
                msg = json.loads(ws.recv())
                data = msg.get("data", {})
                stream = msg.get("stream", "")
                if stream.endswith("@bookTicker"):
                    sym = data.get("s")
                    if sym in self._builders:
                        self._builders[sym].update_book(float(data["b"]), float(data["a"]))
                    continue
                k = data.get("k")
                if not k:
                    continue
                sym = data.get("s")
                bb = self._builders.get(sym)
                if bb is None:
                    continue
                tick = Tick(ts=pd.to_datetime(int(k["t"]), unit="ms", utc=True),
                            price=float(k["c"]), size=float(k["v"]))
                done = bb.add_tick(tick)
                if k.get("x") and done is not None:   # k['x']=True ⇒ this kline just closed
                    emitted += 1
                    yield done.ts + pd.Timedelta(minutes=1), {sym: done}
        finally:
            ws.close()
