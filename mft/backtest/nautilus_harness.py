"""
NautilusTrader validation harness — Phase 2.

Wraps AlphaBase.compute_signal() in a NautilusTrader BacktestEngine.
Purpose: prove the single code-path principle holds across three engines.

The same AlphaBase object runs in:
  1. vectorbt_harness  (vectorised, fast research)
  2. event_harness     (bar-by-bar Python sim, validation)
  3. nautilus_harness  (this file — production-grade event engine, final proof)

All three must reconcile within tolerance before any alpha goes live.

Usage:
    result = run_nautilus(alpha, data, symbol="SPY")
    metrics = full_metrics(result.returns)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mft.alphas.base import AlphaBase

# ── Data conversion helpers ───────────────────────────────────────────────────

def _df_to_bars(df: pd.DataFrame, bar_type, price_precision: int = 2) -> list:
    """Convert OHLCV DataFrame to list[Bar]. Skips rows with non-positive close."""
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.objects import Price, Quantity

    scale = 10 ** price_precision
    fmt = f"{{:.{price_precision}f}}"
    bars = []
    for ts, row in df.iterrows():
        if row["close"] <= 0:
            continue
        # Round each price independently then enforce OHLC consistency.
        # Independent rounding can violate low <= open/close <= high.
        o = round(row["open"] * scale) / scale
        h = round(row["high"] * scale) / scale
        lo = round(row["low"] * scale) / scale
        c = round(row["close"] * scale) / scale
        tick = 1 / scale
        lo = min(lo, o, c)
        h = max(h, o, c)
        if h <= lo:
            h = lo + tick
        ts_ns = int(ts.value)
        bars.append(Bar(
            bar_type=bar_type,
            open=Price.from_str(fmt.format(o)),
            high=Price.from_str(fmt.format(h)),
            low=Price.from_str(fmt.format(lo)),
            close=Price.from_str(fmt.format(c)),
            volume=Quantity.from_str(f"{max(1, int(row['volume']))}"),
            ts_event=ts_ns,
            ts_init=ts_ns,
        ))
    return bars


def _bars_to_df(bars: list) -> pd.DataFrame:
    """Convert list[Bar] to pd.DataFrame for AlphaBase.compute_signal()."""
    return pd.DataFrame(
        {
            "open":   [float(b.open)   for b in bars],
            "high":   [float(b.high)   for b in bars],
            "low":    [float(b.low)    for b in bars],
            "close":  [float(b.close)  for b in bars],
            "volume": [float(b.volume) for b in bars],
        },
        index=pd.DatetimeIndex(
            [pd.Timestamp(b.ts_event, unit="ns", tz="UTC") for b in bars],
            name="timestamp",
        ),
    )


# ── Strategy ──────────────────────────────────────────────────────────────────

def _make_strategy(alpha: AlphaBase, symbol: str, venue_str: str, bar_type_str: str, initial_cash: float):
    """
    Return an AlphaStrategy instance that wraps alpha.compute_signal().
    Defined inside a function to avoid polluting the module namespace with
    NautilusTrader imports at import time (heavy C-extension load).
    """
    from nautilus_trader.model.data import BarType
    from nautilus_trader.model.enums import OrderSide, TimeInForce
    from nautilus_trader.model.events import OrderFilled
    from nautilus_trader.model.identifiers import InstrumentId, Venue
    from nautilus_trader.model.objects import Quantity
    from nautilus_trader.trading.strategy import Strategy, StrategyConfig

    class _Config(StrategyConfig, frozen=True):
        symbol: str
        venue: str
        bar_type_str: str

    class _Strategy(Strategy):
        def __init__(self, cfg: _Config, alpha: AlphaBase, initial_cash: float) -> None:
            super().__init__(cfg)
            self._alpha = alpha
            self._symbol = cfg.symbol
            self._venue = Venue(cfg.venue)
            self._bar_type_str = cfg.bar_type_str
            self._instrument_id = InstrumentId.from_str(f"{cfg.symbol}.{cfg.venue}")
            self._initial_cash = initial_cash

            # manual accounting (mirrors event_harness logic)
            self._cash: float = initial_cash
            self._shares: int = 0
            self._prev_signal: float | None = None
            self._window: list = []

            # output
            self.equity_curve: list[tuple[pd.Timestamp, float]] = []
            self.fills_count: int = 0

        def on_start(self) -> None:
            self.subscribe_bars(BarType.from_str(self._bar_type_str))

        def on_bar(self, bar) -> None:
            self._window.append(bar)

            close = float(bar.close)
            ts = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")
            equity = self._cash + self._shares * close
            self.equity_curve.append((ts, equity))

            if len(self._window) < self._alpha.lookback:
                return

            window_df = _bars_to_df(self._window[-(self._alpha.lookback + 1):])
            signals = self._alpha.compute_signal(window_df)
            target = signals.get(self._symbol, 0.0)

            if target == self._prev_signal:
                return
            self._prev_signal = target

            if target > 0 and self._shares == 0:
                qty = max(1, int(self._cash / close))
                self.submit_order(self.order_factory.market(
                    instrument_id=self._instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=Quantity.from_int(qty),
                    time_in_force=TimeInForce.GTC,
                ))
            elif target == 0 and self._shares > 0:
                self.submit_order(self.order_factory.market(
                    instrument_id=self._instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=Quantity.from_int(self._shares),
                    time_in_force=TimeInForce.GTC,
                ))

        def on_order_filled(self, event: OrderFilled) -> None:
            qty = int(float(event.last_qty))
            price = float(event.last_px)
            if event.is_buy:
                self._shares += qty
                self._cash -= qty * price
            else:
                self._shares -= qty
                self._cash += qty * price
            self.fills_count += 1

        def on_stop(self) -> None:
            pass

    cfg = _Config(symbol=symbol, venue=venue_str, bar_type_str=bar_type_str)
    return _Strategy(cfg=cfg, alpha=alpha, initial_cash=initial_cash)


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class NautilusResult:
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    n_fills: int = 0

    @property
    def returns(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float, name="returns")
        ts, vals = zip(*self.equity_curve)
        equity = pd.Series(list(vals), index=list(ts), name="equity")
        return equity.pct_change().dropna()

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1][1] if self.equity_curve else 0.0


# ── Main entry point ──────────────────────────────────────────────────────────

def run_nautilus(
    alpha: AlphaBase,
    data: pd.DataFrame,
    symbol: str,
    *,
    initial_cash: float = 100_000.0,
    venue_str: str = "SIM",
    price_precision: int = 2,
) -> NautilusResult:
    """
    Run alpha through NautilusTrader BacktestEngine on daily bars.

    Args:
        alpha:        Any AlphaBase implementation.
        data:         OHLCV DataFrame with UTC DatetimeIndex.
        symbol:       Ticker string (e.g. "SPY").
        initial_cash: Starting portfolio value.
        venue_str:    Simulated venue name.

    Returns:
        NautilusResult with equity_curve, n_fills, and returns property.
    """
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.data import BarSpecification, BarType
    from nautilus_trader.model.enums import (
        AccountType,
        AggregationSource,
        BarAggregation,
        OmsType,
        PriceType,
    )
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
    from nautilus_trader.model.instruments import Equity
    from nautilus_trader.model.objects import Money, Price, Quantity

    venue = Venue(venue_str)
    instrument_id = InstrumentId(Symbol(symbol), venue)

    bar_spec = BarSpecification(step=1, aggregation=BarAggregation.DAY, price_type=PriceType.LAST)
    bar_type = BarType(instrument_id, bar_spec, AggregationSource.EXTERNAL)
    bar_type_str = str(bar_type)

    engine = BacktestEngine(config=BacktestEngineConfig(
        logging=LoggingConfig(log_level="ERROR", bypass_logging=True),
    ))

    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money(initial_cash, USD)],
    )

    tick_size = f"{'0.' + '0' * (price_precision - 1) + '1' if price_precision > 0 else '1'}"
    instrument = Equity(
        instrument_id=instrument_id,
        raw_symbol=Symbol(symbol),
        currency=USD,
        price_precision=price_precision,
        price_increment=Price.from_str(tick_size),
        lot_size=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
    )
    engine.add_instrument(instrument)

    bars = _df_to_bars(data, bar_type, price_precision=price_precision)
    engine.add_data(bars)

    strategy = _make_strategy(
        alpha=alpha,
        symbol=symbol,
        venue_str=venue_str,
        bar_type_str=bar_type_str,
        initial_cash=initial_cash,
    )
    engine.add_strategy(strategy)
    engine.run()

    return NautilusResult(
        equity_curve=strategy.equity_curve,
        n_fills=strategy.fills_count,
    )
