"""
Intraday execution simulator — the realism layer between a per-minute target
signal and a tradable PnL (spec §5, continuation instructions Task 5).

For minute-level trading the alpha and the fill are inseparable, so this models
what a taker actually pays:

  - NEXT-BAR execution: a target from a CLOSED bar t fills on bar t+1's open. No
    same-bar fill at the breakout level (that optimism is what we are replacing).
  - SPREAD CROSSING: buys lift the ask, sells hit the bid. Spread comes from
    quotes (bid/ask/spread columns) when present, else an assumed bps.
  - SLIPPAGE + COMMISSION on top of the spread.
  - PARTICIPATION CAP: a single bar can absorb at most `max_participation` of its
    volume; the remainder is a partial fill (carried, retried next bar).
  - SKIP RULES: skip a fill if the spread is too wide or the bar volume too thin
    (logged, reason-tagged).
  - EOD LIQUIDATION: force flat on each session's last bar (market-on-close),
    bypassing the participation cap — being flat by the close is non-negotiable.

Single symbol, pure, deterministic. A book is the mean of per-symbol results.
The flat-cost research engine (mft/backtest/intraday_session.py) stays as the
fast approximation; THIS is what Step 7 uses to re-validate with real fills.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 1.5 bp one-way ~= BASE_COST_PER_TRADE in mft/backtest/intraday_session.py.
DEFAULT_SLIPPAGE_BPS = 1.5


@dataclass(frozen=True)
class IntradayExecConfig:
    notional: float = 100_000.0          # capital deployed at |weight|=1
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS   # one-way, beyond the spread
    commission_bps: float = 0.0          # commission as bps of traded notional
    assumed_spread_bps: float = 0.0      # used when quotes are absent
    max_participation: float = 0.10      # max shares as a fraction of bar volume
    min_bar_volume: float = 0.0          # skip a fill if bar volume below this
    max_spread_bps: float = 50.0         # skip a fill if spread wider than this
    flatten_at_close: bool = True        # force flat on each session's last bar
    tz: str = "America/New_York"


@dataclass(frozen=True)
class Fill:
    ts: pd.Timestamp
    side: int            # +1 buy, -1 sell
    shares: float
    price: float         # includes spread + slippage
    reason: str          # 'fill' | 'partial' | 'eod' | 'skipped-spread' | 'skipped-volume'


@dataclass
class IntradayExecResult:
    fills: list[Fill] = field(default_factory=list)
    skipped: list[Fill] = field(default_factory=list)
    position: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def n_fills(self) -> int:
        return len(self.fills)


def _spread_bps_array(bars: pd.DataFrame, config: IntradayExecConfig) -> np.ndarray:
    """Per-bar spread in bps, vectorized: from `spread`, else `ask-bid`, else the
    assumed fallback. Pre-computed once so the fill loop stays O(n) and fast."""
    n = len(bars)
    out = np.full(n, float(config.assumed_spread_bps))
    close = bars["close"].to_numpy(dtype=float)
    if "spread" in bars.columns:
        spread = pd.to_numeric(bars["spread"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(spread) & (close > 0)
        out[ok] = spread[ok] / close[ok] * 1e4
    if "bid" in bars.columns and "ask" in bars.columns:
        bid = pd.to_numeric(bars["bid"], errors="coerce").to_numpy(dtype=float)
        ask = pd.to_numeric(bars["ask"], errors="coerce").to_numpy(dtype=float)
        mid = (bid + ask) / 2.0
        # only fill in where the spread column didn't already provide a value
        no_spread = out == config.assumed_spread_bps
        ok = np.isfinite(bid) & np.isfinite(ask) & (mid > 0) & no_spread
        out[ok] = (ask[ok] - bid[ok]) / mid[ok] * 1e4
    return out


def _fill_price(open_px: float, side: int, spread_bps: float, config: IntradayExecConfig) -> float:
    """Buys lift the ask and pay slippage; sells hit the bid and pay slippage."""
    half = (spread_bps / 2.0 + config.slippage_bps) * 1e-4
    return open_px * (1.0 + side * half)


def simulate_intraday(
    bars: pd.DataFrame,
    target_weight: pd.Series,
    config: IntradayExecConfig = IntradayExecConfig(),
) -> IntradayExecResult:
    """
    Simulate taker execution of `target_weight` (signed, in [-1, 1], the signal on
    each CLOSED bar) over `bars` (normalized OHLCV, optional bid/ask/spread).
    Returns fills, the share-position path, and a mark-to-market equity/return curve.
    """
    bars = bars.sort_index()
    n = len(bars)
    if n == 0:
        return IntradayExecResult()

    tw = target_weight.reindex(bars.index).fillna(0.0).to_numpy(dtype=float)
    opens = bars["open"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)
    vols = bars["volume"].to_numpy(dtype=float)
    sbps_arr = _spread_bps_array(bars, config)
    et_date = bars.index.tz_convert(config.tz).date

    pos_shares = 0.0
    held_w = 0.0          # the discrete position STATE we currently hold (-1/0/+1)
    cash = 0.0
    fills: list[Fill] = []
    skipped: list[Fill] = []
    positions = np.empty(n)
    equities = np.empty(n)

    for i in range(n):
        ts = bars.index[i]
        # Target effective at bar i is the signal from the PREVIOUS closed bar.
        desired_w = tw[i - 1] if i > 0 else 0.0
        is_session_last = (i == n - 1) or (et_date[i] != et_date[i + 1])
        force_flat = config.flatten_at_close and is_session_last
        if force_flat:
            desired_w = 0.0

        # Trade ONLY on a state transition. Between transitions we hold fixed
        # shares — a hold-to-close signal must NOT re-trade every bar to chase a
        # constant notional as price drifts (that manufactures spurious churn).
        if desired_w != held_w:
            desired_shares = desired_w * config.notional / opens[i]
            delta = desired_shares - pos_shares
            if abs(delta) > 1e-9:
                side = int(np.sign(delta))
                sbps = sbps_arr[i]
                if not force_flat and sbps > config.max_spread_bps:
                    skipped.append(Fill(ts, side, 0.0, opens[i], "skipped-spread"))
                    # transition not completed -> retry next bar (held_w unchanged)
                elif not force_flat and vols[i] < config.min_bar_volume:
                    skipped.append(Fill(ts, side, 0.0, opens[i], "skipped-volume"))
                else:
                    reason = "eod" if force_flat else "fill"
                    # EOD liquidation must complete -> bypasses the participation cap.
                    if not force_flat:
                        cap = config.max_participation * vols[i]
                        if abs(delta) > cap:
                            delta = side * cap
                            reason = "partial"
                    shares = abs(delta)
                    price = _fill_price(opens[i], side, sbps, config)
                    commission = config.commission_bps * 1e-4 * shares * price
                    cash -= side * shares * price + commission
                    pos_shares += delta
                    held_w = desired_w
                    fills.append(Fill(ts, side, shares, price, reason))
            else:
                held_w = desired_w

        positions[i] = pos_shares
        equities[i] = cash + pos_shares * closes[i]

    pos = pd.Series(positions, index=bars.index, name="position")
    eq = pd.Series(equities, index=bars.index, name="equity")
    rets = (eq.diff() / config.notional).fillna(0.0).rename("return")
    return IntradayExecResult(fills=fills, skipped=skipped, position=pos, equity=eq, returns=rets)
