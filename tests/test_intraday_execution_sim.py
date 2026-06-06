"""
Execution-realism tests for the intraday simulator (spec §5, Task 5/Step 6).

Prove the simulator charges what a taker actually pays: next-bar fills, spread
crossing, participation caps, skip rules, forced EOD liquidation, and that cost
strictly reduces PnL.
"""

from __future__ import annotations

import pandas as pd

from mft.execution.intraday_sim import IntradayExecConfig, simulate_intraday


def _bars(values, day="2023-01-03", *, volume=1e6, bid=None, ask=None, spread=None):
    start = pd.Timestamp(f"{day} 09:30", tz="America/New_York")
    idx = pd.date_range(start, periods=len(values), freq="1min").tz_convert("UTC")
    c = pd.Series(values, index=idx, dtype=float)
    data = {"open": c, "high": c, "low": c, "close": c, "volume": float(volume)}
    if bid is not None:
        data["bid"] = float(bid)
    if ask is not None:
        data["ask"] = float(ask)
    if spread is not None:
        data["spread"] = float(spread)
    return pd.DataFrame(data, index=idx)


def _target(index, weights: dict[int, float]) -> pd.Series:
    s = pd.Series(0.0, index=index)
    for i, w in weights.items():
        s.iloc[i] = w
    return s


def test_next_bar_execution():
    """A target on closed bar t fills on bar t+1's open, not bar t."""
    bars = _bars([100, 101, 102, 103, 104, 105], volume=1e6)
    tw = _target(bars.index, {2: 1.0})  # signal becomes +1 at bar index 2
    cfg = IntradayExecConfig(slippage_bps=0.0, flatten_at_close=False)
    res = simulate_intraday(bars, tw, cfg)
    assert res.n_fills >= 1
    first = res.fills[0]
    assert first.ts == bars.index[3]            # acted on the NEXT bar
    assert abs(first.price - bars["open"].iloc[3]) < 1e-9


def test_spread_crossing_buy_pays_ask_sell_hits_bid():
    bars = _bars([100, 100, 100, 100], bid=99.5, ask=100.5)  # 100 bp spread
    cfg = IntradayExecConfig(slippage_bps=0.0, flatten_at_close=False, max_spread_bps=1e9)
    buy = simulate_intraday(bars, _target(bars.index, {0: 1.0}), cfg)
    assert buy.fills[0].side == 1
    assert buy.fills[0].price > 100.0           # lifted the ask

    sell = simulate_intraday(bars, _target(bars.index, {0: -1.0}), cfg)
    assert sell.fills[0].side == -1
    assert sell.fills[0].price < 100.0          # hit the bid


def test_participation_cap_creates_partial_fills():
    # Tiny bar volume -> can't fill the full target in one bar.
    bars = _bars([100, 100, 100, 100], volume=10.0)
    cfg = IntradayExecConfig(
        notional=100_000.0, slippage_bps=0.0, max_participation=0.10, flatten_at_close=False
    )
    # Want 1000 shares ($100k/$100); cap = 0.1*10 = 1 share/bar. Hold long the
    # whole time so the cap keeps binding on the way IN (never reaches target).
    res = simulate_intraday(bars, pd.Series(1.0, index=bars.index), cfg)
    assert all(f.reason == "partial" for f in res.fills)
    assert res.position.iloc[1] <= 1.0 + 1e-9   # only ~1 share got filled
    assert res.position.iloc[-1] < 5.0          # nowhere near the 1000-share target


def test_skip_when_spread_too_wide():
    bars = _bars([100, 100, 100], spread=5.0)   # 500 bp spread
    cfg = IntradayExecConfig(slippage_bps=0.0, max_spread_bps=50.0, flatten_at_close=False)
    res = simulate_intraday(bars, _target(bars.index, {0: 1.0}), cfg)
    assert res.n_fills == 0
    assert any(s.reason == "skipped-spread" for s in res.skipped)
    assert (res.position == 0.0).all()


def test_skip_when_volume_too_low():
    bars = _bars([100, 100, 100], volume=5.0)
    cfg = IntradayExecConfig(slippage_bps=0.0, min_bar_volume=100.0, flatten_at_close=False)
    res = simulate_intraday(bars, _target(bars.index, {0: 1.0}), cfg)
    assert res.n_fills == 0
    assert any(s.reason == "skipped-volume" for s in res.skipped)


def test_eod_liquidation_forces_flat():
    """Holding +1 into the close must be liquidated; the session ends flat."""
    bars = _bars([100, 101, 102, 103], volume=1e6)
    tw = _target(bars.index, {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})  # always long
    cfg = IntradayExecConfig(slippage_bps=0.0, flatten_at_close=True)
    res = simulate_intraday(bars, tw, cfg)
    assert abs(res.position.iloc[-1]) < 1e-9            # flat by close
    assert res.fills[-1].reason == "eod"


def test_no_overnight_across_two_sessions():
    day1 = _bars([100, 101, 102, 103], day="2023-01-03", volume=1e6)
    day2 = _bars([110, 111, 112, 113], day="2023-01-04", volume=1e6)
    bars = pd.concat([day1, day2])
    tw = pd.Series(1.0, index=bars.index)               # long the whole time
    cfg = IntradayExecConfig(slippage_bps=0.0, flatten_at_close=True)
    res = simulate_intraday(bars, tw, cfg)
    # Flat at the end of BOTH sessions.
    assert abs(res.position.loc[day1.index[-1]]) < 1e-9
    assert abs(res.position.loc[day2.index[-1]]) < 1e-9


def test_cost_strictly_reduces_pnl():
    """Same price path: a round trip with spread+slippage ends below zero-cost."""
    bars = _bars([100, 101, 102, 101, 100], spread=0.2, volume=1e6)  # 20 bp spread
    tw = _target(bars.index, {0: 1.0, 1: 1.0, 2: 0.0})  # long then flat
    free = IntradayExecConfig(slippage_bps=0.0, assumed_spread_bps=0.0, flatten_at_close=True)
    paid = IntradayExecConfig(slippage_bps=1.5, flatten_at_close=True)  # uses the 20bp quote
    eq_free = simulate_intraday(bars, tw, free).equity.iloc[-1]
    eq_paid = simulate_intraday(bars, tw, paid).equity.iloc[-1]
    assert eq_paid < eq_free
