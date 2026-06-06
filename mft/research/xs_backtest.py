"""
Cross-sectional long-short backtest (5-minute bars, intraday, flat overnight).

Design choices, stated so they can't hide:
  - DOLLAR-NEUTRAL top/bottom-N: long the top_n signal names (+0.5/n each), short the
    bottom_n (-0.5/n each) -> $1 gross, $0 net. We want alpha, not beta.
  - NON-OVERLAPPING: rebalance every `hold_bars`, so only one position lifecycle is
    open at a time. This sidesteps the overlapping-position bookkeeping trap AND keeps
    the return series serially independent (so the Sharpe isn't autocorrelation-inflated).
    Overlapping (Jegadeesh-Titman averaging) is a documented later extension.
  - ENTRY LAG 1: decision at close of bar t -> enter at close of t+1 -> exit t+1+hold.
    Uses the SAME forward_return as signal_lab, so grading and trading agree.
  - COST: charged on turnover. Opening the $1-gross book trades $1; closing it trades
    $1; so a round trip = 2 units * cost_per_side. Reported gross AND net.

A 33-name, 2.9-yr, survivor universe cannot yield a deployable XS alpha (see project
breadth finding T0056). This proves the ENGINE and measures the signal honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mft.research.targets import forward_return_panel
from mft.validation.metrics import max_drawdown, sharpe, sortino


def _et_date(index: pd.DatetimeIndex) -> np.ndarray:
    return index.tz_convert("America/New_York").normalize().tz_localize(None).to_numpy()


def _utc_date(index: pd.DatetimeIndex) -> np.ndarray:
    return index.tz_convert("UTC").normalize().tz_localize(None).to_numpy()


def _rebalance_points(index: pd.DatetimeIndex, hold_bars: int,
                      tod_windows: list[tuple[str, str]] | None = None,
                      continuous: bool = False) -> pd.DatetimeIndex:
    """
    Non-overlapping decision bars.

    Equity (default): every hold_bars-th bar WITHIN each ET calendar day (the grid
    resets at the daily open; positions go flat overnight). `tod_windows` restricts to
    ET time windows (config 8).

    Continuous (24/7, crypto): every hold_bars-th bar across the WHOLE series with NO
    daily reset — uniform spacing through UTC/ET midnight and weekends. tod_windows is
    rejected (it is an equity-session concept).
    """
    if continuous:
        if tod_windows:
            raise ValueError("tod_windows is equity-session logic; not valid in continuous (24/7) mode")
        return index[np.arange(0, len(index), hold_bars)]

    dates = _et_date(index)
    keep = np.zeros(len(index), dtype=bool)
    _, starts = np.unique(dates, return_index=True)
    starts = np.sort(starts)
    bounds = list(starts) + [len(index)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = np.arange(a, b, hold_bars)
        keep[sel] = True
    reb = index[keep]
    if tod_windows:
        et_t = reb.tz_convert("America/New_York").time
        in_win = np.zeros(len(reb), dtype=bool)
        for lo, hi in tod_windows:
            tlo, thi = pd.Timestamp(lo).time(), pd.Timestamp(hi).time()
            in_win |= np.array([tlo <= t < thi for t in et_t])
        reb = reb[in_win]
    return reb


@dataclass
class BacktestResult:
    gross: pd.Series
    net: pd.Series
    metrics: dict
    n_trades: int
    trades_per_day: float
    periods_per_year: float

    def equity(self) -> pd.Series:
        return (1.0 + self.net).cumprod()


def cross_sectional_ls(
    signal_wide: pd.DataFrame,
    close_wide: pd.DataFrame,
    top_n: int = 5,
    hold_bars: int = 6,
    cost_bps_per_side: float = 2.0,
    min_names: int | None = None,
    tod_windows: list[tuple[str, str]] | None = None,
    continuous: bool = False,
) -> BacktestResult:
    """
    Run the non-overlapping dollar-neutral top/bottom-N book.

    hold_bars is in 5-min bars (6 = 30 min). cost_bps_per_side in basis points/side.
    tod_windows optionally restricts trading to ET time windows (config 8, equity only).
    continuous=True ⇒ 24/7 crypto mode: uniform rebalance grid, holds span midnight/weekends
    (forward returns are NOT nulled at ET-day boundaries).
    """
    min_names = min_names or (2 * top_n)
    cols = [c for c in signal_wide.columns if c in close_wide.columns]
    sig = signal_wide[cols]
    fwd = forward_return_panel(close_wide[cols], hold_bars, entry_lag=1, intraday_only=not continuous)

    reb = _rebalance_points(sig.index, hold_bars, tod_windows, continuous=continuous)
    cost = 2.0 * (cost_bps_per_side * 1e-4)  # open + close the $1-gross book

    g_rows, long_rows, short_rows = {}, {}, {}
    for r in reb:
        s = sig.loc[r].dropna()
        f = fwd.loc[r] if r in fwd.index else None
        if f is None:
            continue
        names = s.index.intersection(f.dropna().index)
        if len(names) < min_names:
            continue
        s = s.loc[names].sort_values()
        longs = s.index[-top_n:]
        shorts = s.index[:top_n]
        long_rows[r] = float(f[longs].mean())     # forward return of the long names
        short_rows[r] = float(f[shorts].mean())    # forward return of the short names (P&L = -this)
        g_rows[r] = 0.5 * (long_rows[r] - short_rows[r])  # $1 gross book

    gross = pd.Series(g_rows, dtype=float).sort_index()
    net = (gross - cost).rename("net")
    gross = gross.rename("gross")

    if len(gross) < 2:
        empty = {"sharpe": np.nan, "sharpe_gross": np.nan, "max_drawdown": np.nan,
                 "sortino": np.nan, "ann_return": np.nan, "mean_gross_per_trade_bps": np.nan,
                 "mean_net_per_trade_bps": np.nan, "cost_per_trade_bps": float(cost * 1e4),
                 "n_trades": len(gross)}
        return BacktestResult(gross, net, empty, len(gross), 0.0, 0.0)

    long_leg = pd.Series(long_rows, dtype=float).sort_index()
    short_leg = pd.Series(short_rows, dtype=float).sort_index()

    span_days = max((gross.index[-1] - gross.index[0]).days, 1)
    years = span_days / 365.25
    ppy = len(gross) / years
    n_days = len(np.unique((_utc_date if continuous else _et_date)(gross.index)))
    trades_per_day = len(gross) / max(n_days, 1)

    metrics = {
        "sharpe": sharpe(net, periods_per_year=ppy),
        "sharpe_gross": sharpe(gross, periods_per_year=ppy),
        "sortino": sortino(net, periods_per_year=ppy),
        "max_drawdown": max_drawdown(net),
        "ann_return": float((1 + net.mean()) ** ppy - 1),
        "mean_gross_per_trade_bps": float(gross.mean() * 1e4),
        "mean_net_per_trade_bps": float(net.mean() * 1e4),
        "cost_per_trade_bps": float(cost * 1e4),
        "long_leg_bps_per_trade": float(long_leg.mean() * 1e4),    # you EARN this
        "short_leg_bps_per_trade": float(short_leg.mean() * 1e4),   # you earn MINUS this
        "hit_rate": float((net > 0).mean()),
        "avg_win_bps": float(net[net > 0].mean() * 1e4) if (net > 0).any() else 0.0,
        "avg_loss_bps": float(net[net < 0].mean() * 1e4) if (net < 0).any() else 0.0,
        "n_trades": int(len(net)),
    }
    return BacktestResult(gross, net, metrics, len(net), trades_per_day, ppy)
