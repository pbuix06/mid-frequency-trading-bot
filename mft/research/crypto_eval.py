"""
Shared crypto evaluation helpers — regime labels + per-trade attribution.

Used by the funding-backfill and momentum validation runners so they classify regimes and
attribute PnL with identical logic. Trend/vol/drawdown labels are causal; `btc_fwd` is forward
and used ONLY for conditional attribution (is the book just short/long beta), never to trade.
"""

from __future__ import annotations

import pandas as pd

from mft.research.targets import forward_return_panel
from mft.research.xs_backtest import _rebalance_points

DAY = 288  # 5-min bars/day


def regimes(close: pd.DataFrame, btc: str = "BTCUSDT") -> dict:
    """BTC trend (monthly), drawdown, vol terciles, and month label, on the bar grid."""
    b = close[btc]
    month = pd.PeriodIndex(b.index, freq="M")
    mret = b.groupby(month).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1.0)
    mlab = mret.apply(lambda r: "up" if r > 0.05 else ("down" if r < -0.05 else "sideways"))
    trend = pd.Series(month.map(mlab), index=b.index)
    dd = b / b.rolling(30 * DAY, min_periods=1).max() - 1.0
    ddlab = pd.cut(dd, [-1, -0.20, -0.10, 1], labels=["stress", "correction", "normal"])
    vol = b.pct_change().rolling(7 * DAY).std()
    vlab = pd.qcut(vol.rank(method="first"), 3, labels=["lowvol", "medvol", "highvol"])
    return {
        "trend": trend,
        "drawdown": pd.Series(ddlab, index=b.index).astype(object),
        "vol": pd.Series(vlab, index=b.index).astype(object),
        "month": pd.Series(month.astype(str), index=b.index),
    }


def attribute_trades(sig: pd.DataFrame, close: pd.DataFrame, cum: pd.DataFrame,
                     top_n: int, hold: int, btc: str = "BTCUSDT") -> pd.DataFrame:
    """
    Per-(rebalance, name) contribution rows for the continuous 24/7 dollar-neutral book.
    Columns: ts, sym, side, price, funding, total, fwd (the name's forward return), btc_fwd.
    """
    fwd = forward_return_panel(close, hold, entry_lag=1, intraday_only=False)
    carry = cum.shift(-(1 + hold)) - cum.shift(-1)
    btc_fwd = close[btc].shift(-(1 + hold)) / close[btc].shift(-1) - 1.0
    rows = []
    for r in _rebalance_points(sig.index, hold, continuous=True):
        s = sig.loc[r].dropna()
        f = fwd.loc[r] if r in fwd.index else None
        c = carry.loc[r] if r in carry.index else None
        if f is None or c is None:
            continue
        names = s.index.intersection(f.dropna().index).intersection(c.dropna().index)
        if len(names) < 2 * top_n:
            continue
        s = s.loc[names].sort_values()
        for x in s.index[-top_n:]:    # longs (highest signal)
            rows.append((r, x, "long", 0.5 * f[x] / top_n, 0.5 * (-c[x]) / top_n, float(f[x]), btc_fwd.loc[r]))
        for x in s.index[:top_n]:     # shorts (lowest signal)
            rows.append((r, x, "short", 0.5 * (-f[x]) / top_n, 0.5 * (c[x]) / top_n, float(f[x]), btc_fwd.loc[r]))
    df = pd.DataFrame(rows, columns=["ts", "sym", "side", "price", "funding", "fwd", "btc_fwd"])
    df["total"] = df["price"] + df["funding"]
    return df
