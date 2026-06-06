"""
Cross-sectional long-short backtest WITH funding-carry attribution (crypto perp, 24/7).

Same non-overlapping continuous machinery as xs_backtest, but each trade's P&L is split:
  - PRICE PnL  : the perp price move (entry next bar -> exit at horizon), dollar-neutral book.
  - FUNDING PnL: the funding settled DURING the hold. A short receives positive funding; a long
                 pays it. So shorting crowded-longs (high +funding) EARNS carry — aligned with the
                 reversion. carry over hold = cum_funding[exit] - cum_funding[entry], per name.
  - TOTAL = price + funding ; NET = total - taker cost (funding is NOT a transaction cost).

Book convention matches xs_backtest: $1 gross (0.5 long / 0.5 short), so price gross here equals
cross_sectional_ls's gross on the same names. Carry uses cumulative funding mapped to the bar grid
(crypto_panel.map_to_bars), which is known-only (no future funding).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mft.research.targets import forward_return_panel
from mft.research.xs_backtest import _rebalance_points, _utc_date
from mft.validation.metrics import max_drawdown, sharpe


@dataclass
class FundingResult:
    price: pd.Series
    funding: pd.Series
    total: pd.Series
    net: pd.Series
    metrics: dict
    n_trades: int
    trades_per_day: float

    def equity(self) -> pd.Series:
        return (1.0 + self.net).cumprod()


def funding_ls_backtest(signal_wide: pd.DataFrame, close_wide: pd.DataFrame,
                        cum_funding_wide: pd.DataFrame, top_n: int = 2, hold_bars: int = 96,
                        cost_bps_per_side: float = 5.0) -> FundingResult:
    """
    Continuous 24/7 dollar-neutral top/bottom-N book with price + funding attribution.
    Signal: HIGH = long (e.g. -funding_z so crowded-shorts/neg-funding are long).
    hold_bars in 5-min bars (96 = 8h). Entry next bar (no funding look-ahead in the signal).
    """
    cols = [c for c in signal_wide.columns if c in close_wide.columns and c in cum_funding_wide.columns]
    sig = signal_wide[cols]
    fwd = forward_return_panel(close_wide[cols], hold_bars, entry_lag=1, intraday_only=False)
    cf = cum_funding_wide[cols]
    carry = cf.shift(-(1 + hold_bars)) - cf.shift(-1)      # funding settled during the hold, per name

    reb = _rebalance_points(sig.index, hold_bars, continuous=True)
    cost = 2.0 * (cost_bps_per_side * 1e-4)
    min_names = 2 * top_n

    p_rows, f_rows, long_rows, short_rows = {}, {}, {}, {}
    for r in reb:
        s = sig.loc[r].dropna()
        fr = fwd.loc[r] if r in fwd.index else None
        cr = carry.loc[r] if r in carry.index else None
        if fr is None or cr is None:
            continue
        names = s.index.intersection(fr.dropna().index).intersection(cr.dropna().index)
        if len(names) < min_names:
            continue
        s = s.loc[names].sort_values()
        longs, shorts = s.index[-top_n:], s.index[:top_n]
        pl, ps = float(fr[longs].mean()), float(fr[shorts].mean())
        cl, cs = float(cr[longs].mean()), float(cr[shorts].mean())
        p_rows[r] = 0.5 * (pl - ps)                 # price book ($1 gross)
        f_rows[r] = 0.5 * (cs - cl)                 # short RECEIVES funding, long PAYS
        long_rows[r], short_rows[r] = pl, ps

    price = pd.Series(p_rows, dtype=float).sort_index().rename("price")
    funding = pd.Series(f_rows, dtype=float).sort_index().rename("funding")
    if len(price) < 2:
        empty = {k: np.nan for k in ("sharpe", "max_drawdown", "price_bps_per_trade",
                                     "funding_bps_per_trade", "total_gross_bps_per_trade",
                                     "net_bps_per_trade")}
        empty["n_trades"] = len(price)
        z = pd.Series(dtype=float)
        return FundingResult(price, funding, z, z, empty, len(price), 0.0)

    total = (price + funding).rename("total")
    net = (total - cost).rename("net")
    long_leg = pd.Series(long_rows).sort_index()
    short_leg = pd.Series(short_rows).sort_index()

    span_days = max((price.index[-1] - price.index[0]).days, 1)
    ppy = len(price) / (span_days / 365.25)
    n_days = len(np.unique(_utc_date(price.index)))

    metrics = {
        "sharpe": sharpe(net, periods_per_year=ppy),
        "sharpe_total_gross": sharpe(total, periods_per_year=ppy),
        "max_drawdown": max_drawdown(net),
        "price_bps_per_trade": float(price.mean() * 1e4),
        "funding_bps_per_trade": float(funding.mean() * 1e4),
        "total_gross_bps_per_trade": float(total.mean() * 1e4),
        "net_bps_per_trade": float(net.mean() * 1e4),
        "cost_per_trade_bps": float(cost * 1e4),
        "long_leg_bps": float(long_leg.mean() * 1e4),
        "short_leg_bps": float(short_leg.mean() * 1e4),
        "win_rate": float((net > 0).mean()),
        "n_trades": int(len(net)),
    }
    return FundingResult(price, funding, total, net, metrics, len(net), len(net) / max(n_days, 1))
