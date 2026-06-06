"""
Forward-return targets — what a signal is graded against.

Honest by construction:
  - ENTRY LAG: a signal at the close of bar t can only be acted on at t+1. So the
    forward return is measured from the close of bar t+entry_lag to t+entry_lag+horizon,
    NOT from t. This is the difference between a real edge and a fantasy one.
  - INTRADAY ONLY: returns that would span overnight are set NaN (we go flat at the
    close). This prevents counting un-tradeable overnight gaps as "alpha".

`forward_return` is used both to grade signals (IC, in signal_lab) and to price the
cross-sectional book (xs_backtest), so the two can never silently disagree.
"""

from __future__ import annotations

import pandas as pd

from mft.research.features import rolling_beta


def _et_date(index: pd.DatetimeIndex) -> pd.Series:
    et = index.tz_convert("America/New_York").normalize().tz_localize(None)
    return pd.Series(et, index=index)


def forward_return(
    close: pd.Series,
    horizon_bars: int,
    entry_lag: int = 1,
    intraday_only: bool = True,
) -> pd.Series:
    """
    Tradeable forward return: enter at close of bar t+entry_lag, exit `horizon_bars`
    later. Returns NaN where the holding period crosses a session boundary.
    """
    entry = close.shift(-entry_lag)
    exit_ = close.shift(-(entry_lag + horizon_bars))
    fwd = exit_ / entry - 1.0

    if intraday_only:
        # The DECISION bar (t), entry (t+lag) and exit (t+lag+h) must all be in the same
        # session — otherwise a decision at the day's last bar leaks into a next-day trade.
        d = _et_date(close.index)
        exit_day = d.shift(-(entry_lag + horizon_bars))
        same_session = (d.values == exit_day.values)
        fwd = fwd.where(same_session)
    return fwd.rename(f"fwd_{horizon_bars}b")


def beta_adjusted_forward_return(
    close: pd.Series,
    market_close: pd.Series,
    horizon_bars: int,
    entry_lag: int = 1,
    beta_window: int = 78,
    intraday_only: bool = True,
) -> pd.Series:
    """
    Market-neutral forward return = fwd_stock - beta * fwd_market, with beta estimated
    PAST-ONLY (so removing market exposure introduces no look-ahead). This isolates
    the part of the forward move that is NOT just beta to SPY.
    """
    fwd_s = forward_return(close, horizon_bars, entry_lag, intraday_only)
    mkt = market_close.reindex(close.index).ffill()
    fwd_m = forward_return(mkt, horizon_bars, entry_lag, intraday_only)
    beta = rolling_beta(close, market_close, beta_window, skip=1).reindex(close.index)
    out = fwd_s - beta * fwd_m
    return out.rename(f"fwd_madj_{horizon_bars}b")


def forward_return_panel(close_wide: pd.DataFrame, horizon_bars: int,
                         entry_lag: int = 1, intraday_only: bool = True) -> pd.DataFrame:
    """Apply forward_return to every column of a wide [time x symbol] close frame."""
    return pd.DataFrame(
        {s: forward_return(close_wide[s].dropna(), horizon_bars, entry_lag, intraday_only)
         for s in close_wide.columns}
    ).reindex(close_wide.index)


def beta_adjusted_forward_panel(close_wide: pd.DataFrame, market_close: pd.Series,
                                horizon_bars: int, entry_lag: int = 1,
                                beta_window: int = 78, intraday_only: bool = True) -> pd.DataFrame:
    """Beta-adjusted (residual) forward returns for every column — the clean alpha target."""
    return pd.DataFrame(
        {s: beta_adjusted_forward_return(close_wide[s].dropna(), market_close, horizon_bars,
                                         entry_lag, beta_window, intraday_only)
         for s in close_wide.columns}
    ).reindex(close_wide.index)
