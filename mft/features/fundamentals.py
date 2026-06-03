"""
Fundamental factor panels from EDGAR + prices — PIT-aligned.

Builds dates × tickers signal panels for cross-sectional fundamental factors.
Point-in-time discipline: each fundamental series is indexed by its SEC `filed`
date and forward-filled onto the trading calendar, so the value on day t is the
most recent figure KNOWABLE by day t. Market cap uses that day's close × the
last-known shares outstanding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mft.data_layer.edgar_ingest import load_fundamentals


def _pit_item_series(fund: pd.DataFrame, item: str, index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill one fundamental item (by filed date) onto `index`."""
    sub = fund[fund["item"] == item]
    if sub.empty:
        return pd.Series(np.nan, index=index)
    s = pd.Series(sub["value"].to_numpy(float), index=pd.DatetimeIndex(sub["filed"]))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    # Reindex onto the union, ffill, then restrict to the trading index.
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def build_bm_panel(
    close_df: pd.DataFrame,
    edgar_dir: Path,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Book-to-market panel (high = cheap = value-long).

    B/M[t, ticker] = book_equity_knowable(t) / (close(t) × shares_knowable(t))

    Only includes tickers that have EDGAR fundamentals on disk. Returns a
    DataFrame aligned to close_df.index; NaN where book equity or shares are not
    yet known (pre-first-filing) or non-positive (book equity <= 0 dropped — a
    negative-equity firm has no meaningful B/M).
    """
    cols = tickers if tickers is not None else list(close_df.columns)
    out = {}
    for t in cols:
        if t not in close_df.columns:
            continue
        try:
            fund = load_fundamentals(t, edgar_dir)
        except FileNotFoundError:
            continue
        be = _pit_item_series(fund, "book_equity", close_df.index)
        sh = _pit_item_series(fund, "shares", close_df.index)
        mktcap = close_df[t] * sh
        bm = be / mktcap
        # Drop non-positive book equity / cap (meaningless B/M)
        bm = bm.where((be > 0) & (mktcap > 0))
        out[t] = bm
    if not out:
        return pd.DataFrame(index=close_df.index)
    return pd.DataFrame(out, index=close_df.index)
