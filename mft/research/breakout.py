"""
Opening-range breakout (ORB) signal — session-structured, past-only.

This is the CROSS-SECTIONAL reading of ORB: a signed breakout-strength score per name per
bar that the existing top/bottom-N harness ranks (long strongest up-breakouts, short
strongest down-breakouts), held non-overlapping. It is NOT the single-name event-ORB the
project already tested as a taker (T0052–T0054, taker-dead) — it asks the new question:
"is breakout direction a cross-sectionally rankable continuation signal at 5-min?"

Per session day (ET):
  - opening range = high/low (raw) or relative-to-SPY ratio (spy_adjusted) over the first
    `or_minutes`. Known only AFTER the OR window closes — the signal is NaN before then.
  - after the OR window, for each bar t (close known at t):
      up   if base[t] > OR_high * (1 + buffer)   -> signal = base[t]/OR_high - 1  (>0, long)
      down if base[t] < OR_low  * (1 - buffer)   -> signal = base[t]/OR_low  - 1  (<0, short)
      else NaN (no breakout -> not traded)
  - `volume_z` (same-time-of-day) optionally gates: signal kept only where vol_z > threshold.

No look-ahead: OR uses bars <= or_bars-1; the breakout test at t uses base[t] (known at t's
close); entry is next bar (enforced by the harness). Verified in tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.research.features import volume_zscore_tod


def _session_groups(df: pd.DataFrame):
    et_day = df.index.tz_convert("America/New_York").normalize().tz_localize(None)
    return df.groupby(et_day.values, sort=True)


def opening_range_signal(
    bars: pd.DataFrame,
    market_close: pd.Series | None,
    or_bars: int,
    buffer_bps: float = 0.0,
    spy_adjusted: bool = False,
    vol_z: pd.Series | None = None,
    vol_threshold: float | None = None,
) -> pd.Series:
    """
    Signed opening-range breakout strength for one symbol. `or_bars` = OR length in bars
    (5-min: 15m->3). `vol_z`/`vol_threshold` optionally require volume confirmation.
    """
    buf = buffer_bps * 1e-4
    out = pd.Series(np.nan, index=bars.index)

    if spy_adjusted:
        if market_close is None:
            raise ValueError("spy_adjusted requires market_close")
        mkt = market_close.reindex(bars.index).ffill()

    for _, g in _session_groups(bars):
        if len(g) < or_bars + 2:
            continue
        if spy_adjusted:
            stock_perf = g["close"] / g["open"].iloc[0]
            spy_perf = (mkt.loc[g.index] / mkt.loc[g.index].iloc[0])
            base = (stock_perf / spy_perf).to_numpy()
            or_high = np.nanmax(base[:or_bars])
            or_low = np.nanmin(base[:or_bars])
        else:
            base = g["close"].to_numpy()
            or_high = float(g["high"].iloc[:or_bars].max())
            or_low = float(g["low"].iloc[:or_bars].min())

        post = np.arange(or_bars, len(g))
        b = base[post]
        sig = np.full(len(post), np.nan)
        up = b > or_high * (1 + buf)
        dn = b < or_low * (1 - buf)
        sig[up] = b[up] / or_high - 1.0
        sig[dn] = b[dn] / or_low - 1.0
        out.loc[g.index[post]] = sig

    if vol_z is not None and vol_threshold is not None:
        out = out.where(vol_z.reindex(out.index) > vol_threshold)
    return out.rename("orb_signal")


def orb_signal_panel(
    bars_by_symbol: dict[str, pd.DataFrame],
    symbols: list[str],
    index: pd.DatetimeIndex,
    market_close: pd.Series | None,
    or_bars: int,
    buffer_bps: float = 0.0,
    spy_adjusted: bool = False,
    vol_threshold: float | None = None,
    vol_lookback_days: int = 20,
) -> pd.DataFrame:
    """Wide [time x symbol] ORB signal, with optional same-time-of-day volume gating."""
    cols = {}
    for s in symbols:
        df = bars_by_symbol[s]
        vz = volume_zscore_tod(df, vol_lookback_days) if vol_threshold is not None else None
        cols[s] = opening_range_signal(
            df, market_close, or_bars, buffer_bps, spy_adjusted, vz, vol_threshold)
    return pd.DataFrame(cols).reindex(index)
