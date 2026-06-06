"""
Intraday Opening-Range Breakout (ORB) — the frozen mid-frequency candidate.

Economic rationale (one sentence): high-volatility, high-attention U.S. equities
that break their opening range tend to continue in the breakout direction during
the same session, because early order imbalance, attention/flow, stop-triggering,
and volatility-chasing create short-horizon intraday trend.

This is the PRODUCTION, AlphaBase-compatible implementation of the candidate
discovered in RESEARCH_LOG.md §11 and specified in
docs/INTRADAY_ORB_EXPERIMENT_SPEC.md. The research session backtest
(mft/backtest/intraday_session.py) is a fast approximation; THIS is the one code
path that runs in backtest, paper, and live. No strategy logic may live only in a
script.

Per-minute decision (uses only bars up to the current one — PIT, no look-ahead):
  - flat until the opening range (first `or_minutes` session bars) has formed;
  - on the first post-OR breakout, take its direction (+1 above OR high, -1 below
    OR low) and HOLD to the close;
  - forced flat in the final minute(s) of the session -> guaranteed EOD
    liquidation, never any overnight exposure.

Entry timing is the harness's next-bar convention (signal on a CLOSED bar, applied
on the NEXT bar) — deliberately more honest than the research approximation's
optimistic "fill at the breakout level." See the spec §5.
"""

from __future__ import annotations

from datetime import time

import pandas as pd

from mft.alphas.base import AlphaBase


def _parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


class IntradayORB(AlphaBase):
    """
    Single-name intraday opening-range breakout, flat by close.

    Parameters:
        symbol:        ticker this sleeve trades.
        or_minutes:    opening-range length in BARS/minutes (default 30). The
                       spec's primary; 15/60 are the plateau check only.
        session_open:  regular-session open, ET "HH:MM" (default "09:30").
        session_close: regular-session close, ET "HH:MM" (default "16:00").
        tz:            session timezone (default America/New_York). The data
                       index is UTC (AlphaBase contract); we convert per call.
        flatten_before_close_min:
                       force flat this many minutes before the close (default 1)
                       so the position is liquidated into the close and NEVER
                       held overnight. The execution layer fills the flatten at
                       the close (MOC); the signal's job is to express the intent.
    """

    def __init__(
        self,
        symbol: str,
        or_minutes: int = 30,
        session_open: str = "09:30",
        session_close: str = "16:00",
        tz: str = "America/New_York",
        flatten_before_close_min: int = 1,
    ):
        self.symbol = symbol
        self.or_minutes = or_minutes
        self.tz = tz
        self._open_t = _parse_hhmm(session_open)
        self._close_t = _parse_hhmm(session_close)
        self.flatten_before_close_min = flatten_before_close_min
        # A full session of bars, so the window always spans the current open.
        self._session_minutes = _minutes(self._close_t) - _minutes(self._open_t)

    @property
    def lookback(self) -> int:
        # Enough history that the current session's open is always in the window.
        return self._session_minutes

    def compute_signal(self, window: pd.DataFrame) -> dict[str, float]:
        flat = {self.symbol: 0.0}
        if len(window) == 0:
            return flat

        idx_et = window.index.tz_convert(self.tz)
        last_t = idx_et[-1].time()

        # Outside the regular session -> flat (also guarantees no overnight hold).
        if last_t < self._open_t or last_t >= self._close_t:
            return flat

        # Force flat into the close: liquidate, never carry overnight.
        if _minutes(self._close_t) - _minutes(last_t) <= self.flatten_before_close_min:
            return flat

        # Restrict to the CURRENT session's bars (same ET date, within session).
        day = idx_et[-1].date()
        in_session = (
            (idx_et.date == day)
            & (idx_et.time >= self._open_t)
            & (idx_et.time < self._close_t)
        )
        g = window[in_session]

        # Opening range must be formed AND at least one post-OR bar must exist.
        if len(g) <= self.or_minutes:
            return flat

        # Opening range from the first `or_minutes` session bars (positional,
        # matching the research feature builder so the two paths agree).
        or_block = g.iloc[: self.or_minutes]
        or_high = float(or_block["high"].max())
        or_low = float(or_block["low"].min())

        # First post-OR breakout, up to and including the current bar. Once a
        # session breaks, the direction is held for the rest of the session.
        post = g.iloc[self.or_minutes :]
        up = post.index[post["high"] >= or_high]
        dn = post.index[post["low"] <= or_low]
        first_up = up[0] if len(up) else None
        first_dn = dn[0] if len(dn) else None

        if first_up is not None and (first_dn is None or first_up <= first_dn):
            return {self.symbol: 1.0}     # broke above OR high -> long the trend
        if first_dn is not None:
            return {self.symbol: -1.0}    # broke below OR low  -> short the trend
        return flat

    def target_series(self, bars: pd.DataFrame) -> pd.Series:
        """
        Vectorized per-bar target over a full multi-session history.

        Equivalent to calling `compute_signal` on every prefix (asserted exactly by
        tests/test_intraday_orb_parity.py), but O(n) instead of O(n^2). The per-bar
        `compute_signal` is the CANONICAL live path; this is the research-speed twin.
        Returns the closed-bar target at each bar; the execution layer applies the
        usual next-bar fill.
        """
        bars = bars.sort_index()
        out = pd.Series(0.0, index=bars.index)
        if len(bars) == 0:
            return out

        et = bars.index.tz_convert(self.tz)
        in_sess = (et.time >= self._open_t) & (et.time < self._close_t)
        close_min = _minutes(self._close_t)
        mins_to_close = pd.Series(
            close_min - (et.hour * 60 + et.minute), index=bars.index
        )
        dates = pd.Series(et.date, index=bars.index)

        for _, idx in bars.index[in_sess].to_series().groupby(dates[in_sess]):
            g = bars.loc[idx]
            if len(g) <= self.or_minutes:
                continue
            or_block = g.iloc[: self.or_minutes]
            or_high = float(or_block["high"].max())
            or_low = float(or_block["low"].min())
            post = g.iloc[self.or_minutes :]
            up = post.index[post["high"] >= or_high]
            dn = post.index[post["low"] <= or_low]
            first_up = up[0] if len(up) else None
            first_dn = dn[0] if len(dn) else None
            if first_up is not None and (first_dn is None or first_up <= first_dn):
                direction, brk_ts = 1.0, first_up
            elif first_dn is not None:
                direction, brk_ts = -1.0, first_dn
            else:
                continue
            held = (
                (g.index >= brk_ts)
                & (mins_to_close.loc[g.index] > self.flatten_before_close_min)
            )
            out.loc[g.index[held]] = direction
        return out

    def __repr__(self) -> str:
        return (
            f"IntradayORB(symbol={self.symbol!r}, or_minutes={self.or_minutes}, "
            f"flatten_before_close_min={self.flatten_before_close_min})"
        )
