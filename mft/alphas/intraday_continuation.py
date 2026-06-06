"""
Intraday Afternoon Continuation — momentum at the SESSION scale, not the breakout
instant.

Economic rationale (one sentence): a stock with a strong directional move in the
morning tends to continue in that direction into the close, because attention,
institutional VWAP execution, and intraday momentum carry the move through the
afternoon.

Why this, after ORB failed (RESEARCH_LOG §11e): ORB's whole edge sat in the ~5 bp
breakout instant, so any entry slippage erased it. This sleeve's edge is the
AFTERNOON DRIFT — distributed over hours — and it enters at a SCHEDULED time
(midday) at the prevailing price, with no breakout instant to miss. So it should be
execution-ROBUST exactly where ORB was fragile. That is the hypothesis under test.

Per-minute decision (PIT, no look-ahead):
  - flat all morning (before the decision time);
  - at the decision time, sign the morning return (close-at-decision / session-open
    - 1); if |morning return| > threshold, take that direction and HOLD to close;
  - forced flat in the final minute(s) -> EOD liquidation, never overnight.

Entry is the harness's next-bar fill at a scheduled time — deliberately realistic.
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


class IntradayAfternoonContinuation(AlphaBase):
    """
    Single-name intraday momentum continuation, flat by close.

    Parameters:
        symbol:        ticker this sleeve trades.
        decision_time: ET "HH:MM" at which the morning return is signed and the
                       position is taken (default "12:00", the session midpoint).
        threshold:     only trade when |morning return| > threshold (default 0.0).
        min_morning_bars:
                       require at least this many morning bars for a valid signal
                       (default 30) — a few-minute morning is not a "trend".
        session_open / session_close / tz: regular-session definition.
        flatten_before_close_min:
                       force flat this many minutes before the close (default 1) so
                       the position is liquidated into the close, never overnight.
    """

    def __init__(
        self,
        symbol: str,
        decision_time: str = "12:00",
        threshold: float = 0.0,
        min_morning_bars: int = 30,
        session_open: str = "09:30",
        session_close: str = "16:00",
        tz: str = "America/New_York",
        flatten_before_close_min: int = 1,
    ):
        self.symbol = symbol
        self.threshold = threshold
        self.min_morning_bars = min_morning_bars
        self.tz = tz
        self._open_t = _parse_hhmm(session_open)
        self._decision_t = _parse_hhmm(decision_time)
        self._close_t = _parse_hhmm(session_close)
        self.flatten_before_close_min = flatten_before_close_min
        self._session_minutes = _minutes(self._close_t) - _minutes(self._open_t)

    @property
    def lookback(self) -> int:
        return self._session_minutes  # window must span the morning of the session

    def _session_direction(self, g: pd.DataFrame) -> float:
        """Signed direction for a session's afternoon, from its morning return."""
        g_et_time = g.index.tz_convert(self.tz).time
        morning = g[g_et_time < self._decision_t]
        if len(morning) < self.min_morning_bars:
            return 0.0
        sess_open = float(g["open"].iloc[0])
        morning_px = float(morning["close"].iloc[-1])
        if sess_open <= 0:
            return 0.0
        morning_ret = morning_px / sess_open - 1.0
        if abs(morning_ret) <= self.threshold:
            return 0.0
        return 1.0 if morning_ret > 0 else -1.0

    def compute_signal(self, window: pd.DataFrame) -> dict[str, float]:
        flat = {self.symbol: 0.0}
        if len(window) == 0:
            return flat

        idx_et = window.index.tz_convert(self.tz)
        last_t = idx_et[-1].time()

        # Flat in the morning, after the close, and into the close (EOD liquidation).
        if last_t < self._decision_t or last_t >= self._close_t:
            return flat
        if _minutes(self._close_t) - _minutes(last_t) <= self.flatten_before_close_min:
            return flat

        day = idx_et[-1].date()
        in_session = (
            (idx_et.date == day)
            & (idx_et.time >= self._open_t)
            & (idx_et.time < self._close_t)
        )
        g = window[in_session]
        if len(g) == 0:
            return flat
        return {self.symbol: self._session_direction(g)}

    def target_series(self, bars: pd.DataFrame) -> pd.Series:
        """Vectorized per-bar target (research-speed twin of compute_signal; proven
        equal by tests/test_intraday_continuation.py)."""
        bars = bars.sort_index()
        out = pd.Series(0.0, index=bars.index)
        if len(bars) == 0:
            return out

        et = bars.index.tz_convert(self.tz)
        in_sess = (et.time >= self._open_t) & (et.time < self._close_t)
        mins_to_close = pd.Series(
            _minutes(self._close_t) - (et.hour * 60 + et.minute), index=bars.index
        )
        dates = pd.Series(et.date, index=bars.index)

        for _, idx in bars.index[in_sess].to_series().groupby(dates[in_sess]):
            g = bars.loc[idx]
            direction = self._session_direction(g)
            if direction == 0.0:
                continue
            g_et_time = g.index.tz_convert(self.tz).time
            afternoon = g_et_time >= self._decision_t
            not_flat = mins_to_close.loc[g.index].to_numpy() > self.flatten_before_close_min
            held = afternoon & not_flat
            out.loc[g.index[held]] = direction
        return out

    def __repr__(self) -> str:
        return (
            f"IntradayAfternoonContinuation(symbol={self.symbol!r}, "
            f"decision_time={self._decision_t.strftime('%H:%M')}, threshold={self.threshold})"
        )
