"""
Rolling 1-minute feature snapshots per symbol — the live mirror of mft.research.features.

Every minute, for each symbol, we recompute a `FeatureSnapshot` from a rolling deque of
closed bars. THE RULE is the same as the research engine: every value at bar t uses ONLY
bars with timestamp <= t (the just-closed bar is the latest input; nothing from t+1 enters).
The 24h high/low used for breakout detection deliberately EXCLUDES the current bar, so a
"breakout" is `close_t > max(high over the prior 24h)` — a real new extreme, not a tie.

This module computes features only. It does not trade, claim edge, or place orders.
Windows are configurable so tests can run with tiny histories instead of a full 1440-bar day.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mft.live_monitor.bar_builder import Bar


@dataclass
class FeatureConfig:
    ret_1h_bars: int = 60
    ret_4h_bars: int = 240
    ret_24h_bars: int = 1440
    vol_short_bars: int = 240          # 4h realized vol
    vol_long_bars: int = 1440          # 24h realized vol
    high_low_bars: int = 1440          # 24h high/low window (PRIOR bars, excl. current)
    volz_bars: int = 240               # volume z-score baseline window
    min_history_bars: int = 1441       # full 24h + current bar before "warm"


@dataclass
class FeatureSnapshot:
    ts: pd.Timestamp
    symbol: str
    close: float
    return_1h: float
    return_4h: float
    return_24h: float
    vol_4h: float
    vol_24h: float
    high_24h: float                    # max high over PRIOR 24h (excl. current bar)
    low_24h: float                     # min low over PRIOR 24h (excl. current bar)
    dist_from_high: float              # close / high_24h - 1  (>0 ⇒ above prior high)
    dist_from_low: float               # close / low_24h - 1   (<0 ⇒ below prior low)
    volume_zscore: float
    funding_rate: float | None
    spread_bps: float | None
    n_bars: int
    warm: bool                         # enough history for the breakout rule


class FeatureStream:
    """Maintains a rolling bar window per symbol and computes snapshots on demand."""

    def __init__(self, cfg: FeatureConfig | None = None):
        self.cfg = cfg or FeatureConfig()
        maxlen = max(self.cfg.ret_24h_bars, self.cfg.high_low_bars,
                     self.cfg.vol_long_bars, self.cfg.volz_bars) + 5
        self._bars: dict[str, deque[Bar]] = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, symbol: str, bar: Bar) -> None:
        self._bars[symbol].append(bar)

    def history(self, symbol: str) -> list[Bar]:
        return list(self._bars.get(symbol, ()))

    def last_bar(self, symbol: str) -> Bar | None:
        b = self._bars.get(symbol)
        return b[-1] if b else None

    def snapshot(self, symbol: str, funding_rate: float | None = None) -> FeatureSnapshot | None:
        bars = self._bars.get(symbol)
        if not bars:
            return None
        cfg = self.cfg
        closes = np.array([b.close for b in bars], dtype=float)
        highs = np.array([b.high for b in bars], dtype=float)
        lows = np.array([b.low for b in bars], dtype=float)
        vols = np.array([b.volume for b in bars], dtype=float)
        n = len(closes)
        cur = bars[-1]

        def ret(k: int) -> float:
            return float(closes[-1] / closes[-1 - k] - 1.0) if n > k and closes[-1 - k] > 0 else float("nan")

        rets = np.diff(closes) / closes[:-1] if n > 1 else np.array([])

        def vol(w: int) -> float:
            return float(np.std(rets[-w:], ddof=1)) if len(rets) >= max(w, 2) else float("nan")

        # PRIOR 24h high/low — exclude the current bar so a breakout is a genuine new extreme
        if n >= 2:
            lo = max(0, n - 1 - cfg.high_low_bars)
            prev_high = float(highs[lo:n - 1].max())
            prev_low = float(lows[lo:n - 1].min())
        else:
            prev_high = prev_low = float("nan")

        # volume z-score: current bar vs prior `volz_bars` (excl. current)
        if n >= 3:
            lo = max(0, n - 1 - cfg.volz_bars)
            base = vols[lo:n - 1]
            mu = float(base.mean())
            sd = float(base.std(ddof=1)) if len(base) > 1 else 0.0
            volz = float((vols[-1] - mu) / sd) if sd > 0 else float("nan")
        else:
            volz = float("nan")

        dist_high = float(closes[-1] / prev_high - 1.0) if prev_high == prev_high and prev_high > 0 else float("nan")
        dist_low = float(closes[-1] / prev_low - 1.0) if prev_low == prev_low and prev_low > 0 else float("nan")

        return FeatureSnapshot(
            ts=cur.ts, symbol=symbol, close=float(closes[-1]),
            return_1h=ret(cfg.ret_1h_bars), return_4h=ret(cfg.ret_4h_bars),
            return_24h=ret(cfg.ret_24h_bars),
            vol_4h=vol(cfg.vol_short_bars), vol_24h=vol(cfg.vol_long_bars),
            high_24h=prev_high, low_24h=prev_low,
            dist_from_high=dist_high, dist_from_low=dist_low,
            volume_zscore=volz, funding_rate=funding_rate, spread_bps=cur.spread_bps,
            n_bars=n, warm=(n >= cfg.min_history_bars),
        )


@dataclass
class RegimeConfig:
    """Coarse, causal LIVE BTC-regime thresholds on the trailing 24h return.

    NOTE: this is a cheap real-time proxy, NOT the research monthly-trend regime in
    mft.research.crypto_eval.regimes. It exists only to gate the smoke-test breakout
    trigger ('don't buy a breakout while BTC is dumping').
    """
    bull_threshold: float = 0.03       # 24h BTC return >= +3% ⇒ bullish
    bear_threshold: float = -0.03      # 24h BTC return <= -3% ⇒ bearish


def btc_regime(btc_snap: FeatureSnapshot | None, cfg: RegimeConfig | None = None) -> str:
    """'bullish' | 'bearish' | 'neutral' | 'unknown' from BTC's trailing 24h return."""
    cfg = cfg or RegimeConfig()
    if btc_snap is None or btc_snap.return_24h != btc_snap.return_24h:
        return "unknown"
    r = btc_snap.return_24h
    if r >= cfg.bull_threshold:
        return "bullish"
    if r <= cfg.bear_threshold:
        return "bearish"
    return "neutral"
