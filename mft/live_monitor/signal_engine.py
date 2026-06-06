"""
Signal engine — ONE simple, pre-declared paper-only trigger (a monitoring smoke test).

>>> THIS IS NOT AN ALPHA CLAIM. <<<
The project's research verdict stands: no production-ready edge was found on free/retail
data. This 24h-breakout trigger exists ONLY to exercise the live-monitor plumbing end to
end (stream -> bars -> features -> signal -> risk -> paper ledger -> report). Its expected
forward edge is ZERO; a profitable-looking paper run is treated as a SUSPECTED FALSE
POSITIVE by the reporter, never as a discovery. It is deliberately not in the strategy
registry (which holds rejected research candidates).

Rule:
  long  if close breaks the PRIOR 24h high  AND volume z-score confirms AND BTC not bearish
  short if close breaks the PRIOR 24h low   AND volume z-score confirms AND BTC not bullish
  flat  otherwise — with the exact blocking reason recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

from mft.live_monitor.feature_stream import FeatureSnapshot


@dataclass
class SignalConfig:
    min_volume_z: float = 1.0          # volume-confirmation threshold (z-score)
    breakout_buffer_bps: float = 0.0   # require breaking the prior extreme by this buffer
    require_warm: bool = True          # demand a full 24h of history first


@dataclass
class Signal:
    ts: object
    symbol: str
    side: str                          # 'long' | 'short' | 'flat'
    reason: str
    close: float
    prev_high_24h: float
    prev_low_24h: float
    volume_z: float
    btc_regime: str

    @property
    def is_trade(self) -> bool:
        return self.side in ("long", "short")


class BreakoutSignalEngine:
    def __init__(self, cfg: SignalConfig | None = None):
        self.cfg = cfg or SignalConfig()

    def evaluate(self, snap: FeatureSnapshot, btc_regime: str) -> Signal:
        cfg = self.cfg

        def mk(side: str, reason: str) -> Signal:
            return Signal(ts=snap.ts, symbol=snap.symbol, side=side, reason=reason,
                          close=snap.close, prev_high_24h=snap.high_24h,
                          prev_low_24h=snap.low_24h, volume_z=snap.volume_zscore,
                          btc_regime=btc_regime)

        if cfg.require_warm and not snap.warm:
            return mk("flat", f"warmup: only {snap.n_bars} bars of history (need a full 24h)")

        buf = cfg.breakout_buffer_bps * 1e-4
        has_high = snap.high_24h == snap.high_24h
        has_low = snap.low_24h == snap.low_24h
        broke_high = has_high and snap.close > snap.high_24h * (1.0 + buf)
        broke_low = has_low and snap.close < snap.low_24h * (1.0 - buf)
        vol_ok = snap.volume_zscore == snap.volume_zscore and snap.volume_zscore >= cfg.min_volume_z

        if broke_high:
            if not vol_ok:
                return mk("flat", f"up-breakout but volume z {snap.volume_zscore:.2f} < {cfg.min_volume_z}")
            if btc_regime == "bearish":
                return mk("flat", "up-breakout blocked: BTC regime bearish")
            return mk("long", "24h-high breakout + volume confirm + BTC not bearish")

        if broke_low:
            if not vol_ok:
                return mk("flat", f"down-breakout but volume z {snap.volume_zscore:.2f} < {cfg.min_volume_z}")
            if btc_regime == "bullish":
                return mk("flat", "down-breakout blocked: BTC regime bullish")
            return mk("short", "24h-low breakout + volume confirm + BTC not bullish")

        return mk("flat", "no 24h breakout")
