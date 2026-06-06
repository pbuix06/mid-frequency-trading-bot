"""
Panel loader: 1-minute bars -> resampled, regular-session, aligned multi-symbol panel.

Why per-symbol dicts (not one giant wide frame): rolling/time-of-day features are
far cleaner and less leak-prone as per-symbol time-series ops. We only pivot to a
wide [timestamp x symbol] frame at the cross-sectional (IC / backtest) step.

Regular trading hours only (09:30–16:00 ET): pre/post-market minute bars on the free
IEX feed are thin and unrepresentative, and intraday strategies here go flat overnight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mft.data_layer.alpaca_ingest import load_intraday

ROOT = Path(__file__).parents[2]
INTRADAY_DIR = ROOT / "data" / "intraday"

# minutes per regular session bar-grid; 09:30..15:55 inclusive at 5min = 78 bars/day
RTH_START = pd.Timestamp("09:30").time()
RTH_END = pd.Timestamp("16:00").time()


def _to_et(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_convert("America/New_York")


def regular_session(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 09:30 <= ET time < 16:00 bars (Alpaca bar ts = bar START)."""
    et = _to_et(minute_df.index)
    mask = (et.time >= RTH_START) & (et.time < RTH_END)
    return minute_df[mask]


def resample_bars(minute_df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """
    Resample 1-minute OHLCV(+trades,+vwap) to `freq`, regular session only.

    Aggregation: open=first, high=max, low=min, close=last, volume=sum,
    trades=sum, vwap=volume-weighted mean of the minute vwaps. Empty (overnight)
    buckets are dropped. Buckets are left-labelled (a 09:30 bar covers 09:30–09:35),
    so the bar timestamp is the time at which the bar OPENS — its data is only
    complete (and tradeable info known) at the NEXT bar. The entry-lag discipline
    in targets.py/xs_backtest.py accounts for this.
    """
    rth = regular_session(minute_df)
    if rth.empty:
        return rth.iloc[0:0]

    has_vwap = "vwap" in rth.columns and rth["vwap"].notna().any()
    if has_vwap:
        rth = rth.assign(_pv=rth["vwap"].astype(float) * rth["volume"].astype(float))

    agg = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }
    if "trades" in rth.columns:
        agg["trades"] = "sum"
    if has_vwap:
        agg["_pv"] = "sum"

    out = rth.resample(freq, label="left", closed="left").agg(agg)
    out = out[out["volume"] > 0]  # drop overnight / empty buckets

    if has_vwap:
        out["vwap"] = out["_pv"] / out["volume"].replace(0, pd.NA)
        out = out.drop(columns="_pv")
    return out


@dataclass
class Panel:
    """Aligned multi-symbol intraday panel of resampled bars."""

    bars: dict[str, pd.DataFrame]          # symbol -> resampled OHLCV frame
    freq: str
    market: str = "SPY"                     # market proxy symbol (if present)
    symbols: list[str] = field(default_factory=list)

    def to_wide(self, column: str, symbols: list[str] | None = None) -> pd.DataFrame:
        """Pivot one column into a [timestamp x symbol] frame (union index, NaN-filled)."""
        syms = symbols or self.symbols
        cols = {s: self.bars[s][column] for s in syms if s in self.bars}
        return pd.DataFrame(cols).sort_index()

    def market_close(self) -> pd.Series:
        if self.market not in self.bars:
            raise KeyError(f"market proxy {self.market!r} not in panel")
        return self.bars[self.market]["close"]


def load_panel(
    symbols: list[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    freq: str = "5min",
    data_dir: Path = INTRADAY_DIR,
    market: str = "SPY",
) -> Panel:
    """
    Load + resample + window a set of symbols into an aligned Panel.

    `start`/`end` are inclusive ISO dates (UTC). The market proxy is always loaded
    (added to `symbols` if missing) so market-relative features are available.
    NOTE: pass research windows only (<= intraday lock-box) — see splits.py.
    """
    syms = list(dict.fromkeys([*symbols, market]))  # de-dup, keep order, ensure market
    start_ts = pd.Timestamp(start, tz="UTC") if start is not None else None
    end_ts = pd.Timestamp(end, tz="UTC") if end is not None else None

    bars: dict[str, pd.DataFrame] = {}
    for s in syms:
        path = data_dir / f"{s}.parquet"
        if not path.exists():
            continue
        df = resample_bars(load_intraday(s, data_dir), freq)
        if start_ts is not None:
            df = df[df.index >= start_ts]
        if end_ts is not None:
            # inclusive of the whole end date
            df = df[df.index <= end_ts + pd.Timedelta(days=1)]
        if not df.empty:
            bars[s] = df

    present = [s for s in symbols if s in bars]
    return Panel(bars=bars, freq=freq, market=market, symbols=present)
