"""
Daily session features from minute bars — the basis for longer-hold intraday.

Path B (taker-viable) strategies decide ONCE per session and hold for hours
(open->close), so they trade ~once a day and can capture moves bigger than the
spread — unlike the per-minute reversal that the spread destroys. We compress
minute bars into per-DAY session features and backtest at daily frequency.

Per day (regular session 09:30-16:00 ET):
  prev_close, open, close, high, low,
  overnight_gap   = open / prev_close - 1        (known at the open)
  intraday_ret    = close / open - 1             (the open->close holding return)
  or_high, or_low = high/low of the first `or_minutes` bars (opening range)
  brk_dir, brk_entry = first post-opening-range breakout direction + level
                       (+1 broke above or_high, -1 broke below or_low, 0 none)

All decision inputs (gap, opening range, breakout level) are knowable before the
holding period they predict — no look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _regular_session(minute_df: pd.DataFrame) -> pd.DataFrame:
    et = minute_df.index.tz_convert("America/New_York")
    mask = (et.time >= pd.Timestamp("09:30").time()) & (et.time <= pd.Timestamp("16:00").time())
    out = minute_df[mask].copy()
    out["et_date"] = out.index.tz_convert("America/New_York").date
    return out


def daily_session_features(minute_df: pd.DataFrame, or_minutes: int = 30) -> pd.DataFrame:
    """Compress minute bars to one row per trading day (see module docstring)."""
    df = _regular_session(minute_df)
    rows = []
    for day, g in df.groupby("et_date", sort=True):
        if len(g) < or_minutes + 5:
            continue
        opn = float(g["open"].iloc[0])
        cls = float(g["close"].iloc[-1])
        or_block = g.iloc[:or_minutes]
        or_high = float(or_block["high"].max())
        or_low = float(or_block["low"].min())
        post = g.iloc[or_minutes:]
        # first post-OR breakout
        brk_dir, brk_entry = 0, np.nan
        up = post.index[post["high"] >= or_high]
        dn = post.index[post["low"] <= or_low]
        first_up = up[0] if len(up) else None
        first_dn = dn[0] if len(dn) else None
        if first_up is not None and (first_dn is None or first_up <= first_dn):
            brk_dir, brk_entry = 1, or_high
        elif first_dn is not None:
            brk_dir, brk_entry = -1, or_low
        rows.append({
            "date": pd.Timestamp(day, tz="America/New_York").tz_convert("UTC"),
            "open": opn, "close": cls,
            "high": float(g["high"].max()), "low": float(g["low"].min()),
            "or_high": or_high, "or_low": or_low,
            "brk_dir": brk_dir, "brk_entry": brk_entry,
        })
    feat = pd.DataFrame(rows).set_index("date").sort_index()
    feat["prev_close"] = feat["close"].shift(1)
    feat["overnight_gap"] = feat["open"] / feat["prev_close"] - 1.0
    feat["intraday_ret"] = feat["close"] / feat["open"] - 1.0
    return feat


def realized_intraday_vol(feat: pd.DataFrame, asof: pd.Timestamp, window: int = 63) -> float:
    """
    Realized intraday volatility from the `window` sessions STRICTLY BEFORE `asof`.

    Ex-ante by construction: a session dated `asof` or later can never enter the
    estimate, so this is safe to use for universe selection on the live path.
    Returns NaN until `window` prior sessions exist.
    """
    hist = feat.loc[feat.index < asof, "intraday_ret"].dropna()
    if len(hist) < window:
        return float("nan")
    return float(hist.iloc[-window:].std())


def select_high_vol_universe(
    feats: dict[str, pd.DataFrame],
    asof: pd.Timestamp,
    top: int,
    window: int = 63,
) -> list[str]:
    """
    Ex-ante universe rule (spec §3): rank symbols by realized intraday vol over the
    `window` sessions before `asof` and return the top `top`. Uses only past data,
    so the selection is knowable before the trading session it governs.

    This is the PRODUCTION universe rule. The research script's full-sample ranking
    is a frozen-free-data convenience; the live/validation path uses this.
    """
    vols = {s: realized_intraday_vol(f, asof, window) for s, f in feats.items()}
    ranked = sorted(
        (s for s, v in vols.items() if pd.notna(v)),
        key=lambda s: -vols[s],
    )
    return ranked[:top]
