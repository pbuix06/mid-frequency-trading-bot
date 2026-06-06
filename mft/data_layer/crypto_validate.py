"""
Crypto data validation — the gate before any alpha touches this data.

Pure, deterministic checks on the normalized frames (no network). Every check returns
structured results so the audit report and the loader can both consume them. Crypto is
24/7: the only "expected gap" is an exchange outage, NOT a nightly/weekend close — do NOT
apply equity RTH logic here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_MS = 60_000


def validate_ohlcv(df: pd.DataFrame, bar_ms: int = MIN_MS) -> dict:
    """Run all OHLCV integrity checks. Returns a dict of findings (JSON-serializable)."""
    if df.empty:
        return {"empty": True}
    idx = df.index
    diffs_ms = np.diff(idx.view("int64")) // 1_000_000 if idx.tz is not None else None

    # gaps: consecutive bars further apart than one bar = missing bars (outage or thin trading)
    gap_pos = np.where(diffs_ms > bar_ms)[0] if diffs_ms is not None else np.array([])
    missing_bars = int(((diffs_ms[gap_pos] // bar_ms) - 1).sum()) if len(gap_pos) else 0
    expected = int((idx[-1].value - idx[0].value) // (bar_ms * 1_000_000)) + 1

    o, h, lo, c = df["open"], df["high"], df["low"], df["close"]
    ohlc_bad = int((
        (lo > o + 1e-12) | (lo > c + 1e-12) | (lo > h + 1e-12)
        | (h < o - 1e-12) | (h < c - 1e-12) | (h < lo - 1e-12)
    ).sum())
    neg_price = int(((o <= 0) | (h <= 0) | (lo <= 0) | (c <= 0)).sum())
    neg_vol = int((df["volume"] < 0).sum())
    zero_vol = int((df["volume"] == 0).sum())

    # volume sanity: bars whose volume is a wild outlier (> mean + 20*std) — informational
    v = df["volume"].astype(float)
    vol_outliers = int((v > v.mean() + 20 * v.std()).sum()) if v.std() > 0 else 0

    return {
        "empty": False,
        "n_bars": int(len(df)),
        "start": str(idx[0]),
        "end": str(idx[-1]),
        "tz_is_utc": str(idx.tz) == "UTC",
        "monotonic_increasing": bool(idx.is_monotonic_increasing),
        "duplicate_timestamps": int(idx.duplicated().sum()),
        "expected_bars_if_continuous": expected,
        "coverage_pct": round(100.0 * len(df) / max(expected, 1), 3),
        "n_gaps": int(len(gap_pos)),
        "missing_bars": missing_bars,
        "largest_gap_minutes": int(diffs_ms.max()) if diffs_ms is not None and len(diffs_ms) else 0,
        "ohlc_inconsistent_bars": ohlc_bad,
        "negative_or_zero_prices": neg_price,
        "negative_volume_bars": neg_vol,
        "zero_volume_bars": zero_vol,
        "volume_extreme_outliers": vol_outliers,
        "nan_close_bars": int(c.isna().sum()),
    }


def validate_funding(ser: pd.Series) -> dict:
    if ser is None or ser.empty:
        return {"present": False}
    idx = ser.index
    diffs_h = np.diff(idx.view("int64")) / 3.6e12 if idx.tz is not None else None
    med_h = float(np.median(diffs_h)) if diffs_h is not None and len(diffs_h) else float("nan")
    return {
        "present": True,
        "n": int(len(ser)),
        "start": str(idx[0]), "end": str(idx[-1]),
        "tz_is_utc": str(idx.tz) == "UTC",
        "monotonic_increasing": bool(idx.is_monotonic_increasing),
        "duplicate_timestamps": int(idx.duplicated().sum()),
        "median_spacing_hours": round(med_h, 3),
        "looks_8h_aligned": bool(abs(med_h - 8.0) < 0.5) if med_h == med_h else False,
        "rate_min": float(ser.min()), "rate_max": float(ser.max()),
        "nan_count": int(ser.isna().sum()),
    }


def validate_oi(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"present": False}
    idx = df.index
    diffs_min = np.diff(idx.view("int64")) / 6e10 if idx.tz is not None else None
    med_min = float(np.median(diffs_min)) if diffs_min is not None and len(diffs_min) else float("nan")
    return {
        "present": True,
        "n": int(len(df)),
        "start": str(idx[0]), "end": str(idx[-1]),
        "tz_is_utc": str(idx.tz) == "UTC",
        "monotonic_increasing": bool(idx.is_monotonic_increasing),
        "duplicate_timestamps": int(idx.duplicated().sum()),
        "median_spacing_minutes": round(med_min, 2),
        "negative_oi_bars": int((df["open_interest"] < 0).sum()),
        "nan_bars": int(df["open_interest"].isna().sum()),
    }


def ohlcv_is_clean(report: dict) -> bool:
    """Hard pass/fail for an OHLCV frame: the dealbreakers, not the informational counts."""
    if report.get("empty", True):
        return False
    return (
        report["tz_is_utc"]
        and report["monotonic_increasing"]
        and report["duplicate_timestamps"] == 0
        and report["ohlc_inconsistent_bars"] == 0
        and report["negative_or_zero_prices"] == 0
        and report["negative_volume_bars"] == 0
        and report["nan_close_bars"] == 0
    )
