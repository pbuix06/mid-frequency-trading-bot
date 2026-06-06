"""
Funding-reversal feature + config builder — the SINGLE source for the 10 pre-registered configs,
so the 30-day run and the 365-day backfill use byte-identical signal logic (no accidental drift).

These reproduce scripts/run_crypto_funding_reversal.py exactly. Do NOT add/tune configs here when
validating a backfill — that would be optimising after seeing data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.research.crypto_panel import (
    cumulative_funding_series,
    funding_zscore_series,
    map_to_bars,
)

H = {"4h": 48, "8h": 96, "24h": 288}   # 5-min bars
FUNDING_Z_WINDOW = 10
BASIS_Z_WINDOW = H["8h"]


def build_features(close: pd.DataFrame, funding_by_sym: dict, oi_by_sym: dict,
                   spot_close: pd.DataFrame, syms: list[str]):
    """All causal/known-only. Returns (funding_z, cum_funding, oi_change_4h, ret_4h, basis_z) wide."""
    idx = close.index

    def _has(d, s):
        return s in d and d[s] is not None and len(d[s]) > 0

    fz = pd.DataFrame({s: map_to_bars(funding_zscore_series(funding_by_sym[s], FUNDING_Z_WINDOW), idx)
                       for s in syms if _has(funding_by_sym, s)}).reindex(idx)
    cum = pd.DataFrame({s: map_to_bars(cumulative_funding_series(funding_by_sym[s]), idx)
                        for s in syms if _has(funding_by_sym, s)}).reindex(idx)
    oi4 = pd.DataFrame({s: map_to_bars(oi_by_sym[s], idx).pct_change(H["4h"])
                        for s in syms if _has(oi_by_sym, s)}).reindex(idx)
    ret4h = close.pct_change(H["4h"])
    basis = close / spot_close.reindex(idx) - 1.0
    basis_z = (basis - basis.rolling(BASIS_Z_WINDOW).mean()) / basis.rolling(BASIS_Z_WINDOW).std()
    return fz, cum, oi4, ret4h, basis_z


def build_configs(funding_z: pd.DataFrame, oi4: pd.DataFrame, ret4h: pd.DataFrame,
                  basis_z: pd.DataFrame):
    """The 10 pre-registered configs: (name, signal_wide, hold_bars, top_n, family)."""
    sig_A = -funding_z                                        # high = long crowded-shorts
    sig_B = sig_A.where(np.sign(ret4h) == np.sign(sig_A))     # 4h price confirmation
    sig_C = sig_A.where(oi4 > 0)                              # OI rising
    sig_basis = sig_A.where(basis_z.abs() > 1)                # extreme basis
    return [
        ("1 A fz 20% h4",    sig_A,    H["4h"], 2, "A"),
        ("2 A fz 20% h8",    sig_A,    H["8h"], 2, "A"),
        ("3 A fz 20% h24",   sig_A,    H["24h"], 2, "A"),
        ("4 A fz 10% h8",    sig_A,    H["8h"], 1, "A"),
        ("5 A fz 10% h24",   sig_A,    H["24h"], 1, "A"),
        ("6 B fund+wk4h h8", sig_B,    H["8h"], 2, "B"),
        ("7 B fund+wk4h h24", sig_B,   H["24h"], 2, "B"),
        ("8 C fund+OIup h8", sig_C,    H["8h"], 2, "C"),
        ("9 C fund+OIup h24", sig_C,   H["24h"], 2, "C"),
        ("10 fund+basis h8", sig_basis, H["8h"], 2, "basis"),
    ]
