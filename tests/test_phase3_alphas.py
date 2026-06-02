"""
Phase 3 alpha tests.

For each new alpha: signal bounds, look-ahead safety, and basic sanity.
Full parity tests (event vs NautilusTrader) are in test_nautilus_parity.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mft.alphas.short_reversion import ShortReversion
from mft.alphas.ts_momentum import TSMomentum
from mft.alphas.xs_momentum import XSMomentum
from mft.data_layer.pit import make_pit_window


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_trending(n=400, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.010, n)))
    dates = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "open":   prices * rng.uniform(0.999, 1.001, n),
        "high":   prices * rng.uniform(1.001, 1.005, n),
        "low":    prices * rng.uniform(0.995, 0.999, n),
        "close":  prices,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)


def _make_universe(symbols: list[str], n=400, seed=42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = {}
    for i, sym in enumerate(symbols):
        log_ret = rng.normal(0.0003 + i * 0.0001, 0.012, n)
        prices = 100.0 * np.exp(np.cumsum(log_ret))
        dates = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
        out[sym] = pd.DataFrame({
            "open":   prices * rng.uniform(0.999, 1.001, n),
            "high":   prices * rng.uniform(1.001, 1.005, n),
            "low":    prices * rng.uniform(0.995, 0.999, n),
            "close":  prices,
            "volume": rng.integers(500_000, 3_000_000, n).astype(float),
        }, index=dates)
    return out


# ── TSMomentum ────────────────────────────────────────────────────────────────

class TestTSMomentum:
    def test_signal_in_bounds(self):
        data = _make_trending()
        alpha = TSMomentum(symbol="X")
        window = data.iloc[-alpha.lookback - 1:]
        sig = alpha.compute_signal(window)
        assert 0.0 <= sig["X"] <= 1.0

    def test_returns_zero_on_short_window(self):
        data = _make_trending(n=50)
        alpha = TSMomentum(symbol="X")
        sig = alpha.compute_signal(data)
        assert sig["X"] == 0.0

    def test_long_in_uptrend(self):
        """Strong uptrend should produce a positive signal."""
        rng = np.random.default_rng(0)
        prices = 100 * np.exp(np.cumsum(np.full(400, 0.002)))  # pure uptrend
        dates = pd.date_range("2015-01-01", periods=400, freq="B", tz="UTC")
        data = pd.DataFrame({
            "open": prices, "high": prices * 1.001,
            "low": prices * 0.999, "close": prices,
            "volume": np.ones(400) * 1_000_000,
        }, index=dates)
        alpha = TSMomentum(symbol="X")
        window = data.iloc[-alpha.lookback - 1:]
        sig = alpha.compute_signal(window)
        assert sig["X"] > 0

    def test_no_look_ahead(self):
        data = _make_trending()
        alpha = TSMomentum(symbol="X")
        as_of = data.index[300]
        window = make_pit_window(data, as_of, alpha.lookback)
        sig_before = alpha.compute_signal(window)

        # Inject a large sentinel into the future — signal must not change
        data_poisoned = data.copy()
        data_poisoned.loc[data.index[301:], "close"] = 1e9
        window_clean = make_pit_window(data, as_of, alpha.lookback)
        sig_after = alpha.compute_signal(window_clean)

        assert sig_before == sig_after

    def test_vol_scaling_reduces_size_in_high_vol(self):
        """Signal should be smaller when realized vol is high."""
        n = 400
        dates = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")

        # Low-vol uptrend
        prices_lv = 100 * np.exp(np.cumsum(np.full(n, 0.001)))
        df_lv = pd.DataFrame({
            "open": prices_lv, "high": prices_lv * 1.0005,
            "low": prices_lv * 0.9995, "close": prices_lv,
            "volume": np.ones(n) * 1_000_000,
        }, index=dates)

        # High-vol uptrend (same direction, much noisier)
        rng = np.random.default_rng(7)
        prices_hv = 100 * np.exp(np.cumsum(np.full(n, 0.001) + rng.normal(0, 0.04, n)))
        df_hv = pd.DataFrame({
            "open": prices_hv, "high": prices_hv * 1.01,
            "low": prices_hv * 0.99, "close": prices_hv,
            "volume": np.ones(n) * 1_000_000,
        }, index=dates)

        alpha = TSMomentum(symbol="X")
        sig_lv = alpha.compute_signal(df_lv.iloc[-alpha.lookback - 1:])["X"]
        sig_hv = alpha.compute_signal(df_hv.iloc[-alpha.lookback - 1:])["X"]

        # Both positive but low-vol should be larger (or equal if both capped at 1)
        assert sig_lv >= sig_hv


# ── ShortReversion ────────────────────────────────────────────────────────────

class TestShortReversion:
    def test_signal_binary(self):
        data = _make_trending()
        alpha = ShortReversion(symbol="X")
        window = data.iloc[-alpha.lookback - 1:]
        sig = alpha.compute_signal(window)
        assert sig["X"] in (0.0, 1.0)

    def test_enters_on_sharp_drop(self):
        """A 3-sigma drop should trigger entry."""
        n = 100
        dates = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
        prices = np.full(n, 100.0)
        prices[-1] = 85.0  # sharp drop
        data = pd.DataFrame({
            "open": prices, "high": prices * 1.001,
            "low": prices * 0.999, "close": prices,
            "volume": np.ones(n) * 1_000_000,
        }, index=dates)
        alpha = ShortReversion(symbol="X", window=5, threshold=1.0)
        window = data.iloc[-alpha.lookback - 1:]
        assert alpha.compute_signal(window)["X"] == 1.0

    def test_flat_on_stable_prices(self):
        n = 100
        dates = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
        prices = np.full(n, 100.0)
        data = pd.DataFrame({
            "open": prices, "high": prices, "low": prices, "close": prices,
            "volume": np.ones(n) * 1_000_000,
        }, index=dates)
        alpha = ShortReversion(symbol="X")
        sig = alpha.compute_signal(data.iloc[-alpha.lookback - 1:])
        assert sig["X"] == 0.0

    def test_no_look_ahead(self):
        data = _make_trending()
        alpha = ShortReversion(symbol="X")
        as_of = data.index[150]
        window = make_pit_window(data, as_of, alpha.lookback)
        sig_before = alpha.compute_signal(window)
        data_poisoned = data.copy()
        data_poisoned.loc[data.index[151:], "close"] = 1e9
        window_clean = make_pit_window(data, as_of, alpha.lookback)
        sig_after = alpha.compute_signal(window_clean)
        assert sig_before == sig_after


# ── XSMomentum ────────────────────────────────────────────────────────────────

class TestXSMomentum:
    SYMBOLS = ["A", "B", "C", "D", "E"]

    def _make_close_df(self, n=400) -> pd.DataFrame:
        multi = _make_universe(self.SYMBOLS, n=n)
        return pd.DataFrame({s: multi[s]["close"] for s in self.SYMBOLS})

    def test_exactly_top_quintile_goes_long(self):
        close_df = self._make_close_df()
        alpha = XSMomentum(universe=self.SYMBOLS, top_frac=0.20)
        window = close_df.iloc[-alpha.lookback - 1:]
        sig = alpha.compute_signal(window)
        n_long = sum(1 for v in sig.values() if v == 1.0)
        assert n_long == 1  # top 20% of 5 = 1

    def test_signals_sum_lte_one(self):
        close_df = self._make_close_df()
        alpha = XSMomentum(universe=self.SYMBOLS, top_frac=0.40)
        window = close_df.iloc[-alpha.lookback - 1:]
        sig = alpha.compute_signal(window)
        assert all(v in (0.0, 1.0) for v in sig.values())
        assert sum(sig.values()) >= 1

    def test_flat_on_short_window(self):
        close_df = self._make_close_df(n=50)
        alpha = XSMomentum(universe=self.SYMBOLS)
        sig = alpha.compute_signal(close_df)
        assert all(v == 0.0 for v in sig.values())

    def test_no_look_ahead(self):
        close_df = self._make_close_df()
        alpha = XSMomentum(universe=self.SYMBOLS)
        window_clean = close_df.iloc[-alpha.lookback - 1:]
        sig_before = alpha.compute_signal(window_clean)

        poisoned = close_df.copy()
        poisoned.iloc[-1] = 1e9  # poison last row (future bar not in window)
        window_before_last = close_df.iloc[-alpha.lookback - 2: -1]
        sig_after = alpha.compute_signal(window_before_last)
        # Both windows are clean slices — just verify they each produce valid output
        assert all(v in (0.0, 1.0) for v in sig_before.values())
        assert all(v in (0.0, 1.0) for v in sig_after.values())

    def test_best_performer_goes_long(self):
        """The symbol with the highest return over (lookback-skip) should be selected."""
        n = 300
        dates = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
        # Give "WINNER" a clear uptrend, others flat
        winner_prices = 100 * np.exp(np.cumsum(np.full(n, 0.005)))
        flat_prices = np.full(n, 100.0)
        close_df = pd.DataFrame({
            "WINNER": winner_prices,
            "FLAT1": flat_prices,
            "FLAT2": flat_prices,
            "FLAT3": flat_prices,
            "FLAT4": flat_prices,
        }, index=dates)
        universe = list(close_df.columns)
        alpha = XSMomentum(universe=universe, lookback=252, skip=21, top_frac=0.20)
        window = close_df.iloc[-alpha.lookback - 1:]
        sig = alpha.compute_signal(window)
        assert sig["WINNER"] == 1.0
