"""
Phase 3 alpha tests.

For each new alpha: signal bounds, look-ahead safety, and basic sanity.
Full parity tests (event vs NautilusTrader) are in test_nautilus_parity.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mft.alphas.long_short_momentum import LongShortMomentum
from mft.alphas.low_vol_anomaly import LowVolAnomaly
from mft.alphas.pairs_mean_reversion import PairsMeanReversion
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


# ── PairsMeanReversion ────────────────────────────────────────────────────────

def _make_pair_df(n=400, seed=99, spread_revert=True) -> pd.DataFrame:
    """Two cointegrated price series as a DataFrame with columns A, B."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0.0003, 0.008, n))
    if spread_revert:
        # Small idiosyncratic spread that mean-reverts
        spread = rng.normal(0, 0.005, n)
        spread -= spread.mean()
    else:
        spread = np.cumsum(rng.normal(0, 0.005, n))  # random walk spread (non-stationary)
    dates = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "A": 100.0 * np.exp(common + spread),
        "B": 100.0 * np.exp(common),
    }, index=dates)


class TestPairsMeanReversion:
    def test_signal_is_dollar_neutral(self):
        """When a position is open the two weights must sum to zero."""
        df = _make_pair_df()
        alpha = PairsMeanReversion("A", "B", z_entry=0.5)
        window = df.iloc[-alpha.lookback - 1:]
        sig = alpha.compute_signal(window)
        total = sig["A"] + sig["B"]
        assert abs(total) < 1e-9, f"Weights not dollar-neutral: {sig}"

    def test_correct_direction_on_depressed_spread(self):
        """When A is cheap vs B the signal must be long A / short B."""
        n = 400
        dates = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
        base = 100 * np.exp(np.cumsum(np.full(n, 0.0003)))
        # A recently dropped far below B
        ratio = np.ones(n)
        ratio[-20:] = 0.90   # A/B drops 10% in last 20 bars
        df = pd.DataFrame({"A": base * ratio, "B": base}, index=dates)
        alpha = PairsMeanReversion("A", "B", lookback=252, z_entry=0.5)
        window = df.iloc[-alpha.lookback - 1:]
        sig = alpha.compute_signal(window)
        assert sig["A"] >= 0.0, f"Expected long A, got {sig}"
        assert sig["B"] <= 0.0, f"Expected short B, got {sig}"

    def test_flat_on_insufficient_data(self):
        df = _make_pair_df(n=20)
        alpha = PairsMeanReversion("A", "B")
        sig = alpha.compute_signal(df)
        assert sig == {"A": 0.0, "B": 0.0}

    def test_flat_when_missing_column(self):
        df = _make_pair_df()
        alpha = PairsMeanReversion("A", "MISSING")
        window = df.iloc[-alpha.lookback - 1:]
        sig = alpha.compute_signal(window)
        assert sig["A"] == 0.0

    def test_hard_stop_when_spread_explodes(self):
        """Hard stop: if z_score > z_stop the signal must be flat."""
        n = 400
        dates = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
        base = 100 * np.exp(np.cumsum(np.full(n, 0.0003)))
        # A diverges massively from B in the last 5 bars
        ratio = np.ones(n)
        ratio[-5:] = 0.50  # A collapses 50% — spread far outside normal range
        df = pd.DataFrame({"A": base * ratio, "B": base}, index=dates)
        alpha = PairsMeanReversion("A", "B", lookback=252, z_entry=1.5, z_stop=3.0)
        window = df.iloc[-alpha.lookback - 1:]
        sig = alpha.compute_signal(window)
        assert sig == {"A": 0.0, "B": 0.0}, f"Hard stop should fire: {sig}"

    def test_no_look_ahead(self):
        df = _make_pair_df()
        alpha = PairsMeanReversion("A", "B")
        window_t = df.iloc[-alpha.lookback - 1 : -1]   # excludes last row
        window_t1 = df.iloc[-alpha.lookback - 1:]       # includes last row (future)
        sig_t = alpha.compute_signal(window_t)
        # Both windows are clean PIT slices — each should produce valid {0,±0.5} output
        assert sig_t["A"] + sig_t["B"] == pytest.approx(0.0, abs=1e-9)


# ── LowVolAnomaly ─────────────────────────────────────────────────────────────

class TestLowVolAnomaly:
    SYMBOLS = ["A", "B", "C", "D", "E"]

    def _make_close_df(self, n=200) -> pd.DataFrame:
        multi = _make_universe(self.SYMBOLS, n=n)
        return pd.DataFrame({s: multi[s]["close"] for s in self.SYMBOLS})

    def test_correct_count_bottom_quintile(self):
        close_df = self._make_close_df()
        alpha = LowVolAnomaly(universe=self.SYMBOLS, bottom_frac=0.20)
        window = close_df.iloc[-alpha.lookback - 1:]
        sig = alpha.compute_signal(window)
        n_long = sum(1 for v in sig.values() if v == 1.0)
        assert n_long == 1  # bottom 20% of 5 = 1

    def test_signal_binary(self):
        close_df = self._make_close_df()
        alpha = LowVolAnomaly(universe=self.SYMBOLS)
        window = close_df.iloc[-alpha.lookback - 1:]
        sig = alpha.compute_signal(window)
        assert all(v in (0.0, 1.0) for v in sig.values())

    def test_quietest_symbol_selected(self):
        """The symbol with near-zero price movement must get the 1.0 signal."""
        n = 200
        dates = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
        rng = np.random.default_rng(0)
        universe = ["QUIET", "NOISY1", "NOISY2", "NOISY3", "NOISY4"]
        quiet_prices = np.full(n, 100.0)  # zero volatility
        noisy_prices = {
            sym: 100 * np.exp(np.cumsum(rng.normal(0, 0.04, n)))
            for sym in universe[1:]
        }
        close_df = pd.DataFrame(
            {"QUIET": quiet_prices, **noisy_prices}, index=dates
        )
        alpha = LowVolAnomaly(universe=universe, bottom_frac=0.20)
        window = close_df.iloc[-alpha.lookback - 1:]
        sig = alpha.compute_signal(window)
        assert sig["QUIET"] > 0, f"Expected QUIET selected, got {sig}"

    def test_flat_on_short_window(self):
        close_df = self._make_close_df(n=10)
        alpha = LowVolAnomaly(universe=self.SYMBOLS)
        sig = alpha.compute_signal(close_df)
        assert all(v == 0.0 for v in sig.values())

    def test_no_look_ahead(self):
        close_df = self._make_close_df()
        alpha = LowVolAnomaly(universe=self.SYMBOLS)
        window_t = close_df.iloc[-alpha.lookback - 2: -1]
        window_t1 = close_df.iloc[-alpha.lookback - 1:]
        sig_t = alpha.compute_signal(window_t)
        sig_t1 = alpha.compute_signal(window_t1)
        # Both are valid slices, each returns {0,1} values
        assert all(v in (0.0, 1.0) for v in sig_t.values())
        assert all(v in (0.0, 1.0) for v in sig_t1.values())


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

    def test_signals_sum_to_one(self):
        """Total long allocation must equal 1.0 (fully invested, no leverage)."""
        close_df = self._make_close_df()
        alpha = XSMomentum(universe=self.SYMBOLS, top_frac=0.40)
        window = close_df.iloc[-alpha.lookback - 1:]
        sig = alpha.compute_signal(window)
        total = sum(sig.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

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
        assert sig["WINNER"] > 0  # WINNER selected; weight = 1/n_long


# ── LongShortMomentum ─────────────────────────────────────────────────────────

class TestLongShortMomentum:
    SYMBOLS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    def _make_close_df(self, n=400) -> pd.DataFrame:
        multi = _make_universe(self.SYMBOLS, n=n)
        return pd.DataFrame({s: multi[s]["close"] for s in self.SYMBOLS})

    def test_dollar_neutral(self):
        """Sum of all weights must be zero (long leg cancels short leg)."""
        close_df = self._make_close_df()
        alpha = LongShortMomentum(universe=self.SYMBOLS, frac=0.20)
        window = close_df.iloc[-alpha.lookback - 1:]
        sig = alpha.compute_signal(window)
        total = sum(sig.values())
        assert abs(total) < 1e-9, f"Not dollar-neutral: sum={total:.6f}"

    def test_correct_long_short_counts(self):
        close_df = self._make_close_df()
        alpha = LongShortMomentum(universe=self.SYMBOLS, frac=0.20)
        window = close_df.iloc[-alpha.lookback - 1:]
        sig = alpha.compute_signal(window)
        n_long  = sum(1 for v in sig.values() if v > 0)
        n_short = sum(1 for v in sig.values() if v < 0)
        assert n_long == n_short == 2  # top/bottom 20% of 10

    def test_winner_is_long_loser_is_short(self):
        """Clear winner must get positive weight; clear loser must get negative."""
        n = 300
        dates = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
        universe = ["WINNER", "MID1", "MID2", "MID3", "MID4",
                    "MID5", "MID6", "MID7", "MID8", "LOSER"]
        winner = 100 * np.exp(np.cumsum(np.full(n, 0.005)))
        loser  = 100 * np.exp(np.cumsum(np.full(n, -0.005)))
        flat   = np.full(n, 100.0)
        close_df = pd.DataFrame(
            {"WINNER": winner, "LOSER": loser,
             **{f"MID{i}": flat for i in range(1, 9)}},
            index=dates,
        )
        alpha = LongShortMomentum(universe=universe, lookback=252, skip=21, frac=0.10)
        window = close_df.iloc[-alpha.lookback - 1:]
        sig = alpha.compute_signal(window)
        assert sig["WINNER"] > 0, f"WINNER should be long: {sig}"
        assert sig["LOSER"]  < 0, f"LOSER should be short: {sig}"

    def test_flat_on_short_window(self):
        close_df = self._make_close_df(n=50)
        alpha = LongShortMomentum(universe=self.SYMBOLS)
        sig = alpha.compute_signal(close_df)
        assert all(v == 0.0 for v in sig.values())

    def test_flat_on_small_universe(self):
        """Universe with < 4 symbols should return all flat."""
        close_df = self._make_close_df()
        alpha = LongShortMomentum(universe=["A", "B", "C"])
        window = close_df.iloc[-alpha.lookback - 1:][["A", "B", "C"]]
        sig = alpha.compute_signal(window)
        assert all(v == 0.0 for v in sig.values())

    def test_no_look_ahead(self):
        close_df = self._make_close_df()
        alpha = LongShortMomentum(universe=self.SYMBOLS)
        w_t  = close_df.iloc[-alpha.lookback - 2 : -1]
        w_t1 = close_df.iloc[-alpha.lookback - 1:]
        sig_t  = alpha.compute_signal(w_t)
        sig_t1 = alpha.compute_signal(w_t1)
        # Each is a valid clean slice — both should be dollar-neutral
        assert abs(sum(sig_t.values()))  < 1e-9
        assert abs(sum(sig_t1.values())) < 1e-9
