"""
Gate 0 Parity Check — Day 7 of Appendix C.

Runs the SMA strategy through both engines on the same data.
Equity curves must reconcile within tolerance.

If they match: Gate 0's parity test is half-passed and you've de-risked
the single biggest failure mode (strategy works in backtest, breaks live
because live doesn't behave like the simulator).

Usage:
    python scripts/parity_check.py --data data/pit/SPY/ --symbol SPY

With synthetic data (no real data needed for Phase 0):
    python scripts/parity_check.py --synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from mft.alphas.sma_crossover import SMACrossover
from mft.backtest.event_harness import equity_to_returns, run_event_driven
from mft.data_layer.eodhd_ingest import LOCKBOX_CUTOFF
from mft.monitoring.trial_log import TrialLog
from mft.validation.metrics import full_metrics


def make_synthetic(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0003, 0.012, n)
    prices = 100.0 * np.exp(np.cumsum(log_ret))
    dates = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": prices * rng.uniform(0.998, 1.000, n),
            "high": prices * rng.uniform(1.001, 1.005, n),
            "low": prices * rng.uniform(0.995, 0.999, n),
            "close": prices,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )


def run_parity(data: pd.DataFrame, symbol: str, fast: int = 20, slow: int = 50) -> None:
    alpha = SMACrossover(symbol=symbol, fast=fast, slow=slow)

    print(f"\n{'=' * 60}")
    print("SMA Crossover Parity Check")
    print(f"  symbol={symbol}  fast={fast}  slow={slow}")
    print(f"  data: {len(data)} bars  {data.index[0].date()} → {data.index[-1].date()}")
    print(f"{'=' * 60}\n")

    # ── Event-driven harness ──────────────────────────────────────────────────
    print("Running event-driven harness...")
    state = run_event_driven(alpha, data, symbol=symbol)
    event_returns = equity_to_returns(state)
    event_metrics = full_metrics(event_returns)
    event_equity = [v for _, v in state.equity_curve]
    print(f"  Fills: {len(state.fills)}")
    print(f"  Final equity: {event_equity[-1]:,.2f}")
    print(f"  Sharpe: {event_metrics['sharpe']:.3f}")
    print(f"  Max DD: {event_metrics['max_drawdown']:.2%}")

    # ── vectorbt harness ─────────────────────────────────────────────────────
    try:
        from mft.backtest.vectorbt_harness import get_metrics as get_vbt_metrics
        from mft.backtest.vectorbt_harness import run_research

        print("\nRunning vectorbt research harness...")
        pf = run_research(alpha, data, symbol=symbol)
        vbt_m = get_vbt_metrics(pf)
        print(f"  Trades: {vbt_m['n_trades']}")
        print(f"  Final equity: {pf.value().iloc[-1]:,.2f}")
        print(f"  Sharpe: {vbt_m['sharpe']:.3f}")
        print(f"  Max DD: {vbt_m['max_dd']:.2%}")

        # ── Parity comparison ──────────────────────────────────────────────
        # vectorbt counts round-trips; event harness counts individual fills.
        # Convert fills to approximate round-trips for comparison.
        event_roundtrips = len(state.fills) / 2
        sharpe_diff = abs(event_metrics["sharpe"] - vbt_m["sharpe"])
        roundtrip_diff = abs(event_roundtrips - vbt_m["n_trades"])

        print(f"\n{'─' * 40}")
        print("Parity check:")
        print(f"  Sharpe diff:    {sharpe_diff:.4f}  {'✓ PASS' if sharpe_diff < 0.1 else '✗ FAIL'}")
        print(f"  Round-trips:    event≈{event_roundtrips:.0f} vbt={vbt_m['n_trades']}  "
              f"{'✓ PASS' if roundtrip_diff <= 2 else '✗ FAIL'}")

        if sharpe_diff < 0.1 and roundtrip_diff <= 2:
            print("\n✓ GATE 0 PARITY: PASSED")
            print("  Same strategy object → reconciling results in both engines.")
            print("  You have de-risked the biggest backtest-to-live gap.")
        else:
            print("\n✗ GATE 0 PARITY: FAILED")
            print("  Investigate execution semantics differences.")

    except ImportError:
        print("\nvectorbt not installed. Install with: pip install vectorbt")
        print("Event-driven harness complete. Install vectorbt for full parity check.")

    # ── Log the trial ────────────────────────────────────────────────────────
    log = TrialLog()
    trial_id = log.log(
        strategy="SMACrossover",
        asset_universe=[symbol],
        params={"fast": fast, "slow": slow},
        data_window=f"{data.index[0].date()}:{data.index[-1].date()}",
        is_sharpe=event_metrics["sharpe"],
        max_dd=event_metrics["max_drawdown"],
        notes="Gate 0 parity check",
    )
    print(f"\nLogged as {trial_id} (total trials: {log.count()})")
    print(f"MinBTL budget check: {log.count()} trials used.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 0 Parity Check")
    parser.add_argument("--data", type=str, default=None, help="Path to Parquet data")
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument(
        "--include-lockbox",
        action="store_true",
        help="DANGER: use data past LOCKBOX_CUTOFF. Only for the Phase 4 final exam.",
    )
    args = parser.parse_args()

    if args.synthetic or args.data is None:
        print("Using synthetic data (for Phase 0 setup — use real PIT data from Phase 1)")
        data = make_synthetic()
    else:
        from mft.data_layer.loader import load_parquet
        data = load_parquet(args.data, symbol=args.symbol)
        if args.include_lockbox:
            print("\n  ⚠  --include-lockbox set: USING LOCK-BOX DATA. Final exam only.\n")
        else:
            data = data[data.index <= LOCKBOX_CUTOFF]
            print(f"\n  Lock-box enforced: data truncated at {LOCKBOX_CUTOFF.date()}\n")

    run_parity(data, symbol=args.symbol, fast=args.fast, slow=args.slow)


if __name__ == "__main__":
    main()
