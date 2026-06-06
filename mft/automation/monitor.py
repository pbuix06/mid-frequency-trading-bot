"""
Monitors: data integrity, backtest-deviation (false-positive guard), and risk/cost.

The DeviationMonitor is the heart of this framework: it compares near-live paper results to the
registry's backtest `expected` and raises warnings when they DIVERGE — both ways. A paper result
that looks much BETTER than the (rejected) backtest is treated as a SUSPECTED FALSE POSITIVE, not a
discovery; a much worse one flags decay; a sign flip flags instability. Nothing here ever endorses
going live.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mft.automation.registry import StrategySpec
from mft.data_layer.crypto_validate import ohlcv_is_clean, validate_ohlcv
from mft.execution.costs import DEFAULT_COMMISSION_PCT, DEFAULT_SLIPPAGE_PCT


@dataclass
class Warning_:
    level: str       # 'info' | 'warn' | 'critical'
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level.upper():8}] {self.code}: {self.message}"


# ── data freshness / integrity ────────────────────────────────────────────────
class DataMonitor:
    def __init__(self, stale_minutes: int = 120):
        self.stale_minutes = stale_minutes

    def freshness(self, bars: pd.DataFrame, now: pd.Timestamp | None = None) -> dict:
        now = now or pd.Timestamp.now(tz="UTC")
        last = bars.index[-1]
        age = (now - last).total_seconds() / 60.0
        return {"last_bar": str(last), "age_minutes": round(age, 1), "stale": age > self.stale_minutes}

    def validate(self, bars: pd.DataFrame) -> tuple[dict, list[Warning_]]:
        rep = validate_ohlcv(bars)
        warns = []
        if not ohlcv_is_clean(rep):
            warns.append(Warning_("critical", "DATA_DIRTY",
                                  f"OHLCV failed integrity (dup={rep.get('duplicate_timestamps')}, "
                                  f"ohlc_bad={rep.get('ohlc_inconsistent_bars')}, "
                                  f"neg={rep.get('negative_or_zero_prices')})"))
        if rep.get("n_gaps", 0) > 0:
            warns.append(Warning_("warn", "DATA_GAPS",
                                  f"{rep['n_gaps']} gaps / {rep['missing_bars']} missing bars "
                                  f"({rep['coverage_pct']}% coverage)"))
        return rep, warns


# ── backtest-deviation / false-positive guard ────────────────────────────────
class DeviationMonitor:
    """Compare a paper summary to the registry's backtest expectation."""

    def __init__(self, fp_net_bps_tolerance: float = 5.0):
        # how far paper net may exceed backtest net before we suspect a FALSE POSITIVE
        self.fp_tol = fp_net_bps_tolerance

    def compare(self, spec: StrategySpec, summary: dict) -> list[Warning_]:
        warns: list[Warning_] = []
        exp = spec.expected
        if summary.get("n_trades", 0) == 0:
            return [Warning_("warn", "NO_TRADES", f"{spec.name}: paper produced no trades")]

        # 1. FALSE-POSITIVE guard — paper much better than the (rejected) backtest
        paper_net, exp_net = summary["mean_net_bps"], exp.get("net_bps_5", float("nan"))
        if np.isfinite(exp_net) and paper_net > exp_net + self.fp_tol:
            warns.append(Warning_("warn", "SUSPECTED_FALSE_POSITIVE",
                                  f"{spec.name}: paper net {paper_net:+.2f} bps >> backtest {exp_net:+.2f} "
                                  f"(Δ{paper_net - exp_net:+.2f}). A REJECTED strategy looking good live is a "
                                  f"red flag (regime luck / overfit), NOT a discovery. Do not act."))
        # 2. degradation
        if np.isfinite(exp_net) and paper_net < exp_net - 3 * self.fp_tol:
            warns.append(Warning_("info", "DEGRADATION",
                                  f"{spec.name}: paper net {paper_net:+.2f} << backtest {exp_net:+.2f} — further decay."))
        # 3. IC sign flip / collapse
        exp_ic, paper_ic = exp.get("ic", float("nan")), summary.get("ic", float("nan"))
        if np.isfinite(exp_ic) and np.isfinite(paper_ic) and np.sign(exp_ic) != np.sign(paper_ic) and abs(exp_ic) > 0.005:
            warns.append(Warning_("warn", "IC_SIGN_FLIP",
                                  f"{spec.name}: IC sign flipped (backtest {exp_ic:+.4f} -> paper {paper_ic:+.4f}) — unstable."))
        # 4. still net-negative (the expected, healthy state for a rejected strategy)
        if paper_net <= 0:
            warns.append(Warning_("info", "STAYS_REJECTED",
                                  f"{spec.name}: paper net {paper_net:+.2f} bps ≤ 0 — consistent with REJECTED verdict."))
        return warns


# ── risk + cost monitor ───────────────────────────────────────────────────────
class RiskCostMonitor:
    # realistic per-side taker cost floors; assuming below this is dishonest
    COST_FLOOR_BPS = {"crypto": 2.0, "equity": 1.0}
    MAX_GROSS_EXPOSURE = 1.0      # $1 gross book (dollar-neutral)
    MAX_NAME_WEIGHT = 0.20

    def check_cost_assumption(self, asset_class: str, cost_bps_per_side: float) -> list[Warning_]:
        floor = self.COST_FLOOR_BPS.get(asset_class, 1.0)
        if cost_bps_per_side < floor:
            return [Warning_("warn", "COST_TOO_LOW",
                             f"cost {cost_bps_per_side} bps/side < realistic {asset_class} floor {floor} — "
                             f"results will be optimistic.")]
        return []

    def slippage_assumptions(self) -> dict:
        return {"default_commission_bps_per_side": DEFAULT_COMMISSION_PCT * 1e4,
                "default_slippage_bps_per_side": DEFAULT_SLIPPAGE_PCT * 1e4,
                "note": "research assumptions; real impact calibrates to live fills (Phase 6) — N/A, no live."}

    def check_exposure(self, ledger: pd.DataFrame) -> list[Warning_]:
        if ledger.empty:
            return []
        warns = []
        # dollar-neutral by construction (equal top/bottom-N); flag if a rebalance is lopsided
        bal = ledger.groupby(["ts", "side"]).size().unstack(fill_value=0)
        if "long" in bal and "short" in bal and (bal["long"] != bal["short"]).any():
            warns.append(Warning_("warn", "NOT_DOLLAR_NEUTRAL", "some rebalances have unequal long/short legs"))
        return warns
