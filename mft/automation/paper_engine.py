"""
Paper-trading engine — SIMULATED FILLS ONLY. There is no real-order adapter here, by design.

Runs a registered strategy forward over a data window, records EVERY hypothetical trade
(signal, side, entry/exit price, hold, cost, regime, reason, gross/funding/net) to a ledger, and
returns a summary (IC + net bps/trade). Reuses the project's continuous 24/7 rebalance + forward
returns + funding-carry conventions so paper numbers are comparable to the backtest.

Fill models: 'taker' (cross the spread, conservative) or 'maker_optimistic' (capture spread minus
fee+adverse — flagged optimistic; the project showed naive maker is adverse-selected).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mft.automation.registry import StrategySpec
from mft.research import signal_lab as L
from mft.research.targets import forward_return_panel
from mft.research.xs_backtest import _rebalance_points


@dataclass
class PaperConfig:
    cost_bps_per_side: float = 5.0          # crypto taker default
    mode: str = "taker"                     # 'taker' | 'maker_optimistic'
    maker_spread_bps: float = 5.0
    maker_fee_bps: float = 1.0
    maker_adverse_bps: float = 2.0


@dataclass
class PaperResult:
    strategy: str
    ledger: pd.DataFrame
    summary: dict


def build_signal(spec: StrategySpec, close: pd.DataFrame,
                 funding_z: pd.DataFrame | None = None) -> pd.DataFrame:
    """Construct the signal_wide for a spec. HIGH signal = long."""
    p = spec.params
    if spec.family == "momentum":
        return close.pct_change(p["lookback_bars"])
    if spec.family == "reversal":
        return -close.pct_change(p["lookback_bars"])
    if spec.family == "funding_reversal":
        if funding_z is None:
            raise ValueError("funding_reversal needs funding_z")
        return -funding_z
    raise ValueError(f"unknown family {spec.family!r}")


def _roundtrip_cost_frac(side_signed_fwd: float, funding_frac: float, cfg: PaperConfig) -> tuple[float, float]:
    """Return (cost_frac, fill_note) for one $1 leg round-trip."""
    if cfg.mode == "taker":
        return 2.0 * cfg.cost_bps_per_side * 1e-4, "taker x2"
    # maker_optimistic: capture spread, pay maker fee + adverse on each side (best case)
    eff = (2 * cfg.maker_fee_bps + 2 * cfg.maker_adverse_bps - cfg.maker_spread_bps) * 1e-4
    return eff, "maker_opt"


def run_paper(spec: StrategySpec, close: pd.DataFrame, regimes: dict, cfg: PaperConfig,
              funding_z: pd.DataFrame | None = None, cum_funding: pd.DataFrame | None = None) -> PaperResult:
    """Simulate the strategy forward over `close`'s window. One ledger row per hypothetical leg-trade."""
    uni = [s for s in spec.universe if s in close.columns]
    sig = build_signal(spec, close, funding_z)[uni]
    hold, top_n = spec.params["hold_bars"], spec.params["top_n"]

    entry = close[uni].shift(-1)                       # enter next bar after the signal bar
    exit_ = close[uni].shift(-(1 + hold))              # exit `hold` bars later
    fwd = forward_return_panel(close[uni], hold, entry_lag=1, intraday_only=False)
    carry = (cum_funding[uni].shift(-(1 + hold)) - cum_funding[uni].shift(-1)) if cum_funding is not None else None
    trend = regimes.get("trend")

    rows = []
    for r in _rebalance_points(sig.index, hold, continuous=True):
        s = sig.loc[r].dropna()
        f = fwd.loc[r] if r in fwd.index else None
        if f is None:
            continue
        names = s.index.intersection(f.dropna().index)
        if carry is not None:
            names = names.intersection(carry.loc[r].dropna().index)
        if len(names) < 2 * top_n:
            continue
        s = s.loc[names].sort_values()
        regime = str(trend.loc[r]) if trend is not None and r in trend.index else "n/a"
        for side, group in (("long", s.index[-top_n:]), ("short", s.index[:top_n])):
            for x in group:
                signed_fwd = float(f[x]) if side == "long" else -float(f[x])
                fund = 0.0
                if carry is not None:
                    fund = float(-carry.loc[r][x]) if side == "long" else float(carry.loc[r][x])
                cost, fill = _roundtrip_cost_frac(signed_fwd, fund, cfg)
                net = signed_fwd + fund - cost
                rows.append({
                    "ts": r, "strategy": spec.name, "symbol": x, "side": side,
                    "signal": float(s[x]), "entry_px": float(entry[x].loc[r]),
                    "exit_px": float(exit_[x].loc[r]), "hold_bars": hold, "regime": regime,
                    "gross_bps": signed_fwd * 1e4, "funding_bps": fund * 1e4,
                    "cost_bps": cost * 1e4, "net_bps": net * 1e4, "fill": fill,
                    "reason": f"{side} {spec.family}: signal_rank {'top' if side == 'long' else 'bottom'}{top_n}, "
                              f"value={float(s[x]):+.4f}",
                })

    ledger = pd.DataFrame(rows)
    summary = _summarize(spec, ledger, sig, fwd)
    return PaperResult(spec.name, ledger, summary)


def _summarize(spec, ledger, sig, fwd) -> dict:
    ic = L.ic_summary(L.ic_series(sig, fwd, method="spearman", min_names=3))
    if ledger.empty:
        return {"n_trades": 0, "mean_net_bps": float("nan"), "ic": ic["ic_mean"]}
    by_regime = (ledger.groupby("regime")["net_bps"].mean()).round(3).to_dict()
    return {
        "n_trades": int(len(ledger)),
        "n_rebalances": int(ledger["ts"].nunique()),
        "mean_gross_bps": round(float(ledger["gross_bps"].mean()), 3),
        "mean_funding_bps": round(float(ledger["funding_bps"].mean()), 3),
        "mean_cost_bps": round(float(ledger["cost_bps"].mean()), 3),
        "mean_net_bps": round(float(ledger["net_bps"].mean()), 3),
        "win_rate": round(float((ledger["net_bps"] > 0).mean()), 4),
        "ic": round(float(ic["ic_mean"]), 4),
        "ic_tstat": round(float(ic["ic_tstat"]), 2) if np.isfinite(ic["ic_tstat"]) else float("nan"),
        "net_bps_by_regime": by_regime,
    }
