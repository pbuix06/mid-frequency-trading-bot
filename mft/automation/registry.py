"""
Strategy registry — the single source of truth for which signals we monitor and what the
backtest said to EXPECT. The paper engine and the deviation monitor read `expected` so they can
flag when near-live behaviour diverges from history (false positives / decay).

HARD GOVERNANCE GATE: this project found NO production-ready alpha. Every spec's status is one of
{rejected, research, paper_watch} — NEVER "live". `LIVE_TRADING_APPROVED` is False and nothing in
this codebase sets it True. `paper_watch` means "simulate it forward to confirm it stays dead /
detect a false positive", NOT an endorsement.
"""

from __future__ import annotations

from dataclasses import dataclass

LIVE_TRADING_APPROVED = False                       # never True in this project
VALID_STATUS = {"rejected", "research", "paper_watch"}   # "live" is intentionally absent

CRYPTO_MAJORS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                 "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT")


@dataclass(frozen=True)
class StrategySpec:
    name: str
    asset_class: str                 # 'crypto' | 'equity'
    family: str                      # 'momentum' | 'reversal' | 'funding_reversal'
    universe: tuple[str, ...]
    params: dict                     # hold_bars, top_n, lookback_bars, ...
    expected: dict                   # ic, gross_bps, net_bps_5, verdict — FROM the backtest
    status: str                      # one of VALID_STATUS — never 'live'
    trial_ids: str                   # provenance in trials/trials.csv
    notes: str = ""

    def __post_init__(self):
        if self.status not in VALID_STATUS:
            raise ValueError(f"{self.name}: status {self.status!r} not allowed (no 'live' permitted)")


# Every registered strategy was REJECTED in research; we paper_watch a few to validate they stay dead.
REGISTRY: dict[str, StrategySpec] = {
    "crypto_reversal_30m": StrategySpec(
        name="crypto_reversal_30m", asset_class="crypto", family="reversal", universe=CRYPTO_MAJORS,
        params={"lookback_bars": 3, "hold_bars": 6, "top_n": 3},
        expected={"ic": 0.047, "gross_bps": 0.82, "net_bps_5": -9.18, "verdict": "REJECTED: taker-dead"},
        status="paper_watch", trial_ids="T0089/T0094/T0095-99",
        notes="5m XS reversal; real tiny gross, taker-dead; maker adverse-selected."),
    "crypto_funding_reversal_24h": StrategySpec(
        name="crypto_funding_reversal_24h", asset_class="crypto", family="funding_reversal",
        universe=CRYPTO_MAJORS, params={"hold_bars": 288, "top_n": 2, "funding_z_window": 10},
        expected={"ic": 0.016, "gross_bps": 7.0, "net_bps_5": -10.0, "verdict": "REJECTED: crash-beta"},
        status="paper_watch", trial_ids="T0102/T0112",
        notes="365d backfill: IC collapsed, net-negative OOS, PnL only in down months (crash beta)."),
    "crypto_momentum_24h": StrategySpec(
        name="crypto_momentum_24h", asset_class="crypto", family="momentum", universe=CRYPTO_MAJORS,
        params={"lookback_bars": 288, "hold_bars": 288, "top_n": 3},
        expected={"ic": -0.044, "gross_bps": 1.3, "net_bps_5": -12.0, "verdict": "REJECTED: wrong-signed"},
        status="paper_watch", trial_ids="T0122",
        notes="XS momentum wrong-signed (IC negative ⇒ cross-section mean-reverts at 4h-24h)."),
}


def all_specs() -> dict[str, StrategySpec]:
    return REGISTRY


def get(name: str) -> StrategySpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


def assert_no_live_approved() -> None:
    """Hard governance check — called by the CLI on every run and by the test suite."""
    if LIVE_TRADING_APPROVED:
        raise RuntimeError("LIVE_TRADING_APPROVED must be False — no live deployment is approved.")
    bad = [s.name for s in REGISTRY.values() if s.status not in VALID_STATUS]
    if bad:
        raise RuntimeError(f"strategies with disallowed status: {bad}")
