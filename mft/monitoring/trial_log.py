"""
Append-only trial log.

Rule: every backtest run gets one row. No exceptions.
This is what lets you compute MinBTL and deflate your Sharpe honestly.
Untracked search is guaranteed overfit.

Schema (trials/trials.csv):
    trial_id       — sequential T0001, T0002, ...
    date           — ISO date the trial ran
    strategy       — alpha class name
    asset_universe — JSON list of symbols
    params_json    — JSON dict of strategy parameters
    data_window    — "YYYY-MM-DD:YYYY-MM-DD"
    is_sharpe      — in-sample Sharpe (raw, not deflated)
    oos_sharpe     — out-of-sample Sharpe (if computed; else empty)
    max_dd         — maximum drawdown (negative fraction)
    turnover       — avg daily turnover fraction
    n_trials_so_far — running total at time of this trial
    notes          — free text
    git_commit     — short hash for reproducibility
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import date
from pathlib import Path

TRIAL_LOG_PATH = Path(__file__).parents[2] / "trials" / "trials.csv"

COLUMNS = [
    "trial_id",
    "date",
    "strategy",
    "asset_universe",
    "params_json",
    "data_window",
    "is_sharpe",
    "oos_sharpe",
    "max_dd",
    "turnover",
    "n_trials_so_far",
    "notes",
    "git_commit",
]


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "no-git"


def _count_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with open(path) as f:
        return sum(1 for _ in csv.DictReader(f))


class TrialLog:
    """
    Append-only trial log.

    Usage:
        log = TrialLog()
        trial_id = log.log(
            strategy="SMACrossover",
            asset_universe=["SPY"],
            params={"fast": 20, "slow": 50},
            data_window="2010-01-01:2024-01-01",
            is_sharpe=0.82,
            oos_sharpe=0.71,
            max_dd=-0.14,
        )
        print(f"Trial budget used: {log.count()} of your MinBTL limit")
    """

    def __init__(self, path: Path = TRIAL_LOG_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    def log(
        self,
        *,
        strategy: str,
        asset_universe: list[str],
        params: dict,
        data_window: str,
        is_sharpe: float,
        oos_sharpe: float | None = None,
        max_dd: float | None = None,
        turnover: float | None = None,
        notes: str = "",
    ) -> str:
        """Append one row and return the trial_id."""
        n = _count_rows(self.path) + 1
        trial_id = f"T{n:04d}"

        row = {
            "trial_id": trial_id,
            "date": date.today().isoformat(),
            "strategy": strategy,
            "asset_universe": json.dumps(asset_universe),
            "params_json": json.dumps(params),
            "data_window": data_window,
            "is_sharpe": f"{is_sharpe:.4f}",
            "oos_sharpe": f"{oos_sharpe:.4f}" if oos_sharpe is not None else "",
            "max_dd": f"{max_dd:.4f}" if max_dd is not None else "",
            "turnover": f"{turnover:.4f}" if turnover is not None else "",
            "n_trials_so_far": n,
            "notes": notes,
            "git_commit": _git_commit(),
        }

        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)

        return trial_id

    def count(self) -> int:
        """Total trials logged. Use this to compute your MinBTL budget."""
        return _count_rows(self.path)

    def load(self) -> list[dict]:
        """Load all rows as list of dicts."""
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return list(csv.DictReader(f))
