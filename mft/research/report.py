"""
Research outputs: results tables, master leaderboard, figures, markdown logs.

Conventions:
  - results/alpha_tests/<family>.parquet : one row per (config, split). Append-only-ish
    (rewritten wholesale per run; the trial ledger is the immutable record).
  - results/leaderboard.csv : every config, ranked by VALIDATION Sharpe. Lock-box columns
    are shown but NEVER used for ranking.
  - results/figures/ : IC-decay, bucket, equity PNGs.
  - research_logs/<family>.md : the human write-up (hypothesis -> result -> verdict).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[2]
RESULTS_DIR = ROOT / "results" / "alpha_tests"
FIG_DIR = ROOT / "results" / "figures"
LOG_DIR = ROOT / "research_logs"
LEADERBOARD = ROOT / "results" / "leaderboard.csv"

for _d in (RESULTS_DIR, FIG_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def save_results(rows: list[dict], family: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS_DIR / f"{family}.parquet")
    df.to_csv(RESULTS_DIR / f"{family}.csv", index=False)
    return df


def update_leaderboard(df: pd.DataFrame, rank_col: str = "val_sharpe") -> pd.DataFrame:
    cols = [c for c in df.columns if c != "_skip"]
    board = df[cols].copy()
    if LEADERBOARD.exists():
        prev = pd.read_csv(LEADERBOARD)
        board = pd.concat([prev, board], ignore_index=True)
    if rank_col in board.columns:
        board = board.sort_values(rank_col, ascending=False, na_position="last")
    board.to_csv(LEADERBOARD, index=False)
    return board


# ── figures (guarded: research runs even without matplotlib) ──────────────────

def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def plot_alpha_decay(ic_by_horizon: pd.Series, family: str, tag: str = "") -> Path | None:
    plt = _mpl()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color="grey", lw=0.8)
    ax.plot(ic_by_horizon.index, ic_by_horizon.values, marker="o")
    ax.set_xlabel("forward horizon (bars)")
    ax.set_ylabel("mean rank-IC")
    ax.set_title(f"{family} — alpha decay {tag}")
    p = FIG_DIR / f"{family}_alpha_decay{('_'+tag) if tag else ''}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def plot_buckets(bucket_ret: pd.Series, family: str, tag: str = "") -> Path | None:
    plt = _mpl()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bucket_ret.index.astype(str), bucket_ret.values * 1e4)
    ax.set_xlabel("signal bucket (0=lowest)")
    ax.set_ylabel("mean forward return (bps)")
    ax.set_title(f"{family} — bucket returns {tag}")
    p = FIG_DIR / f"{family}_buckets{('_'+tag) if tag else ''}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def plot_equity(equity: pd.Series, family: str, tag: str = "") -> Path | None:
    plt = _mpl()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(equity.index, equity.values)
    ax.set_ylabel("net equity (x)")
    ax.set_title(f"{family} — net equity {tag}")
    p = FIG_DIR / f"{family}_equity{('_'+tag) if tag else ''}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def write_research_log(family: str, body_md: str) -> Path:
    p = LOG_DIR / f"{family}.md"
    p.write_text(body_md)
    return p
