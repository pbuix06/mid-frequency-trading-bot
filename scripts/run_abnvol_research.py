"""
Stock alpha family 2 of 8 — ABNORMAL-VOLUME CONTINUATION (5-min bars), IC-first.

Hypothesis: a name with abnormal same-time-of-day volume AND a strong market-adjusted
(residual-vs-SPY) move CONTINUES in that direction over the next 30–120 min — informed flow
shows up as volume, and volume-confirmed moves persist. Long up-move+high-vol, short
down-move+high-vol, dollar-neutral.

Main question: does abnormal-volume continuation have a LARGER GROSS edge per trade than
short-term reversal (family 1), and does it survive costs better? We compare gross bps/trade
head-to-head and sweep costs 0/1/2/5/10 bps/side.

Method: IC-first vs RESIDUAL forward returns (15/30/60/120m) -> non-overlapping dollar-neutral
top/bottom-N backtest, gross & net, entry lag 1, flat overnight. Lock-box SEALED. 8 pre-registered
configs (filters 1–5 at top5; score 6–8 at top5 & top3) -> one trial row each.

Same caveats as family 1: 31 survivor mega-caps, ~2.9 yr, one regime. Engine + measurement, not
a deployable alpha. Run: python scripts/run_abnvol_research.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from mft.monitoring.trial_log import TrialLog  # noqa: E402
from mft.research import report as R  # noqa: E402
from mft.research import signal_lab as L  # noqa: E402
from mft.research import splits as SP  # noqa: E402
from mft.research.features import residual_return, volume_zscore_tod  # noqa: E402
from mft.research.panel import INTRADAY_DIR, load_panel  # noqa: E402
from mft.research.targets import beta_adjusted_forward_panel  # noqa: E402
from mft.research.xs_backtest import cross_sectional_ls  # noqa: E402
from mft.validation.metrics import sharpe  # noqa: E402

FAMILY = "abnormal_volume_continuation"
MARKET, EXCLUDE = "SPY", {"SPY", "IWM"}
M = {5: 1, 15: 3, 30: 6, 60: 12, 120: 24}
HORIZONS = [M[15], M[30], M[60], M[120]]
COST_SWEEP = (0.0, 1.0, 2.0, 5.0, 10.0)
CANON_COST = 2.0
TOD_8 = [("09:30", "11:30"), ("15:00", "16:00")]


def filter_signal(resid_h: pd.DataFrame, volz: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Continuation among volume-confirmed names: residual return, masked where vol_z<=thr."""
    return resid_h.where(volz > threshold)


def score_signal(resid30: pd.DataFrame, volz: pd.DataFrame) -> pd.DataFrame:
    """score = centered cross-sectional rank(residual_30m) * rank(vol_z). Signed for L/S."""
    rr = resid30.rank(axis=1, pct=True) - 0.5          # +up-mover / -down-mover
    vz = volz.rank(axis=1, pct=True)                    # weight by volume abnormality
    return rr * vz


def cost_sweep(gross: pd.Series, ppy: float) -> dict:
    """Net Sharpe & net bps/trade at each cost level — cost is sensitivity, not a trial."""
    out = {}
    for c in COST_SWEEP:
        net = gross - 2.0 * (c * 1e-4)
        out[c] = (sharpe(net, periods_per_year=ppy), float(net.mean() * 1e4))
    return out


def main() -> None:
    if not list(INTRADAY_DIR.glob("*.parquet")):
        print("No intraday data. Run scripts/ingest_alpaca.py first.")
        return

    start, end = SP.research_window()
    stocks = [p.stem for p in INTRADAY_DIR.glob("*.parquet") if p.stem not in EXCLUDE]
    panel = load_panel(sorted(stocks), start=start, end=end, freq="5min", market=MARKET)
    close_wide = panel.to_wide("close")
    SP.assert_no_lockbox(close_wide.index)
    mkt = panel.market_close()
    print(f"Loaded {len(panel.symbols)} names, {len(close_wide)} bars, {start}->{end} "
          f"(lock-box {SP.LOCKBOX_START.date()} SEALED)\n")

    # feature wides (built once, past-only)
    volz = pd.DataFrame({s: volume_zscore_tod(panel.bars[s], 20) for s in panel.symbols}
                        ).reindex(close_wide.index)
    resid = {h: pd.DataFrame({s: residual_return(panel.bars[s]["close"], mkt, h)
                              for s in panel.symbols}).reindex(close_wide.index)
             for h in (M[15], M[30], M[60])}

    # residual forward targets, precomputed per horizon (for IC + decay)
    rtgt = {h: beta_adjusted_forward_panel(close_wide, mkt, h) for h in HORIZONS}

    # ── pre-registered configs: (id, builder, hold_bars, top_n, tod_windows) ──
    score_full = score_signal(resid[M[30]], volz)
    configs = [
        ("1 z>1 resid15",   filter_signal(resid[M[15]], volz, 1.0), M[30], 5, None),
        ("2 z>1 resid30",   filter_signal(resid[M[30]], volz, 1.0), M[60], 5, None),
        ("3 z>1.5 resid30", filter_signal(resid[M[30]], volz, 1.5), M[60], 5, None),
        ("4 z>2 resid30",   filter_signal(resid[M[30]], volz, 2.0), M[60], 5, None),
        ("5 z>1.5 resid60", filter_signal(resid[M[60]], volz, 1.5), M[120], 5, None),
        ("6 score h60 N5",  score_full, M[60], 5, None),
        ("7 score h120 N5", score_full, M[120], 5, None),
        ("8 score tod N5",  score_full, M[60], 5, TOD_8),
        ("6 score h60 N3",  score_full, M[60], 3, None),
        ("7 score h120 N3", score_full, M[120], 3, None),
        ("8 score tod N3",  score_full, M[60], 3, TOD_8),
    ]

    log, rows = TrialLog(), []
    print(f"{'config':<16}{'N':>2}{'hold':>5} | {'trIC':>7}{'ICt':>5}{'mono':>5} | "
          f"{'grBps':>6}{'cost':>5}{'netBps':>7} | {'vGrSh':>6}{'vNtSh':>6}{'trd/d':>6} trial")
    print("-" * 100)

    head_cfg = "6 score h60 N5"
    for name, sig, hold, top_n, tod in configs:
        sp = SP.split_train_val(sig)
        cp = SP.split_train_val(close_wide)
        htgt_tr = SP.TRAIN.slice(rtgt[hold]) if hold in rtgt else \
            beta_adjusted_forward_panel(cp["train"], mkt, hold)

        ic = L.ic_series(sp["train"], htgt_tr, method="spearman")
        summ = L.ic_summary(ic)
        buckets = L.bucket_returns(sp["train"], htgt_tr, n_buckets=5)
        mono = L.bucket_monotonicity(buckets)

        tr = cross_sectional_ls(sp["train"], cp["train"], top_n=top_n, hold_bars=hold,
                                cost_bps_per_side=CANON_COST, tod_windows=tod)
        va = cross_sectional_ls(sp["validation"], cp["validation"], top_n=top_n, hold_bars=hold,
                                cost_bps_per_side=CANON_COST, tod_windows=tod)

        params = {"config": name, "hold_bars": hold, "top_n": top_n, "tod_windows": tod,
                  "cost_bps_per_side": CANON_COST, "freq": "5min"}
        tid = log.log(strategy="AbnVolContinuationXS", asset_universe=panel.symbols,
                      params=params, data_window=f"{start}:{end}",
                      is_sharpe=float(tr.metrics["sharpe"]), oos_sharpe=float(va.metrics["sharpe"]),
                      max_dd=float(va.metrics["max_drawdown"]), turnover=float(va.trades_per_day),
                      notes=(f"abn-vol continuation IC-first; LOCKBOX SEALED; trainIC={summ['ic_mean']:.4f} "
                             f"t={summ['ic_tstat']:.2f} grossBps/trade={va.metrics.get('mean_gross_per_trade_bps', float('nan')):.2f}"))

        rows.append({
            "family": FAMILY, "config": name, "hold_bars": hold, "horizon_min": hold * 5,
            "top_n": top_n, "tod_restricted": tod is not None,
            "train_ic_mean": summ["ic_mean"], "train_ic_tstat": summ["ic_tstat"],
            "bucket_monotonicity": mono,
            "val_gross_bps_per_trade": va.metrics["mean_gross_per_trade_bps"],
            "val_cost_bps_per_trade": va.metrics["cost_per_trade_bps"],
            "val_net_bps_per_trade": va.metrics["mean_net_per_trade_bps"],
            "val_sharpe_gross": va.metrics["sharpe_gross"], "val_sharpe_net": va.metrics["sharpe"],
            "val_max_dd": va.metrics["max_drawdown"], "val_trades_per_day": va.trades_per_day,
            "val_n_trades": va.n_trades, "trial_id": tid, "lockbox_touched": False,
        })

        print(f"{name:<16}{top_n:>2}{hold:>5} | {summ['ic_mean']:>7.4f}{summ['ic_tstat']:>5.1f}"
              f"{mono:>5.2f} | {va.metrics['mean_gross_per_trade_bps']:>6.2f}"
              f"{va.metrics['cost_per_trade_bps']:>5.1f}{va.metrics['mean_net_per_trade_bps']:>7.2f} | "
              f"{va.metrics['sharpe_gross']:>6.2f}{va.metrics['sharpe']:>6.1f}{va.trades_per_day:>6.1f} {tid}")

        if name == head_cfg:
            R.plot_alpha_decay(L.alpha_decay(sp["train"], cp["train"], HORIZONS), FAMILY, "train")
            R.plot_buckets(buckets, FAMILY, "train")
            R.plot_equity(va.equity(), FAMILY, "val")
            head = (name, va, cost_sweep(va.gross, va.periods_per_year), summ)

    df = R.save_results(rows, FAMILY)
    R.update_leaderboard(df[["family", "config", "hold_bars", "top_n", "train_ic_tstat",
                             "val_gross_bps_per_trade", "val_sharpe_net", "trial_id"]]
                         .rename(columns={"val_sharpe_net": "val_sharpe"}), rank_col="val_sharpe")

    # head-to-head gross bps/trade vs family-1 reversal (recompute, NO new trial)
    rev_gross = _reversal_gross_bps(close_wide, mkt)

    _print_and_log_report(df, head, rev_gross, start, end, log)


def _reversal_gross_bps(close_wide, mkt) -> float:
    """Family-1 headline (residual, lookback 3 bars, hold 6) gross bps/trade on validation."""
    sig = pd.DataFrame({c: -residual_return(close_wide[c].dropna(), mkt, M[15])
                        for c in close_wide.columns}).reindex(close_wide.index)
    va = cross_sectional_ls(SP.VALIDATION.slice(sig), SP.VALIDATION.slice(close_wide),
                            top_n=5, hold_bars=M[30], cost_bps_per_side=CANON_COST)
    return va.metrics["mean_gross_per_trade_bps"]


def _print_and_log_report(df, head, rev_gross, start, end, log):
    name, va, sweep, summ = head
    abn_gross = va.metrics["mean_gross_per_trade_bps"]
    sweep_tbl = "\n".join(f"  {int(c):>3} bps/side -> net Sharpe {sh:>7.2f}, net {nb:>7.2f} bps/trade"
                          for c, (sh, nb) in sweep.items())
    pos_ic = int((df["train_ic_tstat"].abs() > 2).sum())
    best = df.sort_values("val_sharpe_net", ascending=False).iloc[0]

    print("\n=== HEAD-TO-HEAD gross edge per trade (validation) ===")
    print(f"  abnormal-volume continuation (headline '{name}'): {abn_gross:+.2f} bps/trade")
    print(f"  short-term reversal (family 1 headline):          {rev_gross:+.2f} bps/trade")
    larger = "LARGER" if abn_gross > rev_gross else "NOT larger"
    print(f"  -> abnormal-volume gross edge is {larger} than reversal.\n")
    print("Cost sweep (headline config, validation):")
    print(sweep_tbl)

    body = _md(df, best, head, rev_gross, pos_ic, start, end)
    R.write_research_log(FAMILY, body)
    print(f"\nLogged {len(df)} trials (now {log.count()} total). "
          f"results/alpha_tests/{FAMILY}.csv | research_logs/{FAMILY}.md. LOCK-BOX NOT TOUCHED.")


def _md(df, best, head, rev_gross, pos_ic, start, end) -> str:
    name, va, sweep, summ = head
    abn_gross = va.metrics["mean_gross_per_trade_bps"]
    tbl = df[["config", "top_n", "hold_bars", "train_ic_mean", "train_ic_tstat",
              "bucket_monotonicity", "val_gross_bps_per_trade", "val_net_bps_per_trade",
              "val_sharpe_net", "val_trades_per_day"]].to_markdown(index=False, floatfmt=".3f")
    sweep_tbl = "\n".join(f"| {int(c)} | {sh:.2f} | {nb:.2f} |" for c, (sh, nb) in sweep.items())
    larger = "LARGER than" if abn_gross > rev_gross else "NOT larger than"
    return f"""# Abnormal-volume continuation (5-min) — research log

**Family:** Stock alpha 2/8 · **Universe:** 31 survivor mega-caps (survivorship-biased) ·
**Window:** {start} → {end} (train+val) · **Lock-box:** SEALED.

## Hypothesis
Abnormal same-time-of-day volume + strong residual-vs-SPY move ⇒ continuation over 30–120 min.
Long up-move+high-vol, short down-move+high-vol, dollar-neutral top/bottom-N.

## Main question — gross edge vs short-term reversal
- Abnormal-volume continuation headline (**{name}**): **{abn_gross:+.2f} bps/trade gross**.
- Short-term reversal (family 1 headline): **{rev_gross:+.2f} bps/trade gross**.
- ⇒ Abnormal-volume gross edge is **{larger}** reversal.

## Results (train + validation only; lock-box untouched)

{tbl}

- Configs with |train IC t| > 2: **{pos_ic} / {len(df)}** (t inflated by overlapping windows — directional only).
- Best by **validation** net Sharpe: **{best['config']}** (net {best['val_net_bps_per_trade']:.2f} bps/trade).

## Cost sweep (headline config, validation)

| cost bps/side | net Sharpe | net bps/trade |
|---:|---:|---:|
{sweep_tbl}

## Critical verdict
The decisive numbers are **gross bps/trade vs cost bps/trade**. Round-trip cost = 2× the per-side
bps (the $1-gross book is opened and closed). If gross bps/trade < round-trip cost at a realistic
1–5 bps/side, the alpha is **taker-dead** regardless of how clean the IC looks. Diagnose which of:
(1) alpha too small, (2) too much turnover, (3) too few trades, (4) bad universe,
(5) weak IEX data, (6) needs maker/limit execution, (7) needs broader universe — see the summary
the runner prints. Reject if weak; a positive IC with sub-cost gross is *mechanism, not money*.

## Reading it honestly
- Read **net bps/trade**, not net Sharpe magnitude (near-constant cost drag inflates the ratio).
- IC t-stats overstate significance (overlapping forward windows).
- 31 survivors, one 2020–23 regime: not a deployable alpha. Engine + measurement only.
- Lock-box not touched; validation is the selection number.
"""


if __name__ == "__main__":
    main()
