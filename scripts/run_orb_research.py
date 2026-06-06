"""
Stock alpha family 3 of 8 — OPENING-RANGE BREAKOUT (ORB), cross-sectional, 5-min, IC-first.

Structurally different from families 1–2 (both small mean-reversion): this tests directional
CONTINUATION. Signal = signed breakout strength beyond the opening range; the top/bottom-N harness
longs the strongest up-breakouts and shorts the strongest down-breakouts (held non-overlapping).

NOTE: this is the CROSS-SECTIONAL reading of ORB. The project already tested single-name EVENT
ORB as a taker (T0052 0.43 / T0053 0.84 optimistic / T0054 0.025 realistic — taker-dead). This asks
the new question: is breakout direction a cross-sectionally rankable continuation signal at 5-min?

top_n=3 (breakout signals are SPARSE — few names break out at once; top5 would thin trades to
noise). min_names=6 genuine breakouts required per rebalance, else the bar is skipped.

Main questions answered in the printed verdict: (1) gross bps/trade vs reversal; (2) survives 1–2
bps/side? (3) consistent across OR windows or cherry-picked? (4) does volume confirmation help?
(5) IC/bucket-supported or backtest artifact? (6) long-leg vs short-leg driven?

Caveats unchanged: 31 survivor mega-caps, ~2.9 yr, one regime, IEX partial tape. Lock-box SEALED.
Run: python scripts/run_orb_research.py
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
from mft.research.breakout import orb_signal_panel  # noqa: E402
from mft.research.features import residual_return, volume_zscore_tod  # noqa: E402
from mft.research.panel import INTRADAY_DIR, load_panel  # noqa: E402
from mft.research.xs_backtest import cross_sectional_ls  # noqa: E402
from mft.validation.metrics import sharpe  # noqa: E402

FAMILY = "opening_range_breakout"
MARKET, EXCLUDE = "SPY", {"SPY", "IWM"}
M = {5: 1, 15: 3, 30: 6, 60: 12, 120: 24}
HORIZONS = [M[15], M[30], M[60], M[120]]
COST_SWEEP = (0.0, 1.0, 2.0, 5.0, 10.0)
CANON_COST, TOP_N = 2.0, 3
MIN_TRADES = 60  # below this the config is too-few-trades -> reject as unreliable

# (name, or_bars, buffer_bps, spy_adjusted, vol_threshold, hold_bars)
CONFIGS = [
    ("1 OR15 b0 h30",        M[15], 0.0,  False, None, M[30]),
    ("2 OR15 b5 h30",        M[15], 5.0,  False, None, M[30]),
    ("3 OR15 b10 vz1 h60",   M[15], 10.0, False, 1.0,  M[60]),
    ("4 OR30 b0 h60",        M[30], 0.0,  False, None, M[60]),
    ("5 OR30 b5 vz1 h60",    M[30], 5.0,  False, 1.0,  M[60]),
    ("6 OR30 b10 vz1.5 h120", M[30], 10.0, False, 1.5, M[120]),
    ("7 OR60 b5 h120",       M[60], 5.0,  False, None, M[120]),
    ("8 OR15 SPYadj vz1 h60", M[15], 0.0, True,  1.0,  M[60]),
    ("9 OR30 SPYadj vz1 h120", M[30], 0.0, True, 1.0,  M[120]),
]
HEADLINE = "4 OR30 b0 h60"


def cost_sweep(gross: pd.Series, ppy: float) -> dict:
    return {c: (sharpe(gross - 2.0 * c * 1e-4, periods_per_year=ppy),
                float((gross - 2.0 * c * 1e-4).mean() * 1e4)) for c in COST_SWEEP}


def main() -> None:
    if not list(INTRADAY_DIR.glob("*.parquet")):
        print("No intraday data. Run scripts/ingest_alpaca.py first.")
        return

    start, end = SP.research_window()
    stocks = sorted(p.stem for p in INTRADAY_DIR.glob("*.parquet") if p.stem not in EXCLUDE)
    panel = load_panel(stocks, start=start, end=end, freq="5min", market=MARKET)
    close_wide = panel.to_wide("close")
    SP.assert_no_lockbox(close_wide.index)
    mkt = panel.market_close()
    volz = pd.DataFrame({s: volume_zscore_tod(panel.bars[s], 20) for s in panel.symbols}
                        ).reindex(close_wide.index)
    print(f"Loaded {len(panel.symbols)} names, {len(close_wide)} bars, {start}->{end} "
          f"(lock-box {SP.LOCKBOX_START.date()} SEALED). top_n={TOP_N}\n")

    log, rows = TrialLog(), []
    print(f"{'config':<23}{'hold':>5} | {'trIC':>7}{'ICt':>5}{'mono':>5} | "
          f"{'grBps':>6}{'netBps':>7} | {'vGrSh':>6}{'vNtSh':>7} | "
          f"{'longL':>6}{'shortL':>7}{'trd/d':>6}{'nTr':>5} trial")
    print("-" * 116)

    head = None
    for name, or_bars, buf, spyadj, vthr, hold in CONFIGS:
        # raw signed breakout strength; apply same-time-of-day volume gate post-hoc
        sig = orb_signal_panel(panel.bars, panel.symbols, close_wide.index, mkt,
                               or_bars, buf, spyadj, vol_threshold=None)
        if vthr is not None:
            sig = sig.where(volz > vthr)

        sp = SP.split_train_val(sig)
        cp = SP.split_train_val(close_wide)

        from mft.research.targets import forward_return_panel
        tgt_tr = forward_return_panel(cp["train"], hold)
        ic = L.ic_series(sp["train"], tgt_tr, method="spearman", min_names=TOP_N)
        summ = L.ic_summary(ic)
        buckets = L.bucket_returns(sp["train"], tgt_tr, n_buckets=5, min_names=TOP_N)
        mono = L.bucket_monotonicity(buckets)

        tr = cross_sectional_ls(sp["train"], cp["train"], top_n=TOP_N, hold_bars=hold,
                                cost_bps_per_side=CANON_COST)
        va = cross_sectional_ls(sp["validation"], cp["validation"], top_n=TOP_N, hold_bars=hold,
                                cost_bps_per_side=CANON_COST)
        vm = va.metrics

        tid = log.log(strategy="OpeningRangeBreakoutXS", asset_universe=panel.symbols,
                      params={"config": name, "or_bars": or_bars, "buffer_bps": buf,
                              "spy_adjusted": spyadj, "vol_threshold": vthr, "hold_bars": hold,
                              "top_n": TOP_N, "cost_bps_per_side": CANON_COST, "freq": "5min"},
                      data_window=f"{start}:{end}",
                      is_sharpe=float(tr.metrics["sharpe"]), oos_sharpe=float(vm["sharpe"]),
                      max_dd=float(vm["max_drawdown"]), turnover=float(va.trades_per_day),
                      notes=(f"ORB cross-sectional continuation, IC-first; LOCKBOX SEALED; "
                             f"trainIC={summ['ic_mean']:.4f} t={summ['ic_tstat']:.2f} "
                             f"grossBps/trade={vm.get('mean_gross_per_trade_bps', float('nan')):.2f} "
                             f"nTrades={vm['n_trades']}"))

        too_few = vm["n_trades"] < MIN_TRADES
        rows.append({
            "family": FAMILY, "config": name, "or_bars": or_bars, "buffer_bps": buf,
            "spy_adjusted": spyadj, "vol_threshold": vthr, "hold_bars": hold,
            "train_ic_mean": summ["ic_mean"], "train_ic_tstat": summ["ic_tstat"],
            "bucket_monotonicity": mono,
            "val_gross_bps_per_trade": vm["mean_gross_per_trade_bps"],
            "val_net_bps_per_trade": vm["mean_net_per_trade_bps"],
            "val_sharpe_gross": vm["sharpe_gross"], "val_sharpe_net": vm["sharpe"],
            "val_long_leg_bps": vm["long_leg_bps_per_trade"],
            "val_short_leg_bps": vm["short_leg_bps_per_trade"],
            "val_win_rate": vm["hit_rate"], "val_avg_win_bps": vm["avg_win_bps"],
            "val_avg_loss_bps": vm["avg_loss_bps"], "val_max_dd": vm["max_drawdown"],
            "val_trades_per_day": va.trades_per_day, "val_n_trades": vm["n_trades"],
            "too_few_trades": too_few, "trial_id": tid, "lockbox_touched": False,
        })

        flag = "  <-too few" if too_few else ""
        print(f"{name:<23}{hold:>5} | {summ['ic_mean']:>7.4f}{summ['ic_tstat']:>5.1f}{mono:>5.2f} | "
              f"{vm['mean_gross_per_trade_bps']:>6.2f}{vm['mean_net_per_trade_bps']:>7.2f} | "
              f"{vm['sharpe_gross']:>6.2f}{vm['sharpe']:>7.1f} | "
              f"{vm['long_leg_bps_per_trade']:>6.2f}{vm['short_leg_bps_per_trade']:>7.2f}"
              f"{va.trades_per_day:>6.1f}{vm['n_trades']:>5} {tid}{flag}")

        if name == HEADLINE:
            R.plot_alpha_decay(L.alpha_decay(sp["train"], cp["train"], HORIZONS), FAMILY, "train")
            R.plot_buckets(buckets, FAMILY, "train")
            R.plot_equity(va.equity(), FAMILY, "val")
            head = (name, va, cost_sweep(va.gross, va.periods_per_year), summ)

    df = R.save_results(rows, FAMILY)
    R.update_leaderboard(df[["family", "config", "hold_bars", "train_ic_tstat",
                             "val_gross_bps_per_trade", "val_sharpe_net", "val_n_trades", "trial_id"]]
                         .rename(columns={"val_sharpe_net": "val_sharpe"}), rank_col="val_sharpe")

    rev_gross = _reversal_gross_bps(close_wide, mkt)
    _verdict(df, head, rev_gross, start, end, log)


def _reversal_gross_bps(close_wide, mkt) -> float:
    """Family-1 reversal headline gross bps/trade at the SAME top_n=3 (fair comparison)."""
    sig = pd.DataFrame({c: -residual_return(close_wide[c].dropna(), mkt, M[15])
                        for c in close_wide.columns}).reindex(close_wide.index)
    va = cross_sectional_ls(SP.VALIDATION.slice(sig), SP.VALIDATION.slice(close_wide),
                            top_n=TOP_N, hold_bars=M[30], cost_bps_per_side=CANON_COST)
    return va.metrics["mean_gross_per_trade_bps"]


def _tod_breakdown(net: pd.Series) -> pd.Series:
    et_hour = net.index.tz_convert("America/New_York").hour
    return (net.groupby(et_hour).mean() * 1e4).rename("net_bps_by_ET_hour")


def _verdict(df, head, rev_gross, start, end, log):
    name, va, sweep, summ = head
    abn = va.metrics
    g = abn["mean_gross_per_trade_bps"]
    # cross-OR consistency: no-vol baselines at OR 15/30/60
    base = df[df["vol_threshold"].isna()].set_index("config")["val_gross_bps_per_trade"]
    # volume help/hurt: OR30 no-vol (4) vs OR30 vol_z>1 (5)
    novol = df.loc[df["config"] == "4 OR30 b0 h60", "val_gross_bps_per_trade"].iloc[0]
    withvol = df.loc[df["config"] == "5 OR30 b5 vz1 h60", "val_gross_bps_per_trade"].iloc[0]
    survivors = df[(~df["too_few_trades"]) & (df["val_net_bps_per_trade"] > 0)]

    print("\n=== VERDICT — opening-range breakout (cross-sectional) ===")
    print(f"Q1 gross/trade vs reversal: ORB headline '{name}' {g:+.2f} bps vs reversal "
          f"{rev_gross:+.2f} bps -> ORB is {'LARGER' if g > rev_gross else 'NOT larger'}.")
    print("Q2 survives cost: headline cost sweep -> "
          + ", ".join(f"{int(c)}bps:{nb:+.2f}" for c, (sh, nb) in sweep.items()) + " (net bps/trade)")
    print("Q3 OR-window consistency (no-vol gross bps/trade): "
          + ", ".join(f"{k.split()[0]}={v:+.2f}" for k, v in base.items()))
    print(f"Q4 volume help/hurt (OR30 gross bps/trade): no-vol {novol:+.2f} vs vol_z>1 {withvol:+.2f} "
          f"-> volume {'HELPS' if withvol > novol else 'HURTS/!help'}")
    head_mono = df.loc[df["config"] == name, "bucket_monotonicity"].iloc[0]
    print(f"Q5 IC support (headline): mean IC {summ['ic_mean']:+.4f}, t {summ['ic_tstat']:+.1f}, "
          f"bucket mono {head_mono:+.2f} (see bucket fig)")
    print(f"Q6 leg attribution (headline): long-leg {abn['long_leg_bps_per_trade']:+.2f} bps, "
          f"short-leg {abn['short_leg_bps_per_trade']:+.2f} bps "
          f"(book wants long high, short low)")
    print(f"\nNet-positive & enough-trades configs: {len(survivors)} of {len(df)}.")
    print("Net bps/trade by ET hour (headline):")
    print("  " + "  ".join(f"{h}:00={v:+.1f}" for h, v in _tod_breakdown(va.net).items()))

    R.write_research_log(FAMILY, _md(df, head, rev_gross, survivors, start, end))
    print(f"\nLogged {len(df)} trials (now {log.count()} total). "
          f"results/alpha_tests/{FAMILY}.csv | research_logs/{FAMILY}.md. LOCK-BOX NOT TOUCHED.")


def _md(df, head, rev_gross, survivors, start, end) -> str:
    name, va, sweep, summ = head
    g = va.metrics["mean_gross_per_trade_bps"]
    tbl = df[["config", "hold_bars", "train_ic_mean", "train_ic_tstat", "bucket_monotonicity",
              "val_gross_bps_per_trade", "val_net_bps_per_trade", "val_sharpe_net",
              "val_long_leg_bps", "val_short_leg_bps", "val_trades_per_day", "val_n_trades",
              "too_few_trades"]].to_markdown(index=False, floatfmt=".3f")
    sweep_tbl = "\n".join(f"| {int(c)} | {sh:.2f} | {nb:.2f} |" for c, (sh, nb) in sweep.items())
    tod = "  ".join(f"{h}:00={v:+.1f}" for h, v in _tod_breakdown(va.net).items())
    return f"""# Opening-range breakout (cross-sectional, 5-min) — research log

**Family:** Stock alpha 3/8 · **Universe:** 31 survivor mega-caps (survivorship-biased) ·
**Window:** {start} → {end} (train+val) · **Lock-box:** SEALED · **top_n:** {TOP_N}.

## Hypothesis
Names breaking beyond their opening range with strength/volume CONTINUE over 30–120 min. Long
strongest up-breakouts, short strongest down-breakouts, dollar-neutral. Cross-sectional reading
(distinct from the single-name event ORB already tested taker-dead in T0052–T0054).

## Results (train + validation; lock-box untouched)

{tbl}

> The short leg is the *weakest* breakouts among genuine breakouts; on up-trending bars those can
> still be up-breakouts (not literal down-breakouts) — read the leg columns with that in mind.

## Cost sweep (headline {name}, validation)

| cost bps/side | net Sharpe | net bps/trade |
|---:|---:|---:|
{sweep_tbl}

Net bps/trade by ET hour (headline): {tod}

## Answers to the 6 questions
1. **Gross/trade vs reversal:** ORB headline **{g:+.2f}** bps vs reversal **{rev_gross:+.2f}** bps.
2. **Survives 1–2 bps/side?** See cost sweep — positive only if net stays >0 at 1–2 bps.
3. **Consistent across OR windows?** Compare the no-vol OR15/30/60 rows — a real edge is a plateau,
   not one setting.
4. **Volume confirmation help/hurt?** Compare config 4 (no vol) vs 5 (vol_z>1) at OR30.
5. **IC/bucket-supported?** Headline mean IC {summ['ic_mean']:+.4f}, t {summ['ic_tstat']:+.1f} — a
   positive monotonic IC is required, else the backtest is an artifact.
6. **Leg attribution:** long-leg {va.metrics['long_leg_bps_per_trade']:+.2f} bps, short-leg
   {va.metrics['short_leg_bps_per_trade']:+.2f} bps.

**Net-positive & enough-trades configs: {len(survivors)} / {len(df)}.**

## Reading it honestly
- Read **net bps/trade**, not net Sharpe magnitude. IC t-stats overstate significance (overlap).
- Reject configs with **too few trades** (n_trades < {MIN_TRADES}), no IC support, no OR-window
  plateau, or one lucky month.
- 31 survivors, one 2020–23 regime, IEX partial tape. Not a deployable alpha. Lock-box untouched.
"""


if __name__ == "__main__":
    main()
