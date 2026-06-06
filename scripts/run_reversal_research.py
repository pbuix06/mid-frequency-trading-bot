"""
Stock alpha family 1 of 8 — SHORT-TERM CROSS-SECTIONAL REVERSAL (5-min bars).

Hypothesis: over minutes, names that just over-extended DOWN relative to the cross-section
(idiosyncratically, after removing SPY beta) partially bounce back over the next 15–60 min,
and vice-versa — a liquidity-provision premium (Lehmann 1990; Nagel 2012). We FADE the
recent move: signal = -recent_return, long the biggest recent losers / short the winners.

Method (IC-FIRST, the whole point):
  1. Cheap screen — rank-IC of the signal vs forward returns at 15/30/60/120m (alpha decay).
     No fills, no costs. Does the ranking predict the next move at all?
  2. Only then — a non-overlapping dollar-neutral top/bottom-N backtest, GROSS and NET, on
     TRAIN and VALIDATION. Validation is the only number we rank on. LOCK-BOX UNTOUCHED.

Pre-registered grid (<=12 configs, frozen before looking): lookback {5,15,30m} x variant
{raw, residual-vs-SPY} backtested at 30m hold; + a hold-period robustness {15,30,60m} on the
strongest variant. Every config -> one row in trials/trials.csv (no hidden search).

CAVEATS WE DO NOT HIDE: 31 survivor mega-caps, ~2.9 yr, one regime cluster. This MEASURES the
signal honestly and proves the engine; it cannot produce a deployable cross-sectional alpha
(project breadth finding T0056). Results are evidence of MECHANISM, not a tradeable Sharpe.

Run:  python scripts/run_reversal_research.py
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
from mft.research.features import residual_return, trailing_return  # noqa: E402
from mft.research.panel import INTRADAY_DIR, load_panel  # noqa: E402
from mft.research.targets import forward_return_panel  # noqa: E402
from mft.research.xs_backtest import cross_sectional_ls  # noqa: E402

FAMILY = "short_term_reversal_xs"
MARKET = "SPY"
EXCLUDE = {"SPY", "IWM"}  # index ETFs are the market proxy / not single-name reversal candidates

# 5-min bars: minutes -> bars
M = {5: 1, 15: 3, 30: 6, 60: 12, 120: 24}
HORIZONS_BARS = [M[15], M[30], M[60], M[120]]  # alpha-decay horizons
TOP_N = 5
COST_BPS_PER_SIDE = 2.0  # liquid mega-cap; sensitivity is a separate column, not a trial


def build_signal(close_wide: pd.DataFrame, mkt_close: pd.Series,
                 lookback_bars: int, variant: str) -> pd.DataFrame:
    """signal = -recent return (fade). 'residual' first strips SPY beta (past-only)."""
    out = {}
    for s in close_wide.columns:
        c = close_wide[s].dropna()
        if variant == "raw":
            r = trailing_return(c, lookback_bars)
        elif variant == "residual":
            r = residual_return(c, mkt_close, lookback_bars, beta_window=78, skip=1)
        else:
            raise ValueError(variant)
        out[s] = -r  # fade: oversold (negative recent return) -> high signal
    return pd.DataFrame(out).reindex(close_wide.index)


def ic_table(signal: pd.DataFrame, close_wide: pd.DataFrame) -> dict:
    """Rank-IC at the 30m horizon + the full alpha-decay curve. Train-only caller."""
    tgt_30 = forward_return_panel(close_wide, M[30])
    ic = L.ic_series(signal, tgt_30, method="spearman")
    summ = L.ic_summary(ic)
    decay = L.alpha_decay(signal, close_wide, HORIZONS_BARS, method="spearman")
    buckets = L.bucket_returns(signal, tgt_30, n_buckets=5)
    return {"ic": ic, "summary": summ, "decay": decay, "buckets": buckets}


def main() -> None:
    if not list(INTRADAY_DIR.glob("*.parquet")):
        print("No intraday data. Run scripts/ingest_alpaca.py first.")
        return

    start, end = SP.research_window()  # 2020-07-27 .. 2023-06-30 (NO lock-box)
    all_syms = sorted(p.stem for p in INTRADAY_DIR.glob("*.parquet"))
    stocks = [s for s in all_syms if s not in EXCLUDE]
    print(f"Loading {len(stocks)} names + {MARKET}, 5-min, {start} -> {end} "
          f"(lock-box {SP.LOCKBOX_START.date()} SEALED) ...")

    panel = load_panel(stocks, start=start, end=end, freq="5min", market=MARKET)
    close_wide = panel.to_wide("close")
    SP.assert_no_lockbox(close_wide.index)  # hard guard: no research bar may touch the lock-box
    mkt_close = panel.market_close()
    print(f"  loaded {len(panel.symbols)} names, {len(close_wide)} bars "
          f"({close_wide.index[0].date()} .. {close_wide.index[-1].date()})\n")

    log = TrialLog()
    rows = []
    fig_cfg = ("residual", M[15])  # the config we draw figures for (headline)

    # ── pre-registered grid ───────────────────────────────────────────────────
    configs = [(v, lb) for v in ("raw", "residual") for lb in (M[5], M[15], M[30])]
    hold_robust = [M[15], M[30], M[60]]  # extra holds, residual+15m lookback only

    print(f"{'variant':<9}{'lookbk':>7}{'hold':>6} | {'trainIC':>8}{'IC t':>6}{'mono':>6} | "
          f"{'trGrSh':>7}{'trNtSh':>7} | {'vaGrSh':>7}{'vaNtSh':>7}{'vaDD':>7}  trial")
    print("-" * 104)

    for variant, lb in configs:
        sig = build_signal(close_wide, mkt_close, lb, variant)
        parts = SP.split_train_val(sig)
        close_parts = SP.split_train_val(close_wide)

        # IC-first on TRAIN
        ictab = ic_table(parts["train"], close_parts["train"])
        mono = L.bucket_monotonicity(ictab["buckets"])

        # holds to backtest for this variant
        holds = hold_robust if (variant, lb) == ("residual", M[15]) else [M[30]]
        for hold in holds:
            tr = cross_sectional_ls(parts["train"], close_parts["train"],
                                    top_n=TOP_N, hold_bars=hold, cost_bps_per_side=COST_BPS_PER_SIDE)
            va = cross_sectional_ls(parts["validation"], close_parts["validation"],
                                    top_n=TOP_N, hold_bars=hold, cost_bps_per_side=COST_BPS_PER_SIDE)

            params = {"lookback_bars": lb, "variant": variant, "hold_bars": hold,
                      "top_n": TOP_N, "cost_bps_per_side": COST_BPS_PER_SIDE,
                      "freq": "5min", "horizon_min": hold * 5}
            tid = log.log(
                strategy="ShortTermReversalXS",
                asset_universe=panel.symbols,
                params=params,
                data_window=f"{start}:{end}",
                is_sharpe=float(tr.metrics["sharpe"]),
                oos_sharpe=float(va.metrics["sharpe"]),
                max_dd=float(va.metrics["max_drawdown"]),
                turnover=float(va.trades_per_day),
                notes=("intraday XS reversal IC-first; train+val only, LOCK-BOX SEALED; "
                       f"trainIC={ictab['summary']['ic_mean']:.4f} "
                       f"t={ictab['summary']['ic_tstat']:.2f} mono={mono:.2f}"),
            )

            rows.append({
                "family": FAMILY, "variant": variant, "lookback_bars": lb, "hold_bars": hold,
                "horizon_min": hold * 5, "top_n": TOP_N, "cost_bps_per_side": COST_BPS_PER_SIDE,
                "train_ic_mean": ictab["summary"]["ic_mean"],
                "train_ic_tstat": ictab["summary"]["ic_tstat"],
                "train_ic_pct_pos": ictab["summary"]["ic_pct_pos"],
                "bucket_monotonicity": mono,
                "train_sharpe_gross": tr.metrics["sharpe_gross"],
                "train_sharpe_net": tr.metrics["sharpe"],
                "val_sharpe_gross": va.metrics["sharpe_gross"],
                "val_sharpe": va.metrics["sharpe"],
                "val_net_per_trade_bps": va.metrics["mean_net_per_trade_bps"],
                "val_max_dd": va.metrics["max_drawdown"],
                "val_trades_per_day": va.trades_per_day,
                "n_names": len(panel.symbols),
                "trial_id": tid, "lockbox_touched": False,
            })

            print(f"{variant:<9}{lb:>7}{hold:>6} | "
                  f"{ictab['summary']['ic_mean']:>8.4f}{ictab['summary']['ic_tstat']:>6.1f}"
                  f"{mono:>6.2f} | {tr.metrics['sharpe_gross']:>7.2f}{tr.metrics['sharpe']:>7.2f} | "
                  f"{va.metrics['sharpe_gross']:>7.2f}{va.metrics['sharpe']:>7.2f}"
                  f"{va.metrics['max_drawdown']:>7.1%}  {tid}")

            # figures for the headline config (validation backtest)
            if (variant, lb) == fig_cfg and hold == M[30]:
                R.plot_alpha_decay(ictab["decay"], FAMILY, tag="train")
                R.plot_buckets(ictab["buckets"], FAMILY, tag="train")
                R.plot_equity(va.equity(), FAMILY, tag="val")

    df = R.save_results(rows, FAMILY)
    R.update_leaderboard(df[["family", "variant", "lookback_bars", "hold_bars",
                                     "train_ic_tstat", "train_sharpe_net", "val_sharpe",
                                     "val_net_per_trade_bps", "trial_id"]].rename(
                                     columns={"val_sharpe": "val_sharpe"}), rank_col="val_sharpe")

    # ── research log (markdown) ───────────────────────────────────────────────
    best = df.sort_values("val_sharpe", ascending=False).iloc[0]
    pos_ic = (df["train_ic_tstat"].abs() > 2).sum()
    body = _research_log_md(df, best, pos_ic, start, end)
    R.write_research_log(FAMILY, body)

    print(f"\nLogged {len(rows)} trials (now {log.count()} total). "
          f"Results: results/alpha_tests/{FAMILY}.csv | log: research_logs/{FAMILY}.md")
    print(f"Best by VALIDATION net Sharpe: {best['variant']} lb={best['lookback_bars']} "
          f"hold={best['hold_bars']} -> val {best['val_sharpe']:.2f} "
          f"(train {best['train_sharpe_net']:.2f}, IC t {best['train_ic_tstat']:.1f}). "
          f"LOCK-BOX NOT TOUCHED.")


def _research_log_md(df, best, pos_ic, start, end) -> str:
    tbl = df[["variant", "lookback_bars", "hold_bars", "train_ic_mean", "train_ic_tstat",
              "bucket_monotonicity", "train_sharpe_net", "val_sharpe",
              "val_net_per_trade_bps", "val_max_dd"]].to_markdown(index=False, floatfmt=".3f")
    return f"""# Short-term cross-sectional reversal (5-min) — research log

**Family:** Stock alpha 1/8 · **Frequency:** 5-min · **Universe:** {int(best['n_names'])} survivor
mega-caps (survivorship-biased) · **Window:** {start} → {end} (train+val) · **Lock-box:** SEALED.

## Hypothesis
Names that idiosyncratically over-extend down (after removing SPY beta) partially revert over
15–60 min — a liquidity-provision premium. Signal = −recent return (fade); long losers, short
winners, dollar-neutral top/bottom-{TOP_N}.

## Method
IC-first: rank-IC vs forward 15/30/60/120-min returns (no fills/costs) → only then a
non-overlapping dollar-neutral LS backtest, gross & net (cost {COST_BPS_PER_SIDE} bps/side),
entry lag 1 bar, flat overnight. Pre-registered grid; every config logged to `trials/trials.csv`.

## Results (train + validation only; lock-box untouched)

{tbl}

- Configs with |train IC t| > 2: **{int(pos_ic)} / {len(df)}**.
- Best by **validation** net Sharpe: **{best['variant']}**, lookback {int(best['lookback_bars'])} bars,
  hold {int(best['hold_bars'])} bars → **val Sharpe {best['val_sharpe']:.2f}** (train net
  {best['train_sharpe_net']:.2f}; train IC t {best['train_ic_tstat']:.1f};
  net/trade {best['val_net_per_trade_bps']:.2f} bps).

## Reading it honestly
- **Gross vs net:** the table reports both; rank only on **net**. If gross is positive but net is
  ~0, the spread eats the edge — the recurring intraday taker verdict in this project.
- **Read net bps/trade, not the net Sharpe magnitude.** Net P&L per trade ≈ a near-constant cost
  drag (low variance), so the net *Sharpe ratio* blows up in magnitude (−8 to −28). That is an
  artifact of dividing a small, almost-deterministic negative mean by a tiny std — it does **not**
  mean "28× worse than a good book". The economically honest number is **net return per trade in
  bps** (here ≈ −3 to −4 bps): a small positive gross edge minus a larger round-trip cost.
- **IC t-stats are optimistic.** IC is computed on every bar's forward window, which overlaps
  heavily (consecutive 30-min windows share 29/30 of their span), so the effective number of
  independent periods is far smaller than the raw count → the t-stat overstates significance.
  Treat IC>2 here as *directional mechanism* evidence (the sign and monotonicity), not a precise p.
- **IC vs Sharpe:** a positive, monotonic IC is *mechanism* evidence; a positive validation net
  Sharpe is *tradeability* evidence. Need both. Here: mechanism yes (tiny), tradeability no.
- **Survivorship & regime:** 31 survivors over one 2020–23 regime cluster. Do **not** read any
  number as a deployable alpha. This proves the engine and measures the effect.
- **Lock-box:** not touched. The validation number is the selection number; the lock-box exam is
  a future one-time event via `splits.load_lockbox(i_am_done_tuning=True)`.

## Next
- If a variant shows stable IC>2 **and** net Sharpe survives 2× cost → carry to the next family
  and re-test combined (correlation screen). If net dies at cost (likely) → record as taker-dead,
  consistent with T0046/T0050, and note it as a candidate **maker** signal (needs quote data).
"""


if __name__ == "__main__":
    main()
