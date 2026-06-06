"""
Crypto Alpha Family 1 — BTC→alt LEAD-LAG (perp 5-min), IC-first. PHASE-2 SMOKE TEST.

DISCIPLINE: only 30 days of data exist. This is a PIPELINE smoke test + a sign-plausibility
check, NOT an alpha claim. No OOS split (30d is too short to split meaningfully); every number
is full-sample and labelled as such. Reject configs with too few trades, or where the backtest
disagrees with IC, or where one coin/day carries it.

Hypothesis: BTC leads ETH/alts. When BTC moves and an alt lags, the alt catches up over 15–60m.
Three experimental arms:
  - CATCH-UP (lag_gap = BTC_ret − alt_ret): long laggards, short leaders. Configs 1–8.
  - CONTINUATION control (own momentum, +alt_ret): long recent winners. Config 9.
  - REVERSAL control (own, −alt_ret): fade strongest movers. Config 10.
(NB: "reversal relative to BTC" is mathematically identical to catch-up, so the reversal control
 uses the alt's OWN return to stay distinct.)

Universe: 9 traded USDT-perp alts (BTC excluded — it is the leader). Config 7 also excludes ETH
(it is a secondary leader there). Backtest: continuous 24/7 grid, dollar-neutral top/bottom-3,
entry next bar, non-overlapping. IC measured vs the BTC-RELATIVE forward return (the catch-up
question: does the laggard out-return BTC next?). Crypto taker cost sweep 0/2/5/10 bps/side.

Run: python scripts/run_crypto_lead_lag.py
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
from mft.research.crypto_panel import PERP_1M, load_crypto_panel  # noqa: E402
from mft.research.features import volume_zscore  # noqa: E402
from mft.research.targets import forward_return_panel  # noqa: E402
from mft.research.xs_backtest import cross_sectional_ls  # noqa: E402
from mft.validation.metrics import sharpe  # noqa: E402

FAMILY = "crypto_btc_lead_lag"
BTC = "BTCUSDT"
ALL_SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
ALTS = [s for s in ALL_SYMS if s != BTC]                       # 9 traded alts
ALTS_NO_ETH = [s for s in ALTS if s != "ETHUSDT"]             # config 7
M = {5: 1, 15: 3, 30: 6, 60: 12, 120: 24}
HORIZONS = [M[15], M[30], M[60], M[120]]
COST_SWEEP = (0.0, 2.0, 5.0, 10.0)
CANON_COST = 5.0   # crypto taker (~Binance 4–5 bps/side + slippage)
TOP_N, MIN_TRADES = 3, 60


def ret_panel(close: pd.DataFrame, bars: int) -> pd.DataFrame:
    return close.pct_change(bars)


def btc_relative_fwd(close: pd.DataFrame, h: int) -> pd.DataFrame:
    """Forward return of each name MINUS BTC's forward return (continuous, entry lag 1)."""
    fwd = forward_return_panel(close, h, entry_lag=1, intraday_only=False)
    return fwd.sub(fwd[BTC], axis=0)


def cost_sweep(gross: pd.Series, ppy: float) -> dict:
    return {c: (sharpe(gross - 2.0 * c * 1e-4, periods_per_year=ppy),
                float((gross - 2.0 * c * 1e-4).mean() * 1e4)) for c in COST_SWEEP}


def main() -> None:
    if not (PERP_1M / f"{BTC}.parquet").exists():
        print("No crypto perp data. Run scripts/ingest_crypto.py first.")
        return

    panel = load_crypto_panel(ALL_SYMS, freq="5min", market=BTC, spot_dir=PERP_1M)
    close = panel.to_wide("close", symbols=ALL_SYMS)
    print(f"Loaded {len(panel.symbols)+1} perp symbols, {len(close)} 5m bars, "
          f"{close.index[0]} -> {close.index[-1]} (24/7, full-sample SMOKE TEST)\n")

    r = {h: ret_panel(close, h) for h in (M[5], M[15], M[30])}
    btc = {h: r[h][BTC] for h in r}
    eth15 = r[M[15]]["ETHUSDT"]
    volz = pd.DataFrame({s: volume_zscore(panel.bars[s]["volume"], 60) for s in ALTS}).reindex(close.index)

    def lag_gap(h, syms):
        return r[h][syms].rsub(btc[h], axis=0)   # BTC_ret - alt_ret, per timestamp

    # ── pre-registered configs ────────────────────────────────────────────────
    # (name, signal_wide, traded_syms, hold_bars, arm)
    g15, g30 = lambda s: lag_gap(M[15], s), lambda s: lag_gap(M[30], s)
    btc_strong15 = btc[M[15]].abs() > 0.0025
    btc_strong30 = btc[M[30]].abs() > 0.0050
    comb = (btc[M[15]] + eth15) / 2.0

    configs = [
        ("1 catchup g15 |btc15|>.25% h15", g15(ALTS).where(btc_strong15, axis=0), ALTS, M[15], "catchup"),
        ("2 catchup g15 |btc15|>.25% h30", g15(ALTS).where(btc_strong15, axis=0), ALTS, M[30], "catchup"),
        ("3 catchup g30 |btc30|>.5% h30",  g30(ALTS).where(btc_strong30, axis=0), ALTS, M[30], "catchup"),
        ("4 catchup g30 |btc30|>.5% h60",  g30(ALTS).where(btc_strong30, axis=0), ALTS, M[60], "catchup"),
        ("5 rank lag_gap15 h30",           g15(ALTS),                              ALTS, M[30], "catchup"),
        ("6 rank lag_gap30 h60",           g30(ALTS),                              ALTS, M[60], "catchup"),
        ("7 BTC+ETH leader g15 h30",       r[M[15]][ALTS_NO_ETH].rsub(comb, axis=0), ALTS_NO_ETH, M[30], "catchup"),
        ("8 lag_gap30 volz>1 h60",         g30(ALTS).where(volz > 1.0),            ALTS, M[60], "catchup"),
        ("9 continuation +altret15 h30",   r[M[15]][ALTS],                         ALTS, M[30], "continuation"),
        ("10 reversal -altret15 h30",      -r[M[15]][ALTS],                        ALTS, M[30], "reversal"),
    ]

    log, rows = TrialLog(), []
    print(f"{'config':<33}{'hold':>5}{'arm':>13} | {'IC':>7}{'ICt':>5}{'mono':>5} | "
          f"{'grBps':>6}{'ntBps@5':>8} | {'Sh':>6}{'longL':>6}{'shortL':>7}{'trd/d':>6}{'nTr':>5} trial")
    print("-" * 124)

    head = None
    for name, sig, traded, hold, arm in configs:
        sig = sig[traded]
        rel = btc_relative_fwd(close, hold)[traded]     # BTC-relative target for IC
        ic = L.ic_series(sig, rel, method="spearman", min_names=TOP_N)
        summ = L.ic_summary(ic)
        buckets = L.bucket_returns(sig, rel, n_buckets=5, min_names=TOP_N)
        mono = L.bucket_monotonicity(buckets)

        bt = cross_sectional_ls(sig, close[traded], top_n=TOP_N, hold_bars=hold,
                                cost_bps_per_side=CANON_COST, continuous=True)
        m = bt.metrics
        tid = log.log(strategy="CryptoBTCLeadLagXS", asset_universe=traded,
                      params={"config": name, "arm": arm, "hold_bars": hold, "top_n": TOP_N,
                              "cost_bps_per_side": CANON_COST, "freq": "5min", "venue": "binance-perp"},
                      data_window=f"{close.index[0].date()}:{close.index[-1].date()}",
                      is_sharpe=float(m["sharpe"]), oos_sharpe=None,
                      max_dd=float(m["max_drawdown"]), turnover=float(bt.trades_per_day),
                      notes=(f"CRYPTO 30d SMOKE (no OOS, full-sample, 24/7); arm={arm}; "
                             f"IC_btcrel={summ['ic_mean']:.4f} t={summ['ic_tstat']:.2f} "
                             f"grossBps/trade={m.get('mean_gross_per_trade_bps', float('nan')):.2f} nTr={m['n_trades']}"))

        too_few = m["n_trades"] < MIN_TRADES
        rows.append({
            "family": FAMILY, "config": name, "arm": arm, "hold_bars": hold,
            "n_traded": len(traded),
            "ic_btcrel_mean": summ["ic_mean"], "ic_btcrel_tstat": summ["ic_tstat"],
            "bucket_monotonicity": mono,
            "gross_bps_per_trade": m["mean_gross_per_trade_bps"],
            "net_bps_per_trade_5bps": m["mean_net_per_trade_bps"],
            "sharpe_net_5bps": m["sharpe"], "sharpe_gross": m["sharpe_gross"],
            "max_dd": m["max_drawdown"], "long_leg_bps": m["long_leg_bps_per_trade"],
            "short_leg_bps": m["short_leg_bps_per_trade"], "win_rate": m["hit_rate"],
            "avg_win_bps": m["avg_win_bps"], "avg_loss_bps": m["avg_loss_bps"],
            "trades_per_day": bt.trades_per_day, "n_trades": m["n_trades"],
            "too_few_trades": too_few, "trial_id": tid,
        })

        print(f"{name:<33}{hold:>5}{arm:>13} | {summ['ic_mean']:>7.4f}{summ['ic_tstat']:>5.1f}{mono:>5.2f} | "
              f"{m['mean_gross_per_trade_bps']:>6.2f}{m['mean_net_per_trade_bps']:>8.2f} | "
              f"{m['sharpe']:>6.1f}{m['long_leg_bps_per_trade']:>6.2f}{m['short_leg_bps_per_trade']:>7.2f}"
              f"{bt.trades_per_day:>6.1f}{m['n_trades']:>5} {tid}"
              + ("  <-few" if too_few else ""))

        if name.startswith("5 "):
            R.plot_buckets(buckets, FAMILY, "btcrel")
            R.plot_equity(bt.equity(), FAMILY, "cfg5")
            head = (name, sig, traded, hold, bt, ic)

    df = R.save_results(rows, FAMILY)
    R.update_leaderboard(df[["family", "config", "arm", "hold_bars", "ic_btcrel_tstat",
                             "gross_bps_per_trade", "sharpe_net_5bps", "n_trades", "trial_id"]]
                         .rename(columns={"sharpe_net_5bps": "val_sharpe"}), rank_col="val_sharpe")
    _verdict(df, head, close, log)


def _verdict(df, head, close, log):
    name, sig, traded, hold, bt, ic = head
    m = bt.metrics
    sweep = cost_sweep(bt.gross, bt.periods_per_year)
    # alpha decay vs BTC-relative target across horizons (headline signal)
    decay = {h: L.ic_summary(L.ic_series(sig, btc_relative_fwd(close, h)[traded],
                                         method="spearman", min_names=TOP_N))["ic_mean"]
             for h in HORIZONS}
    R.plot_alpha_decay(pd.Series(decay, name="ic_by_horizon"), FAMILY, "btcrel")

    # crypto breakdowns on the headline net series
    net = bt.net
    by_hour = (net.groupby(net.index.hour).mean() * 1e4)
    wk = net.groupby(net.index.weekday >= 5).mean() * 1e4
    by_day = net.groupby(net.index.date).sum()
    top_day_share = float(by_day.max() / by_day.sum()) if by_day.sum() > 0 else float("nan")
    ic_by_day = ic.groupby(ic.index.date).mean()

    catch = df[df["arm"] == "catchup"]
    pos_ic = int((catch["ic_btcrel_mean"] > 0).sum())
    survivors = df[(~df["too_few_trades"]) & (df["net_bps_per_trade_5bps"] > 0) & (df["ic_btcrel_mean"] > 0)]

    print("\n=== SMOKE-TEST VERDICT — BTC lead-lag (30 days, full-sample, NOT an alpha claim) ===")
    print(f"Sign: catch-up arms with POSITIVE BTC-relative IC: {pos_ic}/{len(catch)} "
          f"(positive IC ⇒ laggards DO catch up; negative ⇒ they continue/lead).")
    print(f"Headline (cfg5 rank lag_gap15 h30): IC {df.loc[df.config==name,'ic_btcrel_mean'].iloc[0]:+.4f} "
          f"(t {df.loc[df.config==name,'ic_btcrel_tstat'].iloc[0]:+.1f}), gross {m['mean_gross_per_trade_bps']:+.2f} bps/trade, "
          f"net@5bps {m['mean_net_per_trade_bps']:+.2f}.")
    print("Cost sweep (cfg5): " + ", ".join(f"{int(c)}bps:{nb:+.2f}" for c, (sh, nb) in sweep.items()) + " net bps/trade")
    print(f"Robustness: net-positive & IC-positive & enough-trades configs: {len(survivors)}/{len(df)}.")
    print(f"Concentration: top single-day share of cfg5 net P&L = {top_day_share:.1%} "
          f"(high ⇒ one day carries it). Weekday {wk.get(False, float('nan')):+.2f} vs weekend "
          f"{wk.get(True, float('nan')):+.2f} bps/trade.")

    R.write_research_log(FAMILY, _md(df, head, sweep, decay, by_hour, wk, top_day_share, ic_by_day, survivors))
    print(f"\nLogged {len(df)} trials (now {log.count()} total). "
          f"results/alpha_tests/{FAMILY}.csv | research_logs/{FAMILY}.md  (no alpha claimed; awaiting approval)")


def _md(df, head, sweep, decay, by_hour, wk, top_day_share, ic_by_day, survivors) -> str:
    name, sig, traded, hold, bt, ic = head
    m = bt.metrics
    tbl = df[["config", "arm", "hold_bars", "ic_btcrel_mean", "ic_btcrel_tstat", "bucket_monotonicity",
              "gross_bps_per_trade", "net_bps_per_trade_5bps", "sharpe_net_5bps",
              "long_leg_bps", "short_leg_bps", "n_trades", "too_few_trades"]].to_markdown(index=False, floatfmt=".3f")
    sweep_tbl = "\n".join(f"| {int(c)} | {sh:.2f} | {nb:.2f} |" for c, (sh, nb) in sweep.items())
    decay_tbl = ", ".join(f"{h*5}m:{v:+.4f}" for h, v in decay.items())
    return f"""# Crypto BTC→alt lead-lag (perp 5-min) — research log  ·  PHASE-2 SMOKE TEST

**Family:** Crypto alpha 1 · **Universe:** 9 USDT-perp alts (BTC = leader, excluded) ·
**Data:** 30 days, {bt.net.index[0].date()} → {bt.net.index[-1].date()}, 24/7 continuous ·
**No OOS split** (30d too short) · **Cost:** crypto taker, canonical {CANON_COST} bps/side.

> ⚠️ **NOT an alpha claim.** 30 days = one regime, one market move. This proves the crypto harness
> works end-to-end (24/7 backtest, IC, costs) and checks whether the lead-lag SIGN is plausible.

## Hypothesis & arms
BTC leads alts; laggards catch up (catch-up = long lag_gap). Controls: continuation (own momentum)
and reversal (own, fade movers). IC measured vs **BTC-relative** forward return.

## Results (full-sample, 30 days)

{tbl}

## Headline cfg5 (rank lag_gap15, hold 30m)
- IC(BTC-rel) {df.loc[df.config==name,'ic_btcrel_mean'].iloc[0]:+.4f} (t {df.loc[df.config==name,'ic_btcrel_tstat'].iloc[0]:+.1f});
  gross **{m['mean_gross_per_trade_bps']:+.2f}** bps/trade; net@5bps **{m['mean_net_per_trade_bps']:+.2f}**.
- Alpha decay (IC vs BTC-rel by horizon): {decay_tbl}
- Long leg {m['long_leg_bps_per_trade']:+.2f} bps, short leg {m['short_leg_bps_per_trade']:+.2f} bps; win rate {m['hit_rate']:.2f}.
- **Concentration:** top single-day share of net P&L = {top_day_share:.1%}. Weekday {wk.get(False, float('nan')):+.2f}
  vs weekend {wk.get(True, float('nan')):+.2f} bps/trade.

### Cost sweep (cfg5)
| cost bps/side | net Sharpe | net bps/trade |
|---:|---:|---:|
{sweep_tbl}

## Critical interpretation (per the rules)
- **Sign:** positive catch-up IC ⇒ laggards catch up (hypothesis holds); negative ⇒ they continue.
- Net-positive **and** IC-positive **and** enough-trades configs: **{len(survivors)} / {len(df)}**.
- **Do NOT claim alpha from 30 days.** Reject too-few-trade configs; reject backtest-positive-but-IC-
  negative configs; flag if one day/coin carries it (see concentration).
- Gross bps/trade here are on **30 days of crypto** — not comparable in significance to the (rejected)
  stock families' multi-year samples; a bigger gross number over 30d is NOT a stronger result.

## Next
If the sign is plausible and IC-supported, the binding need is **more data** (backfill 6–12 months
via `ingest_crypto.py --days 365`) and an **OOS split** before any alpha claim.
"""


if __name__ == "__main__":
    main()
