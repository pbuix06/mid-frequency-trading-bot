"""
Crypto CROSS-SECTIONAL MOMENTUM — 365-day OOS / regime validation. Fully pre-registered.

A genuinely different hypothesis from everything tested: trend/momentum at SLOWER horizons (4h/8h/24h),
not reversal. Rank the 10 USDT-perp majors by recent return; LONG winners, SHORT losers, dollar-neutral.
Continuous 24/7, entry next bar. Pre-registered split train60/val20/lockbox20 (lock-box reported, not tuned).
Funding is CONTEXT ONLY (attribution), never the signal/filter. NO OI, NO basis, NO tuning, NO new configs.

Decision rules (pre-registered): CANDIDATE only if val AND lock-box net@5bps > 0, IC > 0, and not
one-coin/one-month; BETA if it only works in up months with both legs positive; REJECT if net dies after cost.

Run AFTER the 365d backfill. Output: crypto_cross_sectional_momentum.{csv,md} + figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from mft.monitoring.trial_log import TrialLog  # noqa: E402
from mft.research import report as R  # noqa: E402
from mft.research import signal_lab as L  # noqa: E402
from mft.research.crypto_eval import attribute_trades, regimes  # noqa: E402
from mft.research.crypto_panel import (  # noqa: E402
    FUNDING_DIR,
    PERP_1M,
    cumulative_funding_series,
    load_crypto_panel,
    load_funding,
    map_to_bars,
)
from mft.research.features import volume_zscore  # noqa: E402
from mft.research.funding_backtest import funding_ls_backtest  # noqa: E402
from mft.research.targets import forward_return_panel  # noqa: E402
from mft.validation.metrics import sharpe  # noqa: E402

FAMILY = "crypto_cross_sectional_momentum"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
BTC = "BTCUSDT"
LB = {"4h": 48, "8h": 96, "24h": 288}
COST_SWEEP = (0.0, 2.0, 5.0, 10.0)
CANON, MIN_TRADES, DAY = 5.0, 40, 288
HEADLINE = "3 raw mom lb24 h24"


def cost_sweep(total: pd.Series, ppy: float) -> dict:
    return {c: (sharpe(total - 2 * c * 1e-4, periods_per_year=ppy),
                float((total - 2 * c * 1e-4).mean() * 1e4)) for c in COST_SWEEP}


def split_metrics(sig, close, cum, top_n, hold, mask):
    bt = funding_ls_backtest(sig[mask], close[mask], cum[mask], top_n=top_n,
                             hold_bars=hold, cost_bps_per_side=CANON)
    m = bt.metrics
    return {"n": m["n_trades"], "price": m.get("price_bps_per_trade", np.nan),
            "fund": m.get("funding_bps_per_trade", np.nan), "total": m.get("total_gross_bps_per_trade", np.nan),
            "net5": m.get("net_bps_per_trade", np.nan),
            "price_net5": (m.get("price_bps_per_trade", np.nan) - m.get("cost_per_trade_bps", np.nan)),
            "sharpe": m.get("sharpe", np.nan), "dd": m.get("max_drawdown", np.nan)}, bt


def main() -> None:
    if not (PERP_1M / f"{BTC}.parquet").exists():
        print("No perp data. Run scripts/ingest_crypto.py --days 365 first.")
        return
    perp = load_crypto_panel(SYMS, freq="5min", market=BTC, spot_dir=PERP_1M)
    close = perp.to_wide("close", SYMS)
    vol = perp.to_wide("volume", SYMS)
    idx = close.index
    span = (idx[-1] - idx[0]).days

    cum = pd.DataFrame({s: map_to_bars(cumulative_funding_series(load_funding(s, FUNDING_DIR)), idx)
                        for s in SYMS}).reindex(idx)   # funding CONTEXT only

    # ── momentum features ──────────────────────────────────────────────────────
    ret = {lb: close.pct_change(lb) for lb in LB.values()}
    btc_ret = {lb: ret[lb][BTC] for lb in ret}
    vol24 = close.pct_change().rolling(LB["24h"]).std()
    volz4 = pd.DataFrame({s: volume_zscore(perp.bars[s]["volume"], LB["4h"]) for s in SYMS}).reindex(idx)
    relvol24 = (vol / vol.rolling(LB["24h"]).mean() - 1.0)

    mom_raw = {lb: ret[lb] for lb in ret}
    mom_rel = {lb: ret[lb].sub(btc_ret[lb], axis=0) for lb in ret}

    configs = [
        ("1 raw mom lb4 h4",    mom_raw[LB["4h"]],  LB["4h"], "raw"),
        ("2 raw mom lb8 h8",    mom_raw[LB["8h"]],  LB["8h"], "raw"),
        ("3 raw mom lb24 h24",  mom_raw[LB["24h"]], LB["24h"], "raw"),
        ("4 btcrel lb4 h4",     mom_rel[LB["4h"]],  LB["4h"], "btcrel"),
        ("5 btcrel lb8 h8",     mom_rel[LB["8h"]],  LB["8h"], "btcrel"),
        ("6 btcrel lb24 h24",   mom_rel[LB["24h"]], LB["24h"], "btcrel"),
        ("7 voladj r8/v24 h8",  mom_raw[LB["8h"]] / vol24,  LB["8h"], "voladj"),
        ("8 voladj r24/v24 h24", mom_raw[LB["24h"]] / vol24, LB["24h"], "voladj"),
        ("9 mom8+volz4>0 h8",   mom_raw[LB["8h"]].where(volz4 > 0),  LB["8h"], "volconf"),
        ("10 mom24+relvol>0 h24", mom_raw[LB["24h"]].where(relvol24 > 0), LB["24h"], "volconf"),
    ]
    reg = regimes(close)
    N = len(idx)
    tr_end, va_end = idx[int(0.6 * N)], idx[int(0.8 * N)]
    train, val, lock = idx <= tr_end, (idx > tr_end) & (idx <= va_end), idx > va_end
    print(f"Loaded {len(SYMS)} perp majors, {N} 5m bars, {idx[0].date()}->{idx[-1].date()} ({span}d). "
          f"CROSS-SECTIONAL MOMENTUM (long winners/short losers).")
    print(f"PRE-REGISTERED SPLIT: TRAIN ..{tr_end.date()} | VAL ..{va_end.date()} | "
          f"LOCKBOX ..{idx[-1].date()} (reported, not tuned)\n")
    print(f"{'config':<22}{'h':>4} | {'IC':>7} | {'TRnet5':>7}{'VAnet5':>7}{'LKnet5':>7} | "
          f"{'up':>6}{'down':>6}{'side':>6} | {'price':>6}{'fund':>6} | decision")
    print("-" * 116)

    log, rows = TrialLog(), []
    head = None
    for name, sig, hold, fam in configs:
        s = sig[SYMS]
        ic = L.ic_summary(L.ic_series(s, forward_return_panel(close[SYMS], hold, intraday_only=False),
                                      method="spearman", min_names=4))["ic_mean"]
        trm, _ = split_metrics(s, close[SYMS], cum[SYMS], 3, hold, train)
        vam, _ = split_metrics(s, close[SYMS], cum[SYMS], 3, hold, val)
        lkm, _ = split_metrics(s, close[SYMS], cum[SYMS], 3, hold, lock)
        full, bt_full = split_metrics(s, close[SYMS], cum[SYMS], 3, hold, np.ones(N, bool))

        net = bt_full.net
        by_trend = (net.groupby(reg["trend"].reindex(net.index)).sum() * 1e4)
        up, down, side = (float(by_trend.get(k, 0.0)) for k in ("up", "down", "sideways"))

        oos_ok = (vam["net5"] > 0) and (lkm["net5"] > 0) and (vam["n"] >= MIN_TRADES) and (ic > 0)
        pos_reg = sum(x > 0 for x in (up, down, side))
        if (vam["net5"] <= 0) or (lkm["net5"] <= 0):
            decision = "REJECT (dies OOS)"
        elif (up > 0) and (down <= 0) and (side <= 0):
            decision = "BETA (up-only)"
        elif oos_ok and pos_reg >= 2:
            decision = "CANDIDATE"
        else:
            decision = "weak/mixed"
        if vam["n"] < MIN_TRADES:
            decision += " [few]"

        tid = log.log(strategy="CryptoXSMomentum", asset_universe=SYMS,
                      params={"config": name, "family": fam, "hold_bars": hold, "top_n": 3,
                              "cost_bps_per_side": CANON, "split": "train60/val20/lock20"},
                      data_window=f"{idx[0].date()}:{idx[-1].date()}",
                      is_sharpe=float(trm["sharpe"]) if pd.notna(trm["sharpe"]) else 0.0,
                      oos_sharpe=float(lkm["sharpe"]) if pd.notna(lkm["sharpe"]) else None,
                      max_dd=float(lkm["dd"]) if pd.notna(lkm["dd"]) else None, turnover=None,
                      notes=(f"CRYPTO XS MOMENTUM {span}d OOS; IC={ic:.4f}; VALnet5={vam['net5']:+.2f} "
                             f"LOCKnet5={lkm['net5']:+.2f}; trend up{up:+.0f}/down{down:+.0f}/side{side:+.0f}; "
                             f"price={full['price']:+.2f} fund={full['fund']:+.2f}; {decision}"))

        rows.append({"config": name, "family": fam, "hold_bars": hold, "ic_full": ic,
                     "train_net5": trm["net5"], "val_net5": vam["net5"], "lock_net5": lkm["net5"],
                     "val_n": vam["n"], "lock_n": lkm["n"], "full_price": full["price"],
                     "full_fund": full["fund"], "full_total": full["total"], "full_net5": full["net5"],
                     "trend_up_bps": up, "trend_down_bps": down, "trend_side_bps": side,
                     "decision": decision, "trial_id": tid})
        print(f"{name:<22}{hold//DAY*24 if hold >= DAY else hold//12:>3}h | {ic:>7.4f} | "
              f"{trm['net5']:>7.2f}{vam['net5']:>7.2f}{lkm['net5']:>7.2f} | "
              f"{up:>6.0f}{down:>6.0f}{side:>6.0f} | {full['price']:>6.2f}{full['fund']:>6.2f} | {decision}")
        if name == HEADLINE:
            head = (name, s, hold, bt_full)

    df = pd.DataFrame(rows)
    R.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(R.RESULTS_DIR / f"{FAMILY}.csv", index=False)
    _verdict(df, head, close, cum, reg, log, span, tr_end, va_end, idx)


def _verdict(df, head, close, cum, reg, log, span, tr_end, va_end, idx):
    name, sig, hold, bt = head
    at = attribute_trades(sig, close[SYMS], cum[SYMS], 3, hold)
    at["trend"] = reg["trend"].reindex(at["ts"]).values
    at["btc_dir"] = np.where(at["btc_fwd"] > 0, "btc_up", "btc_down")
    at["month"] = pd.PeriodIndex(at["ts"], freq="M").astype(str)
    by_sym = (at.groupby("sym")["total"].sum() * 1e4).round(1).sort_values()
    by_month = (at.groupby("month")["total"].sum() * 1e4).round(1)
    by_btcdir = (at.groupby("btc_dir")["total"].sum() * 1e4).round(1)
    legs = at.groupby("side")["fwd"].mean() * 1e4
    ic_m = L.ic_series(sig, forward_return_panel(close[SYMS], hold, intraday_only=False), min_names=4)
    ic_by_month = ic_m.groupby(pd.PeriodIndex(ic_m.index, freq="M").astype(str)).mean()
    ppy = len(bt.total) / max((bt.total.index[-1] - bt.total.index[0]).days / 365.25, 1e-9)
    sweep = cost_sweep(bt.total, ppy)

    R.plot_equity(bt.equity(), FAMILY, "cfg3")
    _bar(by_month, "cfg3 monthly net (bps)", f"{FAMILY}_monthly")
    _bar((at.groupby("trend")["total"].sum() * 1e4).round(1), "cfg3 net by BTC trend (bps)", f"{FAMILY}_regime")
    _bar(by_sym, "cfg3 net by symbol (bps)", f"{FAMILY}_symbol")
    _bar(ic_by_month, "cfg3 IC by month", f"{FAMILY}_ic_month")
    _bar(pd.Series({c: nb for c, (sh, nb) in sweep.items()}), "cfg3 net bps/trade vs cost/side", f"{FAMILY}_cost")

    cand = (df["decision"].str.startswith("CANDIDATE")).sum()
    beta = (df["decision"].str.startswith("BETA")).sum()
    rej = (df["decision"].str.startswith("REJECT")).sum()
    raw_ic = df[df.family == "raw"]["ic_full"].mean()
    rel_ic = df[df.family == "btcrel"]["ic_full"].mean()

    print(f"\n=== VERDICT — crypto XS momentum, {span}d ===")
    print(f"IC sign: raw-mom mean IC {raw_ic:+.4f} | btc-relative mean IC {rel_ic:+.4f} "
          f"(positive => momentum/continuation; negative => reversal even at this horizon)")
    print(f"Headline cfg3 (raw 24h): trend UP {df.loc[df.config==name,'trend_up_bps'].iloc[0]:+.0f} | "
          f"DOWN {df.loc[df.config==name,'trend_down_bps'].iloc[0]:+.0f} | SIDE {df.loc[df.config==name,'trend_side_bps'].iloc[0]:+.0f} bps")
    print(f"  legs (mean fwd): long {legs.get('long', float('nan')):+.1f} bps, short {legs.get('short', float('nan')):+.1f} bps; "
          f"btc-dir: {by_btcdir.to_dict()}")
    print("  cost sweep (total net bps/trade): " + ", ".join(f"{int(c)}bps:{nb:+.2f}" for c, (sh, nb) in sweep.items()))
    print(f"  price {bt.metrics['price_bps_per_trade']:+.2f} vs funding {bt.metrics['funding_bps_per_trade']:+.2f} bps/trade")
    print(f"  per-symbol net: {by_sym.to_dict()}")
    print(f"Decisions: CANDIDATE {cand} | BETA {beta} | REJECT {rej} | other {len(df)-cand-beta-rej}")

    R.write_research_log(FAMILY, _md(df, head, by_sym, by_month, by_btcdir, legs, sweep, raw_ic, rel_ic, span, tr_end, va_end, idx))
    print(f"\nLogged {len(df)} trials (now {log.count()} total). results/alpha_tests/{FAMILY}.csv | "
          f"research_logs/{FAMILY}.md  (no alpha claimed; lock-box separate)")


def _bar(series, title, fname):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(series.index.astype(str), series.values,
           color=["tab:green" if v > 0 else "tab:red" for v in series.values])
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(title)
    plt.xticks(rotation=60, ha="right")
    fig.tight_layout()
    fig.savefig(R.FIG_DIR / f"{fname}.png", dpi=110)
    plt.close(fig)


def _md(df, head, by_sym, by_month, by_btcdir, legs, sweep, raw_ic, rel_ic, span, tr_end, va_end, idx):
    name, sig, hold, bt = head
    m = bt.metrics
    tbl = df[["config", "family", "hold_bars", "ic_full", "train_net5", "val_net5", "lock_net5",
              "trend_up_bps", "trend_down_bps", "trend_side_bps", "full_price", "full_fund",
              "val_n", "decision"]].to_markdown(index=False, floatfmt=".3f")
    sweep_tbl = "\n".join(f"| {int(c)} | {sh:.2f} | {nb:.2f} |" for c, (sh, nb) in sweep.items())
    return f"""# Crypto cross-sectional MOMENTUM — {span}-day OOS / regime validation

**Universe:** 10 USDT perp majors · **Span:** {span}d ({idx[0].date()}→{idx[-1].date()}) · **Split:** train 60% /
val 20% / **lock-box 20% (reported, not tuned)** · **Cost:** taker {CANON} bps/side · **Funding = context only.**
Long winners / short losers, dollar-neutral, continuous 24/7, entry next bar. Pre-registered; no tuning.

## Per-config: IC, OOS net@5 by split, PnL by BTC trend regime, price vs funding

{tbl}

> `*_net5` = net bps/trade after 10 bps round-trip **including funding drag**. `full_price`/`full_fund` split
> the full-sample edge into momentum price PnL vs funding carry (long winners usually PAY funding).

## Headline cfg3 (raw momentum, 24h lookback/hold)
- **Trend regime:** up {df.loc[df.config==name,'trend_up_bps'].iloc[0]:+.0f} | down {df.loc[df.config==name,'trend_down_bps'].iloc[0]:+.0f} |
  sideways {df.loc[df.config==name,'trend_side_bps'].iloc[0]:+.0f} bps.
- **BTC direction during trade:** {by_btcdir.to_dict()} — both-legs-positive + only-when-btc-up ⇒ long beta.
- **Legs (mean fwd):** long {legs.get('long', float('nan')):+.1f} bps, short {legs.get('short', float('nan')):+.1f} bps.
- **Price vs funding:** price {m['price_bps_per_trade']:+.2f} + funding {m['funding_bps_per_trade']:+.2f} bps/trade
  (funding is a {'drag' if m['funding_bps_per_trade'] < 0 else 'tailwind'} for momentum).
- **Per-symbol net (bps):** {by_sym.to_dict()}
- **Per-month net (bps):** {by_month.to_dict()}

### Cost sweep (cfg3 total net)
| cost bps/side | net Sharpe | net bps/trade |
|---:|---:|---:|
{sweep_tbl}

## The six questions
1. **Survive OOS / lock-box?** `val_net5` and `lock_net5` columns.
2. **Survive 5 bps/side?** same columns (net is after 10 bps round-trip).
3. **IC-supported?** `ic_full`; raw-mom mean IC {raw_ic:+.4f}, btc-relative {rel_ic:+.4f}
   (positive = momentum; **negative = reversal even here**).
4. **Alpha or beta?** trend-regime row + btc-direction + legs above.
5. **Robust across symbols/months/regimes?** per-symbol / per-month tables + figures.
6. **Candidate / reject / pause?** decisions column; counts in runner output.

## Figures
`{FAMILY}_cfg3_equity_val.png`, `_monthly.png`, `_regime.png`, `_symbol.png`, `_ic_month.png`, `_cost.png`.

**No alpha claimed. Lock-box reported separately, not used to choose anything.**
"""


if __name__ == "__main__":
    main()
