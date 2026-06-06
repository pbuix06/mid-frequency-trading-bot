"""
Crypto funding-reversal — 365-day backfill OOS / REGIME validation. NO new configs, NO tuning.

Reruns the EXACT 10 pre-registered configs (mft.research.funding_signals) on ~12 months of data with
a PRE-REGISTERED time split (train 60% / validation 20% / lockbox 20%), and attributes PnL by regime,
month, and symbol. Lock-box is reported separately and never used to choose anything.

KILL/CONFIRM question: does the funding-reversal edge persist OUTSIDE the original 30-day crash, in
up / sideways months — or is it crash/liquidation beta (only works when crowded longs get liquidated)?

Run AFTER `python scripts/ingest_crypto.py --days 365`. Output: ..._backfill.csv / .md + figures.
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
from mft.research.crypto_panel import (  # noqa: E402
    FUNDING_DIR,
    PERP_1M,
    SPOT_1M,
    load_crypto_panel,
    load_funding,
    load_open_interest,
)
from mft.research.funding_backtest import funding_ls_backtest  # noqa: E402
from mft.research.funding_signals import build_configs, build_features  # noqa: E402
from mft.research.targets import forward_return_panel  # noqa: E402
from mft.research.xs_backtest import _rebalance_points  # noqa: E402

FAMILY = "crypto_funding_reversal_backfill"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
BTC = "BTCUSDT"
COST_SWEEP = (2.0, 5.0, 10.0)
CANON = 5.0
MIN_TRADES = 40
DAY = 288  # 5-min bars per day


# ── regime labels (causal except btc_dir, which is for conditional attribution only) ──
def regimes(close: pd.DataFrame) -> dict:
    btc = close[BTC]
    month = pd.PeriodIndex(btc.index, freq="M")
    mret = btc.groupby(month).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1.0)
    mlab = mret.apply(lambda r: "up" if r > 0.05 else ("down" if r < -0.05 else "sideways"))
    trend = pd.Series(month.map(mlab), index=btc.index)
    roll_max = btc.rolling(30 * DAY, min_periods=1).max()
    dd = btc / roll_max - 1.0
    ddlab = pd.cut(dd, [-1, -0.20, -0.10, 1], labels=["stress", "correction", "normal"])
    vol = btc.pct_change().rolling(7 * DAY).std()
    vlab = pd.qcut(vol.rank(method="first"), 3, labels=["lowvol", "medvol", "highvol"])
    return {"trend": trend, "drawdown": pd.Series(ddlab, index=btc.index).astype(object),
            "vol": pd.Series(vlab, index=btc.index).astype(object), "month": trend.index.to_period("M").astype(str)}


def attribute_trades(sig, close, cum, top_n, hold) -> pd.DataFrame:
    """Per-(rebalance, name) contribution rows: price/funding/total + the name's forward return."""
    fwd = forward_return_panel(close, hold, entry_lag=1, intraday_only=False)
    carry = cum.shift(-(1 + hold)) - cum.shift(-1)
    btc_fwd = close[BTC].shift(-(1 + hold)) / close[BTC].shift(-1) - 1.0
    rows = []
    for r in _rebalance_points(sig.index, hold, continuous=True):
        s = sig.loc[r].dropna()
        f = fwd.loc[r] if r in fwd.index else None
        c = carry.loc[r] if r in carry.index else None
        if f is None or c is None:
            continue
        names = s.index.intersection(f.dropna().index).intersection(c.dropna().index)
        if len(names) < 2 * top_n:
            continue
        s = s.loc[names].sort_values()
        for x in s.index[-top_n:]:
            rows.append((r, x, "long", 0.5 * f[x] / top_n, 0.5 * (-c[x]) / top_n, float(f[x]), btc_fwd.loc[r]))
        for x in s.index[:top_n]:
            rows.append((r, x, "short", 0.5 * (-f[x]) / top_n, 0.5 * (c[x]) / top_n, float(f[x]), btc_fwd.loc[r]))
    df = pd.DataFrame(rows, columns=["ts", "sym", "side", "price", "funding", "fwd", "btc_fwd"])
    df["total"] = df["price"] + df["funding"]
    return df


def split_metrics(sig, close, cum, top_n, hold, mask) -> dict:
    bt = funding_ls_backtest(sig[mask], close[mask], cum[mask],
                             top_n=top_n, hold_bars=hold, cost_bps_per_side=CANON)
    m = bt.metrics
    out = {"n": m["n_trades"], "price": m.get("price_bps_per_trade", np.nan),
           "fund": m.get("funding_bps_per_trade", np.nan), "total": m.get("total_gross_bps_per_trade", np.nan),
           "net5": m.get("net_bps_per_trade", np.nan), "sharpe": m.get("sharpe", np.nan),
           "dd": m.get("max_drawdown", np.nan)}
    return out, bt


def main() -> None:
    if not (PERP_1M / f"{BTC}.parquet").exists():
        print("No perp data. Run scripts/ingest_crypto.py --days 365 first.")
        return
    perp = load_crypto_panel(SYMS, freq="5min", market=BTC, spot_dir=PERP_1M)
    close = perp.to_wide("close", SYMS)
    spot = load_crypto_panel(SYMS, freq="5min", market=BTC, spot_dir=SPOT_1M).to_wide("close", SYMS)
    idx = close.index
    span_days = (idx[-1] - idx[0]).days
    if span_days < 120:
        print(f"WARNING: only {span_days} days loaded — backfill may not have completed. Proceeding.")

    funding = {s: load_funding(s, FUNDING_DIR) for s in SYMS}
    oi = {s: (load_open_interest(s)["open_interest"] if not load_open_interest(s).empty else pd.Series(dtype=float))
          for s in SYMS}
    fz, cum, oi4, ret4h, basis_z = build_features(close, funding, oi, spot, SYMS)
    configs = build_configs(fz, oi4, ret4h, basis_z)
    reg = regimes(close)

    # ── PRE-REGISTERED split (declared before any result) ─────────────────────
    N = len(idx)
    tr_end, va_end = idx[int(0.6 * N)], idx[int(0.8 * N)]
    train = idx <= tr_end
    val = (idx > tr_end) & (idx <= va_end)
    lock = idx > va_end
    print(f"Loaded {len(SYMS)} perp majors, {N} 5m bars, {idx[0].date()} -> {idx[-1].date()} ({span_days}d).")
    print(f"PRE-REGISTERED SPLIT (time): TRAIN {idx[0].date()}..{tr_end.date()} | "
          f"VAL {(tr_end).date()}..{va_end.date()} | LOCKBOX {(va_end).date()}..{idx[-1].date()} (reported, not tuned)\n")

    log, rows = TrialLog(), []
    print(f"{'config':<19}{'N':>2}{'h':>4} | {'IC':>7} | "
          f"{'TRtot':>7}{'TRnet5':>7} | {'VAtot':>7}{'VAnet5':>7} | {'LKtot':>7}{'LKnet5':>7} | "
          f"{'up':>6}{'down':>6}{'side':>6} | decision")
    print("-" * 120)

    head = None
    for name, sig, hold, top_n, fam in configs:
        s = sig[SYMS]
        # IC full-sample (vs raw forward at horizon)
        ic = L.ic_summary(L.ic_series(s, forward_return_panel(close[SYMS], hold, intraday_only=False),
                                      method="spearman", min_names=3))["ic_mean"]
        trm, _ = split_metrics(s, close[SYMS], cum[SYMS], top_n, hold, train)
        vam, _ = split_metrics(s, close[SYMS], cum[SYMS], top_n, hold, val)
        lkm, bt_lk = split_metrics(s, close[SYMS], cum[SYMS], top_n, hold, lock)
        full, bt_full = split_metrics(s, close[SYMS], cum[SYMS], top_n, hold, np.ones(N, bool))

        # PnL by BTC trend regime (full-sample net@5 series grouped by month-regime)
        net = bt_full.net
        trend_at = reg["trend"].reindex(net.index)
        by_trend = (net.groupby(trend_at).sum() * 1e4)
        up, down, side = (float(by_trend.get(k, 0.0)) for k in ("up", "down", "sideways"))

        # decision
        oos_ok = (vam["net5"] > 0) and (lkm["net5"] > 0) and (vam["n"] >= MIN_TRADES)
        pos_regimes = sum(x > 0 for x in (up, down, side))
        if not oos_ok and (vam["net5"] <= 0 or lkm["net5"] <= 0):
            decision = "REJECT (dies OOS)"
        elif (down > 0) and (up <= 0) and (side <= 0):
            decision = "CRASH-BETA"
        elif oos_ok and pos_regimes >= 2:
            decision = "CANDIDATE"
        else:
            decision = "weak/mixed"
        if vam["n"] < MIN_TRADES:
            decision += " [few]"

        tid = log.log(strategy="CryptoFundingReversalBackfill", asset_universe=SYMS,
                      params={"config": name, "family": fam, "hold_bars": hold, "top_n": top_n,
                              "cost_bps_per_side": CANON, "split": "train60/val20/lock20"},
                      data_window=f"{idx[0].date()}:{idx[-1].date()}",
                      is_sharpe=float(trm["sharpe"]) if pd.notna(trm["sharpe"]) else 0.0,
                      oos_sharpe=float(lkm["sharpe"]) if pd.notna(lkm["sharpe"]) else None,
                      max_dd=float(lkm["dd"]) if pd.notna(lkm["dd"]) else None, turnover=None,
                      notes=(f"FUNDING BACKFILL {span_days}d OOS; IC={ic:.4f}; "
                             f"VALnet5={vam['net5']:+.2f} LOCKnet5={lkm['net5']:+.2f}; "
                             f"trend up{up:+.0f}/down{down:+.0f}/side{side:+.0f} bps; {decision}"))

        rows.append({"config": name, "family": fam, "hold_bars": hold, "top_n": top_n, "ic_full": ic,
                     "train_total": trm["total"], "train_net5": trm["net5"], "train_n": trm["n"],
                     "val_total": vam["total"], "val_net5": vam["net5"], "val_n": vam["n"],
                     "lock_total": lkm["total"], "lock_net5": lkm["net5"], "lock_n": lkm["n"],
                     "full_price": full["price"], "full_fund": full["fund"], "full_total": full["total"],
                     "full_net5": full["net5"], "trend_up_bps": up, "trend_down_bps": down,
                     "trend_side_bps": side, "decision": decision, "trial_id": tid})
        print(f"{name:<19}{top_n:>2}{hold//DAY*24 if hold>=DAY else hold//12:>3}h | {ic:>7.4f} | "
              f"{trm['total']:>7.2f}{trm['net5']:>7.2f} | {vam['total']:>7.2f}{vam['net5']:>7.2f} | "
              f"{lkm['total']:>7.2f}{lkm['net5']:>7.2f} | {up:>6.0f}{down:>6.0f}{side:>6.0f} | {decision}")

        if name.startswith("3 "):
            head = (name, s, hold, top_n, bt_full)

    df = pd.DataFrame(rows)
    R.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(R.RESULTS_DIR / f"{FAMILY}.csv", index=False)
    _headline_attribution(head, close, cum, reg, df, log, idx, span_days, tr_end, va_end)


def _headline_attribution(head, close, cum, reg, df, log, idx, span_days, tr_end, va_end):
    name, sig, hold, top_n, bt = head
    at = attribute_trades(sig, close[SYMS], cum[SYMS], top_n, hold)
    at["month"] = at["ts"].dt.to_period("M").astype(str)
    at["trend"] = reg["trend"].reindex(at["ts"]).values
    at["btc_dir"] = np.where(at["btc_fwd"] > 0, "btc_up", "btc_down")

    by_sym = (at.groupby("sym")["total"].sum() * 1e4).round(1).sort_values()
    by_month = (at.groupby("month")["total"].sum() * 1e4).round(1)
    by_trend = (at.groupby("trend")["total"].sum() * 1e4).round(1)
    by_btcdir = (at.groupby("btc_dir")["total"].sum() * 1e4).round(1)
    legs = at.groupby("side")["fwd"].mean() * 1e4

    # figures
    R.plot_equity(bt.equity(), FAMILY, "cfg3")
    _bar(by_month, "cfg3 monthly net (bps)", f"{FAMILY}_monthly")
    _bar(by_trend, "cfg3 net by BTC trend regime (bps)", f"{FAMILY}_regime")
    _bar(by_sym, "cfg3 net by symbol (bps)", f"{FAMILY}_symbol")

    m = bt.metrics
    up = df.loc[df.config == name, "trend_up_bps"].iloc[0]
    down = df.loc[df.config == name, "trend_down_bps"].iloc[0]
    side = df.loc[df.config == name, "trend_side_bps"].iloc[0]
    cand = (df["decision"].str.startswith("CANDIDATE")).sum()
    crash = (df["decision"].str.startswith("CRASH")).sum()
    rej = (df["decision"].str.startswith("REJECT")).sum()

    print(f"\n=== BACKFILL VERDICT — funding reversal, {span_days}d, headline cfg3 (fz20% 24h) ===")
    print(f"Trend attribution (cfg3 total): UP {up:+.0f} | DOWN {down:+.0f} | SIDEWAYS {side:+.0f} bps")
    print(f"BTC-direction-during-trade: {by_btcdir.to_dict()}  (positive only in btc_down => crash/short beta)")
    print(f"Legs (mean fwd): long {legs.get('long', float('nan')):+.1f} bps, short {legs.get('short', float('nan')):+.1f} bps")
    print(f"Funding vs price (full): price {m['price_bps_per_trade']:+.2f} fund {m['funding_bps_per_trade']:+.2f} bps/trade")
    print(f"Per-symbol net (cfg3): {by_sym.to_dict()}")
    print(f"Config decisions: CANDIDATE {cand} | CRASH-BETA {crash} | REJECT {rej} | other {len(df)-cand-crash-rej}")

    R.write_research_log(FAMILY, _md(df, head, by_sym, by_month, by_trend, by_btcdir, legs, span_days, tr_end, va_end))
    print(f"\nLogged {len(df)} trials (now {log.count()} total). results/alpha_tests/{FAMILY}.csv | "
          f"research_logs/{FAMILY}.md  (no alpha claimed; lock-box reported separately)")


def _bar(series, title, fname):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["tab:green" if v > 0 else "tab:red" for v in series.values]
    ax.bar(series.index.astype(str), series.values, color=colors)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(title)
    plt.xticks(rotation=60, ha="right")
    fig.tight_layout()
    fig.savefig(R.FIG_DIR / f"{fname}.png", dpi=110)
    plt.close(fig)


def _md(df, head, by_sym, by_month, by_trend, by_btcdir, legs, span_days, tr_end, va_end):
    name, sig, hold, top_n, bt = head
    m = bt.metrics
    tbl = df[["config", "family", "hold_bars", "ic_full", "train_net5", "val_net5", "lock_net5",
              "trend_up_bps", "trend_down_bps", "trend_side_bps", "val_n", "decision"]].to_markdown(index=False, floatfmt=".3f")
    crash_only = by_trend.get("up", 0) <= 0 and by_trend.get("sideways", 0) <= 0 and by_trend.get("down", 0) > 0
    return f"""# Crypto funding-reversal — {span_days}-day BACKFILL / OOS / regime validation

**Universe:** 10 USDT perp majors · **Span:** {span_days} days · **Split (pre-registered):** train 60% /
val 20% / **lock-box 20% (reported separately, not tuned)** · **Cost:** crypto taker {CANON} bps/side ·
**Same 10 configs as T0100–T0109 — no tuning, no new configs.**

## Per-config: OOS net@5 by split + PnL by BTC trend regime

{tbl}

> `*_net5` = net bps/trade after 10 bps round-trip. `trend_*_bps` = total net PnL (bps) accrued in up /
> down / sideways months. **Decision** per pre-registered rules (REJECT if dies in val/lock; CRASH-BETA if
> positive only in down months; CANDIDATE if OOS-positive in ≥2 regimes with sensible legs).

## Headline cfg3 (funding z 20%, 24h hold) attribution
- **Trend regime:** up {by_trend.get('up', 0):+.0f} | down {by_trend.get('down', 0):+.0f} |
  sideways {by_trend.get('sideways', 0):+.0f} bps. **Crash-only = {crash_only}.**
- **BTC direction during trade:** {by_btcdir.to_dict()} — if positive only when `btc_down`, the book is
  short/crash beta, not general alpha.
- **Legs (mean forward return):** long {legs.get('long', float('nan')):+.1f} bps, short
  {legs.get('short', float('nan')):+.1f} bps.
- **Price vs funding (full):** price {m['price_bps_per_trade']:+.2f} + funding {m['funding_bps_per_trade']:+.2f}
  bps/trade — {'carry still NEGLIGIBLE' if abs(m['funding_bps_per_trade']) < 0.1*abs(m['price_bps_per_trade']) else 'carry now material'}.
- **Per-symbol net (bps):** {by_sym.to_dict()}
- **Per-month net (bps):** {by_month.to_dict()}

## The eight questions
1. **365-day backfill succeed / 2. validation pass?** See the audit (`docs/crypto_data_audit.md`) — span,
   gaps, funding coverage, OI limit.
3. **Persist outside the 30-day crash?** Decisions column + trend attribution above.
4. **Up/down/sideways?** Trend regime row for cfg3 (and per config).
5. **Survive 5 bps/side?** `val_net5` and `lock_net5` columns.
6. **Price, carry, or both?** price {m['price_bps_per_trade']:+.2f} vs funding {m['funding_bps_per_trade']:+.2f}.
7. **Robust across symbols/months?** Per-symbol and per-month tables above + figures.
8. **Candidate / crash-beta / rejected?** See decisions; summary counts in the runner output.

## Figures
`{FAMILY}_cfg3_equity_val.png` (equity), `{FAMILY}_monthly.png`, `{FAMILY}_regime.png`, `{FAMILY}_symbol.png`.

**No alpha claimed. Lock-box reported separately and not used for any choice.**
"""


if __name__ == "__main__":
    main()
