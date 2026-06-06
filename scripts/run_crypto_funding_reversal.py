"""
Crypto Derivatives Phase 1 — FUNDING-RATE reversal / crowded positioning. IC-first.

Genuinely NEW signal source: funding rate = positioning data, independent of price (unlike the
BTC lead-lag, which cross-sectionally collapsed to alt reversal). Lower turnover (funding updates
every 8h) and longer holds (4–24h) so the per-trade edge can clear costs.

Hypothesis: extreme +funding = crowded longs (short); extreme −funding = crowded shorts (long).
Shorting crowded-longs ALSO EARNS funding carry — so we attribute PRICE PnL vs FUNDING PnL separately.
Three families: A funding mean-reversion; B funding + 4h price confirmation; C funding + OI rising;
plus a funding+basis config. Continuous 24/7, dollar-neutral top/bottom-N, entry next bar, no funding
look-ahead (funding mapped known-only).

⚠️ SMOKE TEST: funding history is ~30 days (~90 settlements) on 10 majors. Do NOT overclaim. Reject
too-few-trade configs. Run: python scripts/run_crypto_funding_reversal.py
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
    cumulative_funding_series,
    funding_zscore_series,
    load_crypto_panel,
    load_funding,
    load_open_interest,
    map_to_bars,
)
from mft.research.funding_backtest import funding_ls_backtest  # noqa: E402
from mft.research.targets import forward_return_panel  # noqa: E402
from mft.validation.metrics import sharpe  # noqa: E402

FAMILY = "crypto_funding_reversal"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
BTC = "BTCUSDT"
H = {"4h": 48, "8h": 96, "24h": 288}     # 5-min bars
COST_SWEEP = (0.0, 2.0, 5.0, 10.0)
CANON_COST = 5.0
MIN_TRADES = 25


def main() -> None:
    if not (FUNDING_DIR / f"{BTC}.parquet").exists():
        print("No funding data. Run scripts/ingest_crypto.py first.")
        return

    perp = load_crypto_panel(SYMS, freq="5min", market=BTC, spot_dir=PERP_1M)
    close = perp.to_wide("close", SYMS)
    idx = close.index

    # funding (known-only mapping) + cumulative funding for carry
    fz, cumf = {}, {}
    for s in SYMS:
        fund = load_funding(s, FUNDING_DIR)
        if fund.empty:
            continue
        fz[s] = map_to_bars(funding_zscore_series(fund, window=10), idx)
        cumf[s] = map_to_bars(cumulative_funding_series(fund), idx)
    funding_z = pd.DataFrame(fz).reindex(idx)
    cum_funding = pd.DataFrame(cumf).reindex(idx)

    # OI 4h change (known-only) and price confirmation
    oi4 = pd.DataFrame({s: map_to_bars(load_open_interest(s)["open_interest"], idx).pct_change(H["4h"])
                        for s in SYMS if not load_open_interest(s).empty}).reindex(idx)
    ret4h = close.pct_change(H["4h"])

    # basis (optional context): perp/spot - 1, z-scored
    spot = load_crypto_panel(SYMS, freq="5min", market=BTC, spot_dir=SPOT_1M).to_wide("close", SYMS)
    basis = (close / spot.reindex(idx) - 1.0)
    basis_z = (basis - basis.rolling(H["8h"]).mean()) / basis.rolling(H["8h"]).std()

    sig_A = -funding_z                                   # high = long crowded-shorts
    confirmed = np.sign(ret4h) == np.sign(sig_A)         # price confirms the reversion direction
    sig_B = sig_A.where(confirmed)
    sig_C = sig_A.where(oi4 > 0)                          # OI rising
    sig_basis = sig_A.where(basis_z.abs() > 1)           # extreme basis

    # (name, signal, hold_bars, top_n, family)
    configs = [
        ("1 A fz 20% h4",   sig_A, H["4h"], 2, "A"),
        ("2 A fz 20% h8",   sig_A, H["8h"], 2, "A"),
        ("3 A fz 20% h24",  sig_A, H["24h"], 2, "A"),
        ("4 A fz 10% h8",   sig_A, H["8h"], 1, "A"),
        ("5 A fz 10% h24",  sig_A, H["24h"], 1, "A"),
        ("6 B fund+wk4h h8", sig_B, H["8h"], 2, "B"),
        ("7 B fund+wk4h h24", sig_B, H["24h"], 2, "B"),
        ("8 C fund+OIup h8", sig_C, H["8h"], 2, "C"),
        ("9 C fund+OIup h24", sig_C, H["24h"], 2, "C"),
        ("10 fund+basis h8", sig_basis, H["8h"], 2, "basis"),
    ]

    log, rows = TrialLog(), []
    print(f"Loaded {len(SYMS)} perp majors, {len(close)} 5m bars, "
          f"{idx[0].date()}->{idx[-1].date()}. Funding ~30d. SMOKE TEST.\n")
    print(f"{'config':<19}{'fam':>4}{'hold':>5}{'N':>2} | {'IC':>7}{'ICt':>5} | "
          f"{'price':>7}{'fund':>7}{'totGr':>7}{'net@5':>7} | {'Sh':>6}{'longL':>7}{'shortL':>7}{'nTr':>5} trial")
    print("-" * 116)

    head = None
    for name, sig, hold, top_n, fam in configs:
        tgt = forward_return_panel(close[SYMS], hold, entry_lag=1, intraday_only=False)
        ic = L.ic_series(sig[SYMS], tgt, method="spearman", min_names=3)
        summ = L.ic_summary(ic)
        bt = funding_ls_backtest(sig[SYMS], close[SYMS], cum_funding[SYMS],
                                 top_n=top_n, hold_bars=hold, cost_bps_per_side=CANON_COST)
        m = bt.metrics
        too_few = m["n_trades"] < MIN_TRADES
        tid = log.log(strategy="CryptoFundingReversal", asset_universe=SYMS,
                      params={"config": name, "family": fam, "hold_bars": hold, "top_n": top_n,
                              "cost_bps_per_side": CANON_COST, "freq": "5min", "venue": "binance-perp"},
                      data_window=f"{idx[0].date()}:{idx[-1].date()}",
                      is_sharpe=float(m["sharpe"]) if pd.notna(m["sharpe"]) else 0.0, oos_sharpe=None,
                      max_dd=float(m["max_drawdown"]) if pd.notna(m["max_drawdown"]) else None,
                      turnover=float(bt.trades_per_day),
                      notes=(f"CRYPTO FUNDING reversal SMOKE 30d; fam {fam}; IC={summ['ic_mean']:.4f} "
                             f"t={summ['ic_tstat']:.2f}; price={m['price_bps_per_trade']:+.2f} "
                             f"fund={m['funding_bps_per_trade']:+.2f} tot={m['total_gross_bps_per_trade']:+.2f} "
                             f"net@5={m['net_bps_per_trade']:+.2f} nTr={m['n_trades']}"))

        rows.append({"family_tag": fam, "config": name, "hold_bars": hold, "top_n": top_n,
                     "ic_mean": summ["ic_mean"], "ic_tstat": summ["ic_tstat"],
                     "price_bps": m["price_bps_per_trade"], "funding_bps": m["funding_bps_per_trade"],
                     "total_gross_bps": m["total_gross_bps_per_trade"], "net_bps_5": m["net_bps_per_trade"],
                     "sharpe_net5": m["sharpe"], "max_dd": m["max_drawdown"],
                     "long_leg_bps": m["long_leg_bps"], "short_leg_bps": m["short_leg_bps"],
                     "win_rate": m["win_rate"], "trades_per_day": bt.trades_per_day,
                     "n_trades": m["n_trades"], "too_few_trades": too_few, "trial_id": tid})
        print(f"{name:<19}{fam:>4}{hold:>5}{top_n:>2} | {summ['ic_mean']:>7.4f}{summ['ic_tstat']:>5.1f} | "
              f"{m['price_bps_per_trade']:>7.2f}{m['funding_bps_per_trade']:>7.2f}"
              f"{m['total_gross_bps_per_trade']:>7.2f}{m['net_bps_per_trade']:>7.2f} | "
              f"{m['sharpe']:>6.1f}{m['long_leg_bps']:>7.2f}{m['short_leg_bps']:>7.2f}{m['n_trades']:>5} {tid}"
              + ("  <-few" if too_few else ""))
        if name.startswith("2 "):
            head = (name, sig, hold, top_n, bt, ic)

    df = pd.DataFrame(rows)
    R.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(R.RESULTS_DIR / f"{FAMILY}.csv", index=False)
    R.update_leaderboard(df[["config", "family_tag", "hold_bars", "ic_tstat", "total_gross_bps",
                             "net_bps_5", "n_trades", "trial_id"]].assign(family=FAMILY)
                         .rename(columns={"net_bps_5": "val_sharpe"}), rank_col="val_sharpe")
    _verdict(df, head, close, cum_funding, log)


def cost_sweep(total: pd.Series, ppy: float) -> dict:
    return {c: (sharpe(total - 2 * c * 1e-4, periods_per_year=ppy),
                float((total - 2 * c * 1e-4).mean() * 1e4)) for c in COST_SWEEP}


def _verdict(df, head, close, cum_funding, log):
    name, sig, hold, top_n, bt, ic = head
    m = bt.metrics
    ppy = len(bt.total) / max((bt.total.index[-1] - bt.total.index[0]).days / 365.25, 1e-9)
    sweep = cost_sweep(bt.total, ppy)
    # per-symbol attribution: how often each coin is in the long/short leg, and net by symbol
    by_day = bt.net.groupby(bt.net.index.date).sum()
    top_day_share = float(by_day.max() / by_day[by_day > 0].sum()) if (by_day > 0).any() else float("nan")
    R.plot_buckets(L.bucket_returns(sig[SYMS], forward_return_panel(close[SYMS], hold, intraday_only=False), n_buckets=5, min_names=3), FAMILY, "funding")
    R.plot_equity(bt.equity(), FAMILY, "cfg2")

    enough = df[~df["too_few_trades"]]
    net_pos = enough[enough["net_bps_5"] > 0]
    ic_pos = int((df[df["family_tag"] == "A"]["ic_mean"] > 0).sum())
    rev_gross = 1.2  # short-horizon reversal branch gross ~0.8–1.7 bps/trade (taker)

    print("\n=== SMOKE-TEST VERDICT — funding reversal (30d, NOT an alpha claim) ===")
    print(f"Q1 gross/trade vs 5m reversal (~{rev_gross} bps): headline total gross "
          f"{m['total_gross_bps_per_trade']:+.2f} bps/trade ({m['n_trades']} trades).")
    print("Q2 cost sweep (headline total): "
          + ", ".join(f"{int(c)}bps:{nb:+.2f}" for c, (sh, nb) in sweep.items()) + " net bps/trade")
    print(f"Q3 IC: headline IC {df.loc[df.config==name,'ic_mean'].iloc[0]:+.4f} "
          f"(t {df.loc[df.config==name,'ic_tstat'].iloc[0]:+.1f}); Family-A configs IC-positive: {ic_pos}/5.")
    print(f"Q4 attribution (headline): PRICE {m['price_bps_per_trade']:+.2f} + FUNDING "
          f"{m['funding_bps_per_trade']:+.2f} = TOTAL {m['total_gross_bps_per_trade']:+.2f} bps/trade.")
    print(f"Q5 robustness: net-positive(@5bps) & enough-trades configs: {len(net_pos)}/{len(df)}; "
          f"top single-day share of headline net = {top_day_share:.0%}.")
    print(f"Q6 sample: ~30 days, {m['n_trades']} headline trades — thin; see report.")

    R.write_research_log(FAMILY, _md(df, head, sweep, top_day_share, net_pos, ic_pos, rev_gross))
    print(f"\nLogged {len(df)} trials (now {log.count()} total). results/alpha_tests/{FAMILY}.csv | "
          f"research_logs/{FAMILY}.md  (no alpha claimed; 30d smoke test)")


def _md(df, head, sweep, top_day_share, net_pos, ic_pos, rev_gross) -> str:
    name, sig, hold, top_n, bt, ic = head
    m = bt.metrics
    tbl = df[["config", "family_tag", "hold_bars", "top_n", "ic_mean", "ic_tstat", "price_bps",
              "funding_bps", "total_gross_bps", "net_bps_5", "n_trades", "too_few_trades"]].to_markdown(index=False, floatfmt=".3f")
    sweep_tbl = "\n".join(f"| {int(c)} | {sh:.2f} | {nb:.2f} |" for c, (sh, nb) in sweep.items())
    return f"""# Crypto funding-rate reversal (perp 5-min) — research log  ·  PHASE-1 SMOKE TEST

**Universe:** 10 USDT perp majors · **Data:** ~30 days, funding 8h (~90 settlements) · **24/7 continuous** ·
**Cost:** crypto taker {CANON_COST} bps/side · **No funding look-ahead** (known-only mapping, entry next bar).

> ⚠️ **NOT an alpha claim.** ~30 days / ~90 funding settlements is very thin. Longer-hold configs have few
> trades. This is a smoke test of a NEW (positioning) signal + price-vs-carry attribution.

## Results (full-sample, 30 days)

{tbl}

> `price_bps` = perp price-reversion PnL/trade; `funding_bps` = funding carry PnL/trade (short receives
> +funding, long pays); `total_gross` = price+funding; `net_bps_5` = total − 10 bps round-trip (5/side).

## Cost sweep (headline cfg2: Family A, 20%, 8h hold)
| cost bps/side | net Sharpe | net bps/trade |
|---:|---:|---:|
{sweep_tbl}

## Answers (the six questions)
1. **Gross vs 5m reversal (~{rev_gross} bps/trade):** headline total gross **{m['total_gross_bps_per_trade']:+.2f}**
   bps/trade over **{hold//12}h** holds — {'LARGER' if m['total_gross_bps_per_trade'] > rev_gross else 'NOT larger'},
   as expected for longer holds (bigger moves), but on far fewer trades.
2. **Survives taker cost?** Net@5bps = **{m['net_bps_per_trade']:+.2f}** bps/trade — see cost sweep.
3. **IC-supported?** Headline IC {df.loc[df.config==name,'ic_mean'].iloc[0]:+.4f}
   (t {df.loc[df.config==name,'ic_tstat'].iloc[0]:+.1f}); Family-A IC-positive {ic_pos}/5.
4. **Edge from price, funding, or both?** Headline: PRICE {m['price_bps_per_trade']:+.2f} + FUNDING
   {m['funding_bps_per_trade']:+.2f} bps/trade. (If most of `total` is `funding`, it is a CARRY trade, not
   a price-prediction edge — important distinction.)
5. **Robust or one coin/day?** Net-positive & enough-trades configs: **{len(net_pos)}/{len(df)}**; top single
   day = {top_day_share:.0%} of headline net P&L.
6. **30 days too short?** Yes — funding history is ~30d/~90 settlements; 24h configs have ~25–30 trades.
   Backfill before any belief (see below).

## Critical checks
- No future funding used (known-only mapping + entry next bar; tested in `tests/test_funding.py`).
- Reject too-few-trade configs (flagged). Reject backtest-positive-but-IC-negative.
- **Price vs carry:** read the split — a funding *carry* edge (mechanical) is different from a price
  *reversion* edge (predictive). Both shown per config.

## Limitations / next
- ~30 days, 10 majors — thin; longer holds especially. No spot-perp execution modelled; funding assumed
  received/paid in full (no funding-prediction error). **Backfill 6–12 months of funding + perp before
  any further work** if a config is net-positive with adequate trades AND IC-supported.
"""


if __name__ == "__main__":
    main()
