"""
Maker Execution Phase 1 — can the crypto short-horizon reversal be monetized with PASSIVE
limit orders instead of crossing the spread? PROTOTYPE / SIMULATION ONLY. No alpha claimed.

Signal: cross-sectional reversal = −alt_ret_15 (identical cross-sectionally to BTC lag_gap, proven
in Phase 2). Buy recent losers (top-3), short recent winners (bottom-3), continuous 24/7 grid.
Each leg is then routed through mft.research.maker_sim (passive fill, no fill without price
touching/trading through the limit). Spread is a configurable assumption (no real quotes yet).

We report OPTIMISTIC (touch) and CONSERVATIVE (trade-through) fills separately, sweep
spread/fee/adverse/timeout/exit-mode/horizon, and compare to a same-leg TAKER baseline.

Run: python scripts/run_crypto_maker_reversal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from mft.monitoring.trial_log import TrialLog  # noqa: E402
from mft.research import report as R  # noqa: E402
from mft.research.crypto_panel import PERP_1M, load_crypto_panel  # noqa: E402
from mft.research.maker_sim import MakerConfig, aggregate, simulate  # noqa: E402

FAMILY = "crypto_maker_reversal"
ALL_SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
ALTS = [s for s in ALL_SYMS if s != "BTCUSDT"]
M = {15: 3, 30: 6, 60: 12}
TOP_N = 3
# headline conservative config
HEAD = dict(spread_bps=5, maker_fee_bps=1, taker_fee_bps=5, adverse_selection_bps=2,
            timeout_bars=2, buffer_bps=1, exit_mode=1, hold_bars=M[30])


def gen_events(sig: pd.DataFrame, hold: int):
    events = []
    for t in range(0, len(sig.index), hold):
        s = sig.iloc[t].dropna()
        if len(s) < 2 * TOP_N:
            continue
        s = s.sort_values()
        events += [(x, t, "buy") for x in s.index[-TOP_N:]]    # biggest losers -> buy
        events += [(x, t, "sell") for x in s.index[:TOP_N]]    # biggest winners -> short
    return events


def taker_baseline_bps(trades: pd.DataFrame, spread_bps: float, taker_fee_bps: float) -> float:
    """Same legs, but TAKER: enter+exit cross the spread (full spread + 2 taker fees). Per-leg net bps."""
    cost = (spread_bps + 2 * taker_fee_bps) * 1e-4
    return round(float((trades["signal_ret"] - cost).mean()) * 1e4, 3)


def run_one(label: str, cfg: MakerConfig, events, bars, years: float, extra: dict | None = None) -> dict:
    tr = simulate(events, bars, cfg, seed=7)
    agg = aggregate(tr, years)
    row = {"label": label, "model": cfg.model, "exit_mode": cfg.exit_mode,
           "spread_bps": cfg.spread_bps, "maker_fee_bps": cfg.maker_fee_bps,
           "adverse_bps": cfg.adverse_selection_bps, "timeout": cfg.timeout_bars,
           "buffer_bps": cfg.buffer_bps, "hold_bars": cfg.hold_bars, **agg}
    if extra:
        row.update(extra)
    return row, tr


def main() -> None:
    if not (PERP_1M / "BTCUSDT.parquet").exists():
        print("No crypto perp data. Run scripts/ingest_crypto.py first.")
        return

    panel = load_crypto_panel(ALL_SYMS, freq="5min", market="BTCUSDT", spot_dir=PERP_1M)
    cw = panel.to_wide("close", ALTS)
    hw = panel.to_wide("high", ALTS)
    lw = panel.to_wide("low", ALTS)
    bars = {s: {"close": cw[s].to_numpy(float), "high": hw[s].to_numpy(float),
                "low": lw[s].to_numpy(float), "ts": cw.index} for s in ALTS}
    years = (cw.index[-1] - cw.index[0]).days / 365.25
    sig = -cw.pct_change(M[15])     # cross-sectional reversal (= BTC lag_gap cross-sectionally)
    ev = {h: gen_events(sig, h) for h in (M[15], M[30], M[60])}
    print(f"Loaded 9 alts, {len(cw)} 5m bars, {years*365:.0f} days. "
          f"Reversal signal, {len(ev[M[30]])} leg-events at 30m. SIMULATION ONLY.\n")

    rows, trades_head = [], None

    # ── 1. models A/B/C at headline ───────────────────────────────────────────
    for mdl, tag in [("A", "touch OPTIMISTIC"), ("B", "trade-through CONSERVATIVE"), ("C", "probabilistic")]:
        cfg = MakerConfig(model=mdl, **HEAD)
        row, tr = run_one(f"model {mdl} ({tag})", cfg, ev[HEAD["hold_bars"]], bars, years)
        rows.append(row)
        if mdl == "B":
            trades_head = tr

    taker_net = taker_baseline_bps(trades_head, HEAD["spread_bps"], HEAD["taker_fee_bps"])

    # ── 2. sensitivity sweeps (vary one axis from the conservative headline) ───
    base = {**HEAD, "model": "B"}
    def sweep(axis, values, fixed_label):
        for v in values:
            cfg = MakerConfig(**{**base, axis: v})
            row, _ = run_one(f"{fixed_label}={v}", cfg, ev[cfg.hold_bars], bars, years)
            row["sweep"] = fixed_label
            rows.append(row)

    sweep("spread_bps", [1, 2, 5, 10], "spread")
    sweep("adverse_selection_bps", [0, 1, 2, 5], "adverse")
    sweep("buffer_bps", [0, 1, 2, 5], "buffer")
    sweep("timeout_bars", [1, 2, 3], "timeout")
    sweep("exit_mode", [1, 2, 3], "exit")
    for h, hb in [(15, M[15]), (30, M[30]), (60, M[60])]:
        cfg = MakerConfig(**{**base, "hold_bars": hb})
        row, _ = run_one(f"horizon={h}m", cfg, ev[hb], bars, years)
        row["sweep"] = "horizon"
        rows.append(row)

    df = pd.DataFrame(rows)
    R.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(R.RESULTS_DIR / f"{FAMILY}.csv", index=False)

    # ── breakdowns on the conservative headline ───────────────────────────────
    fh = trades_head[trades_head["filled"]]
    fill_by_sym = (trades_head.groupby("sym")["filled"].mean().round(3)).to_dict()
    by_hour = (fh.set_index("ts")["net_adv"].groupby(lambda t: t.hour).mean() * 1e4).round(2)

    # ── figures: net_adv vs spread, net_adv vs adverse ────────────────────────
    _plot_sweep(df, "spread_bps", "spread", taker_net)
    _plot_sweep(df, "adverse_bps", "adverse", taker_net)

    # ── log a MODEST set of trials (headline per model + B exit modes) ────────
    log = TrialLog()
    to_log = df[df["label"].str.startswith("model ")].copy()
    extra_modes = df[df["label"].isin(["exit=2", "exit=3"])]
    for _, r in pd.concat([to_log, extra_modes]).iterrows():
        log.log(strategy="CryptoMakerReversal", asset_universe=ALTS,
                params={"label": r["label"], "model": r["model"], "exit_mode": int(r["exit_mode"]),
                        "spread_bps": r["spread_bps"], "maker_fee_bps": r["maker_fee_bps"],
                        "adverse_bps": r["adverse_bps"], "timeout": int(r["timeout"]),
                        "hold_bars": int(r["hold_bars"])},
                data_window=f"{cw.index[0].date()}:{cw.index[-1].date()}",
                is_sharpe=float(r["sharpe_filled"]) if pd.notna(r["sharpe_filled"]) else 0.0,
                oos_sharpe=None, max_dd=None, turnover=float(r["fill_rate"]),
                notes=(f"MAKER SIM (crypto reversal, 30d, PROTOTYPE no alpha); fill_rate={r['fill_rate']:.2f} "
                       f"net_adv={r['net_adv_bps']:+.2f}bps gross={r['gross_bps']:+.2f} vs taker {taker_net:+.2f}"))

    _print_and_report(df, taker_net, fill_by_sym, by_hour, years, log)


def _plot_sweep(df, col, sweep, taker_net):
    import matplotlib
    try:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    sub = df[df.get("sweep") == sweep].sort_values(col)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color="grey", lw=0.8)
    ax.axhline(taker_net, color="red", ls="--", lw=1, label=f"taker {taker_net:+.1f}")
    ax.plot(sub[col], sub["net_adv_bps"], marker="o", label="maker net_adv")
    ax.plot(sub[col], sub["gross_bps"], marker="s", ls=":", label="maker gross")
    ax.set_xlabel(col)
    ax.set_ylabel("bps/trade")
    ax.set_title(f"crypto maker reversal — {sweep} sensitivity")
    ax.legend()
    p = R.FIG_DIR / f"{FAMILY}_{sweep}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)


def _print_and_report(df, taker_net, fill_by_sym, by_hour, years, log):
    models = df[df["label"].str.startswith("model ")]
    print("=== MODELS at conservative headline (spread5/mkfee1/tkfee5/adverse2/timeout2/exit1/30m) ===")
    print(f"{'model':<32}{'fill%':>7}{'gross':>8}{'netFee':>8}{'netAdv':>8}{'unfOpp':>8}{'win%':>7}{'nFill':>7}")
    for _, r in models.iterrows():
        print(f"{r['label']:<32}{r['fill_rate']*100:>6.1f}%{r['gross_bps']:>8.2f}{r['net_fees_bps']:>8.2f}"
              f"{r['net_adv_bps']:>8.2f}{r['unfilled_opp_bps']:>8.2f}{r['win_rate']*100:>6.1f}%{r['n_filled']:>7}")
    print(f"\nTAKER baseline (same legs, full spread + 2 taker fees): {taker_net:+.2f} bps/trade")

    best_cons = df[(df["model"] == "B")]["net_adv_bps"].max()
    exit2 = df[df["label"] == "exit=2"]["net_adv_bps"].iloc[0] if (df["label"] == "exit=2").any() else float("nan")
    print(f"\nBest CONSERVATIVE (model B, any sweep) net_adv: {best_cons:+.2f} bps/trade")
    print(f"Maker-exit (mode 2, optimistic exit fills) net_adv: {exit2:+.2f} bps/trade")
    print("Sensitivity (net_adv bps): "
          + " | ".join(f"{s}:" + ",".join(f"{r[_col(s)]:g}->{r['net_adv_bps']:+.1f}"
                                          for _, r in df[df.get('sweep') == s].iterrows())
                       for s in ("spread", "adverse")))
    R.write_research_log(FAMILY, _md(df, taker_net, fill_by_sym, by_hour, best_cons, exit2))
    print(f"\nLogged {len(df[df['label'].str.startswith('model ')]) + 2} trials (now {log.count()} total). "
          f"results/alpha_tests/{FAMILY}.csv | research_logs/{FAMILY}.md  (PROTOTYPE — no alpha claimed)")


def _col(sweep):
    return {"spread": "spread_bps", "adverse": "adverse_bps"}[sweep]


def _md(df, taker_net, fill_by_sym, by_hour, best_cons, exit2):
    models = df[df["label"].str.startswith("model ")]
    mtbl = models[["label", "fill_rate", "gross_bps", "net_fees_bps", "net_adv_bps",
                   "unfilled_opp_bps", "win_rate", "sharpe_filled", "n_filled"]].to_markdown(index=False, floatfmt=".3f")
    def sweep_tbl(s, col):
        return df[df.get("sweep") == s][[col, "fill_rate", "gross_bps", "net_adv_bps", "n_filled"]].to_markdown(index=False, floatfmt=".3f")
    pos_cons = best_cons > 0
    return f"""# Crypto maker-execution reversal (perp 5-min) — PROTOTYPE / SIMULATION ONLY

**No real quotes** — spread is an assumption. No alpha claimed. Signal = cross-sectional short-horizon
reversal (= BTC lag_gap cross-sectionally). Passive limit fills only when price touches/trades through.

## Models at conservative headline (spread 5 bps, maker_fee 1, taker_fee 5, adverse 2, timeout 2, exit=taker, 30m)
{mtbl}

**TAKER baseline (same legs, full spread + 2 taker fees): {taker_net:+.2f} bps/trade.**

## Sensitivity (model B conservative)
### spread_bps
{sweep_tbl('spread', 'spread_bps')}
### adverse_selection_bps
{sweep_tbl('adverse', 'adverse_bps')}
### buffer_bps (trade-through)
{sweep_tbl('buffer', 'buffer_bps')}
### timeout_bars
{sweep_tbl('timeout', 'timeout')}
### exit_mode (1 taker, 2 maker-assume-fill, 3 maker+taker-fallback)
{sweep_tbl('exit', 'exit_mode')}
### horizon
{sweep_tbl('horizon', 'hold_bars')}

## Fill rate by symbol (headline) & performance by UTC hour
- Fill rate by symbol: {fill_by_sym}
- Net_adv bps by UTC hour: {by_hour.to_dict()}

## Answers (the five questions)
1. **Maker vs taker?** Maker net_adv vs taker {taker_net:+.2f} bps/trade — maker is better by saving the
   spread, but see whether it crosses zero.
2. **Survives conservative fills (model B, taker exit)?** Best conservative net_adv = **{best_cons:+.2f}**
   bps/trade ⇒ {'POSITIVE (fragile, verify)' if pos_cons else 'still NEGATIVE'}.
3. **Only under optimistic assumptions?** Maker-exit mode-2 (assumes passive exit always fills) net_adv
   = {exit2:+.2f} bps/trade — compare to conservative taker-exit to see how much of any edge needs the
   optimistic exit-fill assumption.
4. **What assumptions are required?** See the spread/adverse sweeps — a positive result needs WIDE spread
   captured AND LOW adverse selection. If it needs adverse≈0 or maker-exit-always-fills, it is fragile.
5. **Backfill 12 months?** Only if a CONSERVATIVE config (model B, realistic exit, adverse>0) is net
   positive with adequate fill rate. Otherwise the maker idea is not yet worth the data spend.

## Limitations
- No real bid/ask/queue — spread assumed, no queue-position model (you'd be at the back of the queue).
- Adverse selection is a flat per-fill penalty, not flow-conditioned.
- 30 days, 9 majors, one regime. Mode-2 "maker exit" assumes fills — optimistic.
- This is an execution PROTOTYPE, not a live strategy. Do not claim live-tradable alpha.
"""


if __name__ == "__main__":
    main()
