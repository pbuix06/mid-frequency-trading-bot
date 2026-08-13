"""
LESSON 3 — Autopsy of the funding "edge": how the discipline caught its own best mirage.

Goal: relive the project's most dangerous moment. In June 2026 a crypto funding-
reversal signal printed the FIRST net-positive-after-cost numbers of the entire
project (T0100-T0109: cfg3 +14 bps/trade net, IC rising with horizon, Family-B
t-stat 14). Everything before it had died; this one *survived costs*. The 365-day
pre-registered backfill then unmasked it as crash beta. You will now re-run that
unmasking yourself, on the same data, through the same code — and see two more
subtle killers (sample fragility, regime concentration) that the six questions of
Lesson 1 don't yet cover.

Nothing here writes to trials/trials.csv: re-examining logged trials is free.
Uses data/crypto/ (365 days, 10 USDT perp majors) and the byte-identical config
builder mft/research/funding_signals.py — the same one both original runs used.

HOW TO RUN
    source .venv/bin/activate
    python learn/03_funding_autopsy.py

    # or, in VS Code: click "Run Cell" above any `# %%` block to step through it.

Loads ~10 seconds. Pairs with research_logs/crypto_funding_reversal.md (the
seduction), research_logs/crypto_funding_reversal_backfill.md (the autopsy),
and RESEARCH_LOG.md sec.17-18.
"""

# %% ──────────────────────────────────────────────────────────────────────────
# Setup: load the 365-day crypto store and build the pre-registered signal.
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

warnings.filterwarnings("ignore", message="Converting to PeriodArray")

from run_crypto_funding_backfill import BTC, SYMS, regimes  # noqa: E402  (reuse, not reimplement)

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

COST = 5.0            # crypto taker bps/side — 10 bps round-trip, the canonical stress
HOLD, TOP_N = 288, 2  # cfg3: 24h hold (288 five-min bars), top/bottom 2 of 10


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


if not (PERP_1M / f"{BTC}.parquet").exists():
    raise SystemExit("No crypto data. Run scripts/ingest_crypto.py --days 365 first.")

print("Loading 365 days x 10 perp majors (1m -> 5m) + funding ...")
perp = load_crypto_panel(SYMS, freq="5min", market=BTC, spot_dir=PERP_1M)
close = perp.to_wide("close", SYMS)
spot = load_crypto_panel(SYMS, freq="5min", market=BTC, spot_dir=SPOT_1M).to_wide("close", SYMS)
funding = {s: load_funding(s, FUNDING_DIR) for s in SYMS}
oi = {s: (load_open_interest(s)["open_interest"] if not load_open_interest(s).empty
          else pd.Series(dtype=float)) for s in SYMS}

fz, cum, oi4, ret4h, basis_z = build_features(close, funding, oi, spot, SYMS)
name, sig, hold, top_n, fam = build_configs(fz, oi4, ret4h, basis_z)[2]   # cfg3
sig = sig[SYMS]
assert name.startswith("3 ") and hold == HOLD and top_n == TOP_N
print(f"done: {len(close):,} bars, {close.index[0].date()} -> {close.index[-1].date()}, "
      f"config = '{name}' (fade the funding z-score, 24h hold)")

print("""
The signal in one sentence (economic rationale first, always): perpetual futures
settle "funding" between longs and shorts every 8h; an extreme funding rate means
a crowded, levered side — fade it (long the crowded-shorts, short the crowded-
longs) and hold 24h. cfg3 is one of 10 configs PRE-REGISTERED in
mft/research/funding_signals.py — frozen there precisely so the 30-day run and
any later validation use byte-identical logic. That freeze is about to matter.
""")


# %% ──────────────────────────────────────────────────────────────────────────
# LESSON 1 — The seduction: what the crash month looked like from inside.
banner("LESSON 1 — May 2026: the first 'edge' that survived costs")

W0 = pd.Timestamp("2026-05-06", tz="UTC")          # the original 30-day smoke-test window
W1 = pd.Timestamp("2026-06-05 23:59", tz="UTC")
btc_move = close[BTC][(close.index >= W0) & (close.index <= W1)]
print(f"  window: {W0.date()} -> {W1.date()}   BTC move: "
      f"{btc_move.iloc[-1] / btc_move.iloc[0] - 1.0:+.1%}")

print("""
What the ledger recorded from that window (T0100-T0109, 30 days of data — all
the funding history that existed in the store at the time):

    IC rose with horizon:  4h 0.006 -> 8h 0.014 -> 24h 0.031  (t ~ 7.8)
    cfg3 (24h):  gross +24.2 bps/trade  ->  net +14.2 after 10 bps round-trip
    cfg5 (24h, 10%): net +32.4.  Family-B 24h: IC 0.092, t ~ 14.

After ~95 straight rejections, the FIRST net-positive-after-cost configs in the
project. The longer-hold thesis (fewer trades -> costs matter less) seemed
vindicated. It would have been easy to size up right here. The log instead
recorded three caveats: (1) the window is a -25% BTC crash — can't separate
funding alpha from crash beta; (2) the PnL is price reversion, NOT carry;
(3) ~25 trades. Caveat (1) is testable with more data. So: backfill 365 days,
pre-register the split, THEN look.
""")


# %% ──────────────────────────────────────────────────────────────────────────
# LESSON 2 — First subtle killer: 25 trades is a lottery, not a sample.
banner("LESSON 2 — sample fragility: shift the clock, flip the verdict")

print("  Same signal, same window, same code — only the 24h rebalance GRID PHASE")
print("  moves (start the clock at 00:00 / 06:00 / 12:00 / 18:00):\n")
print(f"    {'grid start':<12}{'gross/trade':>12}{'net@5/trade':>12}{'trades':>8}")
nets = []
for off_h in (0, 6, 12, 18):
    start = W0 + pd.Timedelta(hours=off_h)
    m = (close.index >= start) & (close.index <= W1)
    c30 = close[m]
    f30 = {s: f[(f.index >= start) & (f.index <= W1)] for s, f in funding.items()}
    fz30, cum30, oi430, r430, bz30 = build_features(
        c30, f30, {s: pd.Series(dtype=float) for s in SYMS}, spot.reindex(c30.index), SYMS)
    cfg30 = build_configs(fz30, oi430, r430, bz30)[2]
    bt = funding_ls_backtest(cfg30[1][SYMS], c30[SYMS], cum30[SYMS],
                             top_n=TOP_N, hold_bars=HOLD, cost_bps_per_side=COST)
    nets.append(bt.metrics["net_bps_per_trade"])
    print(f"    +{off_h:>2}h{'':<8}{bt.metrics['total_gross_bps_per_trade']:>12.2f}"
          f"{nets[-1]:>12.2f}{bt.metrics['n_trades']:>8}")
assert max(nets) - min(nets) > 5, "the point of this cell is the spread"

print(f"""
Net per trade ranges {min(nets):+.1f} to {max(nets):+.1f} bps — the sign of the
"edge" depends on WHAT HOUR THE BACKTEST CLOCK STARTED. With ~25 trades, each
rebalance is 4% of the P&L; nudge the grid and different trades exist. The
recorded +14.2 was one draw from this lottery (the store has also been re-
ingested since, so bars differ slightly at the margins — another reason exact
tiny-sample numbers don't survive). A number this fragile isn't evidence; it's
noise wearing evidence's clothes. Rule: before believing net-per-trade, ask
"how many trades, and does the grid phase alone flip it?"
""")


# %% ──────────────────────────────────────────────────────────────────────────
# LESSON 3 — The pre-registered split: declared BEFORE the data came back.
banner("LESSON 3 — train 60% / val 20% / lock-box 20% — the kill/confirm test")

idx = close.index
N = len(idx)
tr_end, va_end = idx[int(0.6 * N)], idx[int(0.8 * N)]
masks = {"train": idx <= tr_end,
         "val": (idx > tr_end) & (idx <= va_end),
         "lock": idx > va_end}
print(f"  TRAIN {idx[0].date()} .. {tr_end.date()}  |  VAL .. {va_end.date()}  |  "
      f"LOCK .. {idx[-1].date()}")
print(f"\n    {'split':<8}{'net@5/trade':>12}{'trades':>8}")
res = {}
for lbl, mask in masks.items():
    bt = funding_ls_backtest(sig[mask], close[SYMS][mask], cum[SYMS][mask],
                             top_n=TOP_N, hold_bars=HOLD, cost_bps_per_side=COST)
    res[lbl] = bt.metrics["net_bps_per_trade"]
    print(f"    {lbl:<8}{res[lbl]:>12.3f}{bt.metrics['n_trades']:>8}")

assert abs(res["train"] - (-0.992)) < 0.05, "must reproduce the recorded backfill"
assert abs(res["val"] - 1.855) < 0.05 and abs(res["lock"] - (-10.043)) < 0.05

ic_full = L.ic_summary(L.ic_series(sig, forward_return_panel(close[SYMS], HOLD, intraday_only=False),
                                   method="spearman", min_names=3))["ic_mean"]
print(f"\n  full-sample IC = {ic_full:+.4f}   (was 0.031 on the crash month alone)")
assert abs(ic_full - 0.016) < 0.002

print("""
You just reproduced the recorded backfill row for cfg3 exactly: train -0.99,
val +1.86, lock -10.04 net bps/trade. The decision rule is hardcoded in the
runner (scripts/run_crypto_funding_backfill.py), declared before results came
back: reject unless net > 0 in val AND lock. Val scraped positive; the lock-box
said -10. REJECT. That is what "pre-registered" buys you: when the result came
back mixed, there was no room to redraw the split until it flattered. The IC
halving (0.031 -> 0.016) tells the same story from the correlation side: the
crash month was the signal's best month, and the year dilutes it.
""")


# %% ──────────────────────────────────────────────────────────────────────────
# LESSON 4 — Second subtle killer: WHICH months paid? Regime attribution.
banner("LESSON 4 — regime attribution: the 'alpha' only exists in crashes")

bt_full = funding_ls_backtest(sig, close[SYMS], cum[SYMS],
                              top_n=TOP_N, hold_bars=HOLD, cost_bps_per_side=COST)
reg = regimes(close)                                 # label each month up/down/sideways by BTC
trend_at = reg["trend"].reindex(bt_full.net.index)
by_trend = bt_full.net.groupby(trend_at).sum() * 1e4

print("  Total net P&L (bps) by BTC month-regime, full 365 days:\n")
for k in ("up", "down", "sideways"):
    v = float(by_trend.get(k, 0.0))
    print(f"    {k:<10}{v:>+9.0f} bps   {'<-- ALL of the profit' if k == 'down' else ''}")
assert by_trend["down"] > 0 > by_trend["up"] and by_trend["sideways"] < 0

print("""
Positive ONLY in down months (+1140), negative in up (-670) and sideways
(-1420). This is not a market-neutral funding alpha — it is a SHORT-CRASH bet:
when crowded longs get liquidated, fading them prints money; the rest of the
year it bleeds. The 30-day window "worked" because it WAS a crash. The project's
name for this verdict: CRASH-BETA, rejected. (Also confirmed at the trade level:
price PnL {:+.2f} vs funding carry {:+.2f} bps/trade — the funding rate was a
SIGNAL, never meaningful income, so there is no carry cushion to fall back on.)
""".format(bt_full.metrics["price_bps_per_trade"], bt_full.metrics["funding_bps_per_trade"]))


# %% ──────────────────────────────────────────────────────────────────────────
# CLOSING — the checklist grows from 6 questions to 8.
banner("THE AUTOPSY'S LESSON — two new questions for the checklist")
print("""
Lesson 1's six questions (returns / by-hand / costs / fills / trials / leakage)
would NOT have killed this one: the returns were real, the Sharpe honest, costs
paid, fills taker-honest, trial count logged, no leakage. It died to two
questions the crash month couldn't answer:

  7. REGIME — which months/regimes carry the P&L? If one regime is all of it,
     you own a bet on that regime, not an edge. Demand up/down/sideways
     attribution before believing any short-sample result.
  8. SAMPLE ROBUSTNESS — how many trades, and do harmless choices (grid phase,
     window start) flip the sign? If yes, the number is noise in a suit.

And the meta-lesson, which is the whole project in one line: the machinery is
built to catch YOUR OWN most convincing mirage. The funding signal was the best
thing the project ever found, and the discipline — pre-registered splits, lock-
box, regime attribution, byte-identical frozen configs — is exactly what it
took to prove it wasn't real. That proof cost nothing but compute. Believing
+14 bps/trade would have cost real money in every non-crash month since.

You have now driven the factory through its two hardest calls: Gate 4 (learn/02)
and the funding autopsy. The 8-question checklist is yours.
""")
