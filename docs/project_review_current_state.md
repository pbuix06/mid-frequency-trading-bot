# Project review — current state (2026-06-06)

A sober, critical assessment. Not marketing. The short version: **a rigorous research factory has been
built and used honestly; it has tested ~8 distinct edges across daily equities, intraday equities, and
crypto, and found no production-ready alpha. Every predictive effect found is either too small for costs,
execution-fragile, or regime-confounded.** That is a real result, not a failure — but it must be stated plainly.

Status: **119 trials logged · 216 tests passing · 0 deployable alphas · 0 capital at risk.**

---

## 1. What has been built (the durable asset)

| Component | Where | State |
|---|---|---|
| Equity daily research + 3 parity engines, DSR/CPCV, survivorship harness | `mft/backtest/`, `mft/validation/` | mature (Phases 0–4) |
| **Equity intraday IC-first research engine** (5-min) | `mft/research/` (panel, features, targets, signal_lab, xs_backtest, splits, report) | mature, tested |
| **Crypto data layer** (Binance spot/perp/funding/OI providers + validation) | `mft/data_layer/crypto_provider.py`, `crypto_validate.py` | mature; **365 days real data ingested, clean** |
| **Crypto 24/7 backtest** (continuous UTC grid, no ET reset) | `mft/research/xs_backtest.py` (`continuous=True`) | tested |
| **Maker-fill simulator** (touch/trade-through/probabilistic; exit modes; adverse selection) | `mft/research/maker_sim.py` | tested |
| **Funding backtest** (price vs funding-carry attribution) | `mft/research/funding_backtest.py` | tested |
| **Validation + logging framework** | `trial_log.py` (append-only ledger), DSR, CPCV, pre-registered splits, **regime analysis**, cost sweeps | mature |

This is the real output of the work to date. It is reusable, tested, honest, and frequency/asset-agnostic.

---

## 2. Branches tested and verdicts

| # | Branch | Trials | Verdict |
|---|---|---|---|
| — | Equity daily factor book (momentum/value/quality/FX) | T0001–T0056 | Real but modest (0.5–0.7); **fails Gate 4 DSR** (best book 0.35). Significance, not execution. |
| 1 | Equity intraday **residual reversal** (5m) | T0057–T0064 | Real but tiny gross; **taker-dead** net (≈ −3 to −4 bps/trade). |
| 2 | Equity intraday **abnormal-volume continuation** | T0065–T0075 | **Wrong-signed** (IC negative → reverts); rejected. Volume gate hurts (IEX partial tape). |
| 3 | Equity intraday **opening-range breakout** | T0076–T0084 | Negative IC, **train/val sign flip**, both-legs-positive drift/beta; rejected. |
| 4 | Crypto **BTC→alt lead-lag / reversal** (perp 5m) | T0085–T0094 | IC-supported but **≡ alt cross-sectional reversal** (BTC cancels); **taker-dead**. |
| 5 | Crypto **naive maker reversal** (sim) | T0095–T0099 | **Adverse selection**: filled gross −4 bps, +22 bps of bounces unfilled; rejected. |
| 6 | Crypto **funding reversal** (30d smoke) | T0100–T0109 | *Looked* like the first cost-surviving edge (net +14 bps @5bps, 24h) — but 30d was a crash. |
| 7 | Crypto **funding reversal — 365d backfill/regime** | T0110–T0119 | **REJECTED.** IC collapsed (0.092→0.000), gross 24→7 bps (< cost), net-negative OOS incl. lock-box; PnL only in down months ⇒ **crash/liquidation short-beta, not alpha.** |
| 8 | Crypto **cross-sectional momentum** (4h/8h/24h, 365d) | T0120–T0129 | **REJECTED — wrong-signed.** IC **negative** (~−0.04) ⇒ the cross-section mean-*reverts*, not trends, even at slower horizons; gross ~1 bp < cost; net-negative OOS incl. lock-box across up/down/sideways. The "go slower with momentum" hatch is closed. |

---

## 3. Main research conclusion

- **Predictive effects exist.** Several signals have genuinely positive, sometimes strong, in-sample IC
  (crypto funding/lead-lag ICs were 3–10× the equity ones). The signals are not noise.
- **Most are too small for taker costs.** The honest gross edge per trade at minute-to-hourly frequency is
  ~0.5–2 bps, against ~3–10 bps round-trip taker cost. The cost wall is the dominant, repeatedly-confirmed
  killer. Edges only approach cost-viability at **longer holds** (24h) — i.e. by *leaving* MFT.
- **Naive maker execution suffers adverse selection.** Passive orders on a reversal signal fill the
  continued-losers and miss the bounces (filled gross −4 bps; unfilled opportunity +22 bps). Earning the
  spread requires order-book/queue modelling we do not have.
- **The one signal that cleared cost was regime-confounded.** Funding reversal's 30-day "edge" was a
  −25% BTC crash; on 365 days it is net-negative out-of-sample and makes money only in down months.
- **"Going slower" does not rescue it.** Crypto cross-sectional momentum at 4h–24h (T0120–T0129) is
  *wrong-signed* — IC negative ⇒ the cross-section mean-reverts, not trends, even at slower horizons. The
  longer-hold escape hatch is closed; the effect that exists (weak reversal) is the same one that is untradable.
- **No production alpha has been found.** This is stated without hedging.

The meta-finding: **at MFT frequency on retail-accessible data, the binding constraint is execution
economics, not signal discovery.** The project is very good at finding small true effects and very good at
proving they are not tradable.

---

## 4. What is still useful (do not throw away)

- **The IC-first research engine** — panel → features → targets → signal_lab → xs_backtest, with strict
  past-only features and no-leakage tests. Plugs in any new signal in ~1 runner.
- **The crypto data ingestion + 365 days of clean, validated data** (spot/perp/funding; OI 30d).
- **The validation tests** (216) — especially no-lookahead and the maker/funding causality tests.
- **Regime analysis** — the pre-registered split + trend/vol/drawdown/direction attribution that just
  killed the funding branch. This is the single most valuable methodological capability added.
- **Cost discipline** — single-source cost model, cost sweeps, gross-vs-net-vs-carry attribution.
- **Anti-overfitting process** — append-only trial ledger (119), DSR/MinBTL budget, **pre-registration**,
  lock-box discipline.
- **The honest rejection pipeline** — the project's actual core competency. Seven branches killed cleanly,
  each with a documented economic reason.

---

## 5. What NOT to do next (hard rules)

- **Do NOT live trade.** Zero deployable alphas. There is nothing to deploy.
- **Do NOT buy 10–30 years of equity intraday data.** History length is not the constraint; cost economics
  and breadth are. Long messy intraday history would be wasted money (regime non-stationarity, decimalisation,
  Reg NMS, microstructure drift). This was analysed earlier and remains true.
- **Do NOT keep tuning rejected branches.** Reversal (taker & maker), abnormal-volume, ORB, lead-lag, and
  funding are closed. In particular, **spot-perp basis is economically close to funding** (basis ↔ funding are
  tethered) — re-testing it risks re-litigating a just-rejected idea. Treat with suspicion.
- **Do NOT claim alpha from smoke tests.** The funding episode is the cautionary tale: a 30-day result that
  looked like the first edge was a crash artifact. No belief without OOS + regime validation.
- **Do NOT run more configs without pre-registration.** Every new config raises the multiple-testing bar.
  Pre-register the grid, the split, and the regime cuts *before* seeing results — always.

---

## 6. Sensible next research options (sober assessment)

| Option | What | Honest assessment |
|---|---|---|
| **A** | Spot-perp **basis**, 24h+ holds | Cheap (data in hand) but **economically near-duplicate of the rejected funding signal**; high chance of the same crash-confound. Low expected information. |
| **B** | **Open-interest + price** positioning | Blocked: Binance OI is **~30 days only**. Needs a paid OI backfill source (e.g. Coinglass/Laevitas). Do not pursue without committing to that data spend. |
| **C** | Expand crypto universe to **top 30–50 perps** | Adds breadth, but the rejected signals were not breadth-limited — they were cost/regime-limited. Helps only a signal that already works. Premature. |
| **D** | Move to **slower daily / 4h** strategies | Directly addresses the dominant finding (costs dominate at MFT). More cost headroom per trade. The data already *screams* this — funding only worked at the longest hold (24h). Most aligned with the evidence. |
| **E** | **Order-book / trade-level** research | The proper version of the maker thread, and the only path to a maker edge. But needs paid tick/quote data and a real queue/adverse-selection model. A multi-month, capital-committing program. |

---

## 7. Recommended next step — ✅ EXECUTED (T0120–T0129) → REJECTED

> **Update (2026-06-06):** this step was run. Crypto cross-sectional momentum (10 pre-registered configs,
> 4h/8h/24h, 365d, same split, no tuning) was **REJECTED — wrong-signed.** IC is **negative** (~−0.04) ⇒ the
> cross-section mean-*reverts* even at slower horizons; gross ~1 bp < cost; net-negative OOS incl. lock-box
> across all regimes. The "go slower with momentum" escape hatch is closed. Per the decision gate below
> ("fails the same way"), **the evidence is now overwhelming and alpha hunting on free/retail data is paused.**
> See §8 below for the final verdict.

**Recommendation (now spent): a single, cheap, fully pre-registered test of crypto cross-sectional MOMENTUM
(trend) at 4h / 8h / 24h holds on the existing 365-day data — i.e. Option D, made concrete with a new
signal family.**

Rationale, critically:
- **It is a genuinely untested family.** Everything rejected so far is reversal, breakout, volume, lead-lag,
  or funding. **Trend/momentum at multi-hour horizons has not been tested in crypto.** It is the obvious gap.
- **It targets the actual binding constraint.** The whole project says *costs dominate at MFT*. Going slower
  (4–24h holds, low turnover) is the one structural lever that gives gross room to beat cost — and funding
  showed edges only appear at the longest hold. Momentum is the canonical low-turnover signal.
- **It is nearly free and rigorous.** Uses data already ingested and the exact validation framework
  (`funding_backtest`/`xs_backtest` continuous mode + the pre-registered split + regime attribution from
  `run_crypto_funding_backfill.py`). No new data, no new infrastructure.
- **Pre-register everything up front** (the funding lesson): fixed config grid, train/val/lock-box split,
  and the regime cuts (up/down/sideways, btc-direction) declared *before* running — so a crash-confounded or
  sub-cost result is caught immediately, not after it looks exciting.

**Decision gate:** if 4–24h crypto momentum survives OOS across up/down/sideways regimes *and* clears 5 bps/side,
it is the project's first genuine candidate → then C (breadth) becomes worthwhile. **If it fails the same way
(sub-cost or regime-confounded), the evidence becomes overwhelming that retail-accessible MFT alpha is not
there, and the right move is to bank the factory + documented negative results as the deliverable, or commit
to paid data (E / B) with eyes open.** Either outcome is decision-useful.

What I would explicitly *not* recommend as the next step: A (near-duplicate of rejected funding), B/E
(require paid data — premature), C (breadth before a working signal).

---

## 8. Final retail-data MFT verdict

The momentum test (§7) was the last cheap retail-data experiment. With it, **every signal family accessible
on free/retail data has been tested and rejected** — reversal, mean-reversion, breakouts, volume, lead-lag,
maker execution, funding/positioning, and momentum — across daily equities, intraday equities, and crypto.
**No production-ready alpha was found.** This is the project's settled position, stated plainly:

- **Do NOT live trade.** There are zero deployable alphas. Nothing has earned capital.
- **Do NOT buy 10–30 years of equity intraday data.** History length was never the binding constraint; cost
  economics, execution, and regime-robustness are. Long messy intraday history would be wasted money.
- **Do NOT continue running more free-data MFT configs.** The space is exhausted: minute-to-hourly, taker and
  naive maker, on retail-accessible price/funding data, the edge is consistently too small for costs,
  execution-fragile, or regime-confounded. More configs only add multiple-testing risk for no expected gain.
- **Do NOT tune rejected branches.** Reversal, abnormal-volume, ORB, lead-lag, maker, funding, and momentum
  are all closed. Re-skinning them (e.g. spot-perp basis ≈ funding) is re-litigating settled results.

**The next serious frontier — if the project continues at all — is paid order-book / tick / trade-level data
with explicit queue-position and adverse-selection modelling** (the proper version of the maker thread, the
only un-falsified path to earning rather than paying the spread). That is a deliberate, capital- and
time-committing program, entered with eyes open — not a free experiment.

**Otherwise, bank the project as it stands: a rigorous, tested research engine + validated multi-asset data
stack + a disciplined pre-registration/lock-box/regime-validation methodology + thorough negative-results
documentation.** That is a genuine, honest deliverable. The factory works; it has been used to prove, cheaply
and repeatedly, that retail-accessible MFT alpha is not there — which is itself a valuable, hard-won result.

> No alpha claimed. No live-trading readiness. **Alpha hunting on free/retail data is paused.** This document
> is the project's checkpoint and its honest conclusion, not a green light.
