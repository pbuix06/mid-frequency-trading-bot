# Paper Monitoring Only — scope & governance

**Status:** observation / paper validation. **Live trading: NOT approved.**

This document governs the `mft/live_monitor/` module. Read it before running anything in
that module. It is deliberately short and absolute.

---

## What this module is

A real-time crypto **monitoring** and **paper-decision** system. It watches the live tape
(websocket / 1-minute bars), computes the same past-only features the research engine uses,
runs a single pre-declared trigger, gates it through risk limits, and records **simulated**
fills to a ledger and a daily report.

High-frequency **monitoring**; low-frequency paper **trading**. Most minutes produce no trade.

## What this module is NOT

- It is **not** a live trading bot.
- It is **not** a claim that any alpha was found.
- It does **not** revive any rejected strategy.
- It does **not** change the research verdict.

The project's verdict stands unchanged: **no production-ready alpha was found on
free/retail-accessible data**, and alpha hunting on that data is paused. See
`docs/RESEARCH_VERDICT.md` and `docs/NO_LIVE_DEPLOYMENT.md`.

---

## Hard guarantees (enforced in code + tests)

These are not promises in prose; they are checked by `tests/test_live_monitor.py` and by
assertions that run on every invocation:

1. **No real-order path exists.** `mft.live_monitor.paper_decision.LIVE_TRADING_ENABLED`
   is `False`. There is no `submit_order` / `send_order` / `place_order` / `broker` /
   `exchange_client` method anywhere in the module. `assert_no_live_order_path()` raises if
   one is ever added, and a test asserts none are present.
2. **Registry governance runs first.** `LiveMonitor.__init__` calls
   `mft.automation.registry.assert_no_live_approved()`; `LIVE_TRADING_APPROVED` stays `False`.
3. **Every "trade" is simulated** — a hypothetical fill with configurable taker fee and
   slippage, logged with its signal, costs, regime, and exit reason. Every **rejected**
   signal is logged with the exact risk-gate reasons.
4. **The trigger has zero expected edge.** The 24h-breakout rule is a *plumbing smoke test*,
   not a strategy. A profitable paper run is reported as a **SUSPECTED FALSE POSITIVE**, never
   as a discovery — identical philosophy to the research-automation deviation guard.

If you want live trading, that is a separate, deliberate decision with its own gate, its own
risk sign-off, and a real broker adapter that **does not exist in this repository today**.
Nothing here is that step.

---

## The one trigger (smoke test, not alpha)

```
long  if close breaks the PRIOR 24h high AND volume z-score confirms AND BTC regime not bearish
short if close breaks the PRIOR 24h low  AND volume z-score confirms AND BTC regime not bullish
flat  otherwise (the blocking reason is recorded)
```

The "prior 24h high/low" deliberately **excludes the current bar**, so a breakout is a genuine
new extreme. The BTC regime is a coarse, causal live proxy (trailing 24h return thresholds) —
**not** the research monthly-trend regime. None of this is tuned, and none of it is registered
as a candidate.

---

## How a profit is interpreted

If the paper book shows positive net PnL after costs, that is the **expected-suspicious**
case, because the trigger's expected edge is zero. The reporter raises
`SUSPECTED_FALSE_POSITIVE` and the correct response is: **do not act.** Decompose it — it will
almost always be regime luck (e.g. all the gain in one regime, ~50% win rate) or overfit, not
edge. A negative net is the *healthy* outcome and is reported as `STAYS_REJECTED`.

---

## Running it

See `docs/LIVE_MONITORING_PLAN.md` for the architecture and the exact commands. The safe
default is `--source replay` (offline, deterministic, over stored bars). Network sources
(`rest`, `websocket`) are read-only market-data feeds.
