# Live Monitoring Plan — architecture & roadmap

> **Paper / observation only. Live trading is NOT approved.** This plan describes a real-time
> *monitoring* and *paper-decision* system. It does not place real orders and does not change
> the research verdict (no production-ready alpha found). Read `docs/PAPER_MONITORING_ONLY.md`
> first — it is the binding governance for this module.

This is a new **module/phase**, not a continuation of alpha hunting. Its purpose is
engineering and observation: prove the live plumbing works, watch how a pre-declared trigger
behaves on the live tape, and keep the false-positive discipline running in real time.

---

## Design principle: high-frequency monitoring, low-frequency trading

- **Monitor** every second (websocket) / every minute (bars): features refresh continuously.
- **Decide** only when a slower setup triggers: most minutes are no-trade. The "trade"
  frequency is set by the 24h-breakout setup and the risk gate, not by the bar frequency.

This mirrors how a real mid-frequency book runs: you ingest fast, you act rarely.

---

## Architecture (one direction, no order path)

```
            (read-only market data)
websocket_client.py ──► bar_builder.py ──► feature_stream.py ──► signal_engine.py
   seconds/1m            ticks→1m bars       rolling features       ONE breakout trigger
                                                                          │
                                                                          ▼
   report.py  ◄──────── paper_decision.py  ◄──────────────────────── risk_gate.py
  daily report          SIMULATED fills only                         pre-trade limits
  + FP guard            full ledger + no-trade log                   (block invalid trades)
```

Reuses existing infrastructure rather than duplicating it:
- **Crypto data layer** (`mft/data_layer/crypto_provider.py`) — the same Binance public
  endpoints used for research; the REST poller and websocket client are thin read-only consumers.
- **Governance** (`mft/automation/registry.py`) — `assert_no_live_approved()` runs in the
  monitor constructor; `LIVE_TRADING_APPROVED` stays `False`.
- **Warning vocabulary** (`mft/automation/monitor.Warning_`) — the deviation / false-positive
  guard speaks the same language as the research-automation framework.
- **Cost discipline** — taker fee + slippage are explicit, configurable, and conservative;
  the research cost floors (`mft/execution/costs.py`) are the reference.

### Files

| File | Responsibility |
|---|---|
| `websocket_client.py` | Read-only sources: `ReplayStream` (offline/default), `BinanceRestPoller` (near-live REST), `BinanceWebSocketClient` (seconds-level, optional dep). All yield closed 1m frames. |
| `bar_builder.py` | `Tick`→1m `Bar` aggregation (seconds-level core); `Bar.from_mapping` for already-closed klines; top-of-book spread capture. |
| `feature_stream.py` | Rolling, past-only `FeatureSnapshot` per symbol; coarse causal BTC regime. |
| `signal_engine.py` | The single 24h-breakout smoke-test trigger (NOT an alpha claim). |
| `risk_gate.py` | Pre-trade limits: staleness, spread, max positions, per-symbol exposure, daily loss, extreme vol, duplicate. |
| `paper_decision.py` | Simulated fills only; opens/closes/exits; logs every trade and every rejected signal. No order path. |
| `report.py` | Daily markdown report + parquet ledger + false-positive guard. |
| `runner.py` | `LiveMonitor` orchestrator: one closed-minute step wires it all together. |
| `scripts/run_live_monitor.py` | CLI entrypoint (paper mode). |

---

## Features refreshed every minute

Per symbol, strictly past-only (value at bar *t* uses only bars ≤ *t*):

- `return_1h`, `return_4h`, `return_24h`
- rolling volatility 4h / 24h (std of 1m returns)
- 24h high / low (prior window, **excluding** the current bar) + distance from each
- `volume_zscore` (current bar vs trailing baseline)
- BTC regime (bullish / bearish / neutral — coarse causal proxy)
- funding rate (if a perp funding map is supplied)
- top-of-book spread (if a book feed is available)

Windows are configurable (`FeatureConfig`) so tests run on tiny histories and replays over
short windows can warm up faster.

---

## Risk gate — a trade must clear ALL of these

1. data not stale (snapshot age ≤ `stale_seconds`)
2. spread below threshold (when top-of-book is known)
3. max open positions not exceeded
4. per-symbol notional cap not exceeded
5. daily loss limit not hit (halts new entries)
6. volatility not extreme (unless explicitly allowed)
7. no duplicate trade if already in a position for that symbol

Every block is logged with its exact reason; the daily report tallies rejections by reason.

---

## How to run (paper mode)

```bash
# offline, deterministic — replay stored 1m bars (the safe default)
python scripts/run_live_monitor.py --source replay --days 7 --symbols BTCUSDT ETHUSDT SOLUSDT

# near-live — poll Binance public REST for the latest CLOSED 1m kline (read-only)
python scripts/run_live_monitor.py --source rest --minutes 120 --symbols BTCUSDT ETHUSDT SOLUSDT

# seconds-level — public kline/bookTicker websocket (needs `pip install websocket-client`)
python scripts/run_live_monitor.py --source websocket --minutes 120
```

Outputs (under `reports/live_monitor/`):
- a daily markdown report (signals fired, rejections by reason, paper trades, PnL after costs,
  performance by symbol and by regime, data outages, false-positive guard),
- a parquet ledger of every simulated trade.

The governance gate runs first on every invocation; nothing here can enable live trading.

---

## Explicit non-goals

- No real orders, no broker/exchange adapter, no live flag — not now, not via config.
- No alpha claim; the trigger is a smoke test with zero expected edge.
- No reviving rejected research candidates; no tuning toward a positive paper number.
- No change to the lock-box discipline or the research verdict.

## If we ever wanted to go live (not now)

A real deployment would be a separate, deliberate phase with its own sign-off: a vetted edge
that cleared the research gates (none has), a real broker adapter behind an explicit and
audited approval, live-fill cost calibration, a kill switch, and staged capital. This module
is the *observation* rung of that ladder, intentionally stopping well short of it.
