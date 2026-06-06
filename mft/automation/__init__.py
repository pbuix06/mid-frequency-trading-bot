"""
Research automation + paper-trading VALIDATION framework.

This is NOT a live trading system. It exists to monitor pre-registered candidate signals on
near-live data, compare them to their backtest expectations, and FLAG FALSE POSITIVES — the
exact failure mode this project's discipline is built to prevent. No real orders are ever placed.

Hard guarantees (enforced in code + tested in tests/test_automation.py):
  - `registry.LIVE_TRADING_APPROVED` is False and there is no code path that sets it True.
  - No strategy spec may have status "live"; the paper engine has no real-order adapter at all.
  - Every "trade" is a hypothetical, simulated fill recorded to a ledger.

See docs/NO_LIVE_DEPLOYMENT.md and docs/project_review_current_state.md (the project found no
production-ready alpha).
"""
