"""
Gate 0 — IB "hello world" connection and paper order test.

Requires IB Gateway running with:
  - API socket enabled on port 7497
  - Read-Only API unchecked
  - Paper trading account

Run: python scripts/ib_hello_world.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from ib_insync import IB, MarketOrder, Stock, util

HOST = "127.0.0.1"
PORT = 7497       # 7497 = paper, 7496 = live (never touch live yet)
CLIENT_ID = 1


def main() -> None:
    ib = IB()

    # ── 1. Connect ────────────────────────────────────────────────────────
    print(f"Connecting to IB Gateway at {HOST}:{PORT}...")
    try:
        ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=10)
    except Exception as e:
        print(f"\n✗ Connection failed: {e}")
        print("  Check: IB Gateway running? API socket enabled? Port 7497?")
        return

    print(f"✓ Connected  account={ib.wrapper.accounts}  server={ib.client.serverVersion()}")

    # ── 2. Account summary ────────────────────────────────────────────────
    account = ib.wrapper.accounts[0]
    summary = {s.tag: s.value for s in ib.accountSummary(account)}
    net_liq = summary.get("NetLiquidation", "N/A")
    print(f"✓ Account {account}  NetLiq=${float(net_liq):,.2f}")

    # ── 3. Hello world: qualify SPY contract ─────────────────────────────
    spy = Stock("SPY", "SMART", "USD")
    ib.qualifyContracts(spy)
    print(f"✓ Contract qualified: {spy}")

    # ── 4. Place 1-share paper market order ───────────────────────────────
    order = MarketOrder("BUY", 1)
    trade = ib.placeOrder(spy, order)
    ib.sleep(2)  # wait for fill

    print(f"✓ Order placed: {trade.order}")
    print(f"  Status: {trade.orderStatus.status}")
    if trade.fills:
        fill = trade.fills[0]
        print(f"  Fill: {fill.execution.shares} shares @ ${fill.execution.avgPrice:.2f}")

    # ── 5. Cancel / flatten (we don't want open positions) ────────────────
    if trade.orderStatus.status not in ("Filled", "Cancelled"):
        ib.cancelOrder(trade.order)
        print("  Order cancelled (not filled in time)")
    elif trade.fills:
        # Close the 1-share position
        close_order = MarketOrder("SELL", 1)
        close_trade = ib.placeOrder(spy, close_order)
        ib.sleep(2)
        print(f"✓ Position closed: {close_trade.orderStatus.status}")

    print("\n✓ GATE 0: IB HELLO WORLD PASSED")
    print("  Paper API confirmed working. Gate 0 is complete.")

    ib.disconnect()


if __name__ == "__main__":
    util.startLoop()
    main()
