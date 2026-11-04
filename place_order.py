"""Place one option order by hand on the paper account, with a confirmation prompt.

    python place_order.py SPY call                 # auto-pick a ~0.40 delta contract, buy 1
    python place_order.py AAPL put --qty 2
    python place_order.py SPY call --contract SPY261106C00787000 --limit 7.50
    python place_order.py SPY call --contract SPY261106C00787000 --sell
"""
import argparse

from alpaca.data.enums import OptionsFeed
from alpaca.data.requests import OptionSnapshotRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest

from options_bot.broker import Broker, tick_round
from options_bot.config import Settings
from options_bot.contracts import pick_contract
from options_bot.strategy import daily_closes


def main():
    ap = argparse.ArgumentParser(description="Place one paper option order")
    ap.add_argument("underlying")
    ap.add_argument("kind", choices=["call", "put"])
    ap.add_argument("--contract", help="exact OCC symbol; skips auto-pick")
    ap.add_argument("--qty", type=int, default=1)
    ap.add_argument("--limit", type=float, help="limit price per share (default: mid)")
    ap.add_argument("--sell", action="store_true", help="sell to close instead of buy")
    a = ap.parse_args()

    s = Settings()
    s.validate()
    b = Broker(s, live_orders=True)
    underlying = a.underlying.upper()

    if a.contract:
        symbol = a.contract.upper()
        snap = b.options.get_option_snapshot(
            OptionSnapshotRequest(symbol_or_symbols=symbol, feed=OptionsFeed.INDICATIVE))[symbol]
        bid, ask = float(snap.latest_quote.bid_price), float(snap.latest_quote.ask_price)
    else:
        price = float(daily_closes(b.stocks, underlying, 5).iloc[-1])
        cand, why = pick_contract(b.trading, b.options, underlying, a.kind, price, s)
        if not cand:
            raise SystemExit(f"No contract found: {why}")
        symbol, bid, ask = cand.symbol, cand.bid, cand.ask

    side = OrderSide.SELL if a.sell else OrderSide.BUY
    limit = round(a.limit, 2) if a.limit else tick_round((bid + ask) / 2)
    print(f"\n  {side.value.upper()} {a.qty} x {symbol}")
    print(f"  bid {bid:.2f} / ask {ask:.2f}   limit {limit:.2f}")
    print(f"  approx {'credit' if a.sell else 'cost'}: ${limit * 100 * a.qty:,.2f}   (PAPER account)")
    if input("\n  Submit this order? [y/N] ").strip().lower() != "y":
        print("  Cancelled, nothing sent.")
        return

    order = b.trading.submit_order(LimitOrderRequest(
        symbol=symbol, qty=a.qty, side=side, limit_price=limit, time_in_force=TimeInForce.DAY))
    print(f"  Submitted: id {order.id}  status {order.status}")


if __name__ == "__main__":
    main()
