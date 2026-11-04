"""List 0-5 day options above a delta, and whether each fits the --trade2 price limits.

    python list_options.py                                # QQQ and SPY, delta above 0.30
    python list_options.py --symbols AAPL,NVDA --delta 0.20

Read-only: this never places orders.
"""
import argparse

from options_bot.broker import Broker, tick_round
from options_bot.config import Settings
from options_bot.contracts import MAX_CAPPED_DTE, premium_cap, quote_contracts, within_premium_cap
from options_bot.price_action import fetch_5m


def main():
    ap = argparse.ArgumentParser(description="List 0-5 day options above a delta")
    ap.add_argument("--symbols", default="QQQ,SPY", help="comma-separated tickers")
    ap.add_argument("--delta", type=float, default=0.30, help="show contracts with absolute delta above this")
    a = ap.parse_args()

    s = Settings()
    s.validate()
    b = Broker(s, live_orders=False)
    for sym in [x.strip().upper() for x in a.symbols.split(",") if x.strip()]:
        bars = fetch_5m(b.stocks, sym)
        if bars.empty:
            print(f"\n{sym}: no bars")
            continue
        price = float(bars.close.iloc[-1])
        print(f"\n{sym} last {price:.2f}")
        for kind in ("call", "put"):
            rows = quote_contracts(b.trading, b.options, sym, kind, price, 0, MAX_CAPPED_DTE, 0.10,
                                   s.min_open_interest, s.max_spread_pct)
            over = [c for c in rows if c.delta is not None and abs(c.delta) > a.delta]
            fit = sum(within_premium_cap(c) for c in over)
            missing = sum(c.delta is None for c in rows)
            print(f" {kind}: {len(rows)} liquid, {len(over)} above delta {a.delta:.2f}, "
                  f"{fit} within price limits, {missing} with no delta")
            if not over:
                continue
            print(f"  {'Contract':<22}{'Exp':>11}{'DTE':>4}{'Strike':>8}{'Bid':>7}{'Ask':>7}{'Mid':>7}"
                  f"{'Delta':>7}{'OI':>7}{'Limit':>7}  Fits")
            for c in sorted(over, key=lambda c: (c.expiration, abs(c.delta))):
                print(f"  {c.symbol:<22}{c.expiration.isoformat():>11}{c.dte:>4}{c.strike:>8.1f}{c.bid:>7.2f}"
                      f"{c.ask:>7.2f}{tick_round(c.mid):>7.2f}{c.delta:>+7.2f}{c.open_interest:>7}"
                      f"{premium_cap(c.dte):>7.2f}  {'yes' if within_premium_cap(c) else 'no'}")


if __name__ == "__main__":
    main()
