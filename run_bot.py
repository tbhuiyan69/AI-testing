"""Entry point for the Alpaca paper options bot.

    python run_bot.py --once            # one cycle, dry-run (no orders)
    python run_bot.py                   # loop during market hours, dry-run
    python run_bot.py --trade           # loop and submit real PAPER orders
"""
import argparse
import logging
import sys
import time
from pathlib import Path

from options_bot.bot import run_cycle
from options_bot.broker import Broker
from options_bot.config import Settings


def setup_logging():
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("logs/bot.log", encoding="utf-8")],
    )


def main():
    ap = argparse.ArgumentParser(description="Alpaca paper-trading options bot")
    ap.add_argument("--trade", action="store_true", help="submit orders to the paper account (default: dry-run)")
    ap.add_argument("--once", action="store_true", help="run a single cycle, even if the market is closed")
    args = ap.parse_args()

    setup_logging()
    log = logging.getLogger("bot")
    s = Settings()
    s.validate()
    broker = Broker(s, live_orders=args.trade)

    acct = broker.account()
    log.info("Account %s  equity $%s  options BP $%s  mode=%s",
             acct.account_number, acct.equity, acct.options_buying_power,
             "PAPER ORDERS" if args.trade else "DRY-RUN")

    if args.once:
        run_cycle(broker, s)
        return

    while True:
        try:
            if broker.market_open():
                run_cycle(broker, s)
            else:
                log.info("Market closed; waiting.")
        except KeyboardInterrupt:
            raise
        except Exception:
            log.exception("Cycle failed; will retry next interval.")
        time.sleep(s.poll_minutes * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
