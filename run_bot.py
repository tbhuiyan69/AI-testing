"""Entry point for the Alpaca paper options bot.

    python run_bot.py --once            # one cycle, dry-run (no orders)
    python run_bot.py                   # loop during market hours, dry-run
    python run_bot.py --trade           # loop and submit real PAPER orders
    python run_bot.py --dry2            # price-action patterns on 5m bars, dry-run (no orders)
    python run_bot.py --trade2          # price-action patterns, submit real PAPER orders
"""
import argparse
import logging
import sys
import time
from pathlib import Path

from options_bot.bot import PRICE_ACTION, SMA, run_cycle
from options_bot.broker import Broker
from options_bot.config import Settings
from options_bot.price_action import BAR_MINUTES


def setup_logging():
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("logs/bot.log", encoding="utf-8")],
    )


def sleep_until_next_bar():
    """Wake just after the next 5m bar closes, so no bar is skipped by drift."""
    period = BAR_MINUTES * 60
    time.sleep(period - time.time() % period + 15)


def main():
    ap = argparse.ArgumentParser(description="Alpaca paper-trading options bot")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--trade", action="store_true", help="SMA strategy: submit orders to the paper account (default: dry-run)")
    mode.add_argument("--trade2", action="store_true", help="price-action strategy: submit orders to the paper account")
    mode.add_argument("--dry2", action="store_true", help="price-action strategy: log only, no orders")
    ap.add_argument("--once", action="store_true", help="run a single cycle, even if the market is closed")
    args = ap.parse_args()

    setup_logging()
    log = logging.getLogger("bot")
    s = Settings()
    s.validate()
    live = args.trade or args.trade2
    price_action = args.trade2 or args.dry2
    broker = Broker(s, live_orders=live)
    strategy = PRICE_ACTION if price_action else SMA

    acct = broker.account()
    log.info("Account %s  equity $%s  options BP $%s  mode=%s  strategy=%s",
             acct.account_number, acct.equity, acct.options_buying_power,
             "PAPER ORDERS" if live else "DRY-RUN", "price-action" if price_action else "SMA")

    if args.once:
        run_cycle(broker, s, strategy)
        return

    while True:
        try:
            if broker.market_open():
                run_cycle(broker, s, strategy)
            else:
                log.info("Market closed; waiting.")
        except KeyboardInterrupt:
            raise
        except Exception:
            log.exception("Cycle failed; will retry next interval.")
        if price_action:
            sleep_until_next_bar()
        else:
            time.sleep(s.poll_minutes * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
