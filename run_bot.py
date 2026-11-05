"""Entry point for the Alpaca paper options bot.

    python run_bot.py --once            # one cycle, dry-run (no orders)
    python run_bot.py                   # loop during market hours, dry-run
    python run_bot.py --trade           # loop and submit real PAPER orders
    python run_bot.py --trade2 --auto   # same, but contracts under $100 each buy without asking
    python run_bot.py --dry2            # price-action patterns on 5m bars, dry-run (no orders)
    python run_bot.py --trade2          # price-action patterns; asks yes/no before each PAPER buy
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


def run_price_action_loop(broker, s, strategy, log):
    """Check exits every minute, and look for new setups once per closed 5-minute bar."""
    last_bar, was_open = None, None
    while True:
        try:
            is_open = broker.market_open()
            if is_open:
                bar = int(time.time() // (BAR_MINUTES * 60))
                run_cycle(broker, s, strategy, entries=bar != last_bar)
                last_bar = bar
            elif was_open is not False:
                log.info("Market closed; waiting.")
            was_open = is_open
        except KeyboardInterrupt:
            raise
        except Exception:
            log.exception("Cycle failed; will retry next minute.")
        time.sleep(60 - time.time() % 60 + 15)  # 15s past the minute, so a just-closed bar is available


def main():
    ap = argparse.ArgumentParser(description="Alpaca paper-trading options bot")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--trade", action="store_true", help="SMA strategy: submit orders to the paper account (default: dry-run)")
    mode.add_argument("--trade2", action="store_true", help="price-action strategy: ask yes/no, then submit buys to the paper account")
    mode.add_argument("--dry2", action="store_true", help="price-action strategy: log only, no orders")
    ap.add_argument("--auto", action="store_true", help="with --trade2: buy without asking if one contract costs under $100; otherwise ask")
    ap.add_argument("--once", action="store_true", help="run a single cycle, even if the market is closed")
    args = ap.parse_args()
    if args.auto and not args.trade2:
        ap.error("--auto only works with --trade2")

    setup_logging()
    log = logging.getLogger("bot")
    s = Settings()
    s.auto_approve = args.auto
    s.validate()
    live = args.trade or args.trade2
    price_action = args.trade2 or args.dry2
    broker = Broker(s, live_orders=live)
    strategy = PRICE_ACTION if price_action else SMA

    acct = broker.account()
    log.info("Account %s  equity $%s  options BP $%s  mode=%s  strategy=%s  buys=%s",
             acct.account_number, acct.equity, acct.options_buying_power,
             "PAPER ORDERS" if live else "DRY-RUN", "price-action" if price_action else "SMA",
             "AUTO" if args.auto else "ask" if args.trade2 else "automatic" if args.trade else "none")

    if args.once:
        run_cycle(broker, s, strategy)
        return

    if price_action:
        run_price_action_loop(broker, s, strategy, log)

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
        time.sleep(s.poll_minutes * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
