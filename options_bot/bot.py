"""One cycle of the bot: manage exits, then look for new entries."""
import logging
from dataclasses import dataclass
from typing import Callable

from .approval import ask_yes_no
from .broker import days_to_expiry
from .contracts import pick_capped_contract, pick_contract
from .price_action import expiry_day_exit, price_action_signal
from .profit_lock import ProfitLocks
from .strategy import signal_for

log = logging.getLogger("bot")
PROFIT_LOCKS = ProfitLocks()


def dte_exit(symbol, s):
    dte = days_to_expiry(symbol)
    return f"time exit, {dte} DTE" if dte is not None and dte <= s.exit_dte else None


def pct_take_profit(p, s):
    plpc = float(p.unrealized_plpc or 0)
    return f"take profit {plpc:+.0%}" if plpc >= s.take_profit_pct else None


def profit_lock_exit(p, s):
    """No profit cap: once a position is above the target, sell only if it falls back to the target."""
    pl = float(p.unrealized_pl or 0)
    if pl > s.profit_lock_dollars:
        if not PROFIT_LOCKS.is_locked(p.symbol):
            PROFIT_LOCKS.lock(p.symbol)
            log.info("PROFIT LOCK %s at $%.2f profit; will sell if it falls to $%.2f",
                     p.symbol, pl, s.profit_lock_dollars)
        return None
    if PROFIT_LOCKS.is_locked(p.symbol):
        return f"profit lock ${pl:+.2f}"
    return None


def sma_signal(broker, symbol, s):
    kind, price = signal_for(broker.stocks, symbol, s)
    return kind, price, f"{symbol} SMA{s.fast_sma}/{s.slow_sma} cross -> {kind}"


@dataclass(frozen=True)
class Strategy:
    signal: Callable
    pick: Callable
    time_exit: Callable
    time_exit_at_market: bool = False
    show_blocked: bool = False
    take_profit: Callable = pct_take_profit
    take_profit_at_market: bool = False
    confirm_buys: bool = False


SMA = Strategy(sma_signal, pick_contract, dte_exit)
PRICE_ACTION = Strategy(price_action_signal, pick_capped_contract, expiry_day_exit,
                        time_exit_at_market=True, show_blocked=True, take_profit=profit_lock_exit,
                        take_profit_at_market=True, confirm_buys=True)


def manage_exits(broker, s, strategy):
    positions = broker.option_positions()
    PROFIT_LOCKS.keep_only(p.symbol for p in positions)
    for p in positions:
        try:
            exit_position(broker, s, strategy, p)
        except Exception as e:  # one rejected sell must not stop exit checks on the other positions
            log.error("Exit failed for %s: %s", p.symbol, e)


def exit_position(broker, s, strategy, p):
    plpc = float(p.unrealized_plpc or 0)
    dte = days_to_expiry(p.symbol)
    time_exit = strategy.time_exit(p.symbol, s)
    take_profit = strategy.take_profit(p, s)
    if time_exit and strategy.time_exit_at_market:
        broker.close_now(p, time_exit)
    elif take_profit and strategy.take_profit_at_market:
        broker.close_now(p, take_profit)
    elif take_profit:
        broker.sell(p, take_profit)
    elif plpc <= -s.stop_loss_pct:
        broker.sell(p, f"stop loss {plpc:+.0%}")
    elif time_exit:
        broker.sell(p, time_exit)
    else:
        log.info("HOLD %s  P/L %+.0f%%  %s DTE", p.symbol, plpc * 100, dte)


def confirm_and_buy(broker, s, targets, open_count):
    """List every contract the bot wants to buy with its price, then ask about each one."""
    log.info("TARGETS: %d contract(s) ready to buy", len(targets))
    for cand, reason in targets:
        qty, limit = broker.order_size(cand)
        log.info("  %-22s exp %s (%d DTE)  price %.2f  bid %.2f / ask %.2f  qty %d  cost $%.2f  [%s]",
                 cand.symbol, cand.expiration, cand.dte, limit, cand.bid, cand.ask, qty, limit * 100 * qty, reason)

    for cand, reason in targets:
        qty, limit = broker.order_size(cand)
        if qty < 1:
            broker.buy(cand, reason)  # logs why it's too expensive; sends nothing
            continue
        if open_count >= s.max_open_positions:
            log.info("Max open positions (%d) reached; not asking about the rest.", s.max_open_positions)
            return
        contract_cost = limit * 100
        question = f"BUY {qty} x {cand.symbol} at {limit:.2f}, cost ${contract_cost * qty:,.2f}?"
        auto = s.auto_approve and contract_cost < s.auto_approve_below
        if broker.live_orders and not auto and not ask_yes_no(question, s.approval_seconds):
            log.info("NOT APPROVED %s", cand.symbol)
            continue
        broker.buy(cand, reason)
        open_count += 1


def look_for_entries(broker, s, strategy):
    if broker.daily_loss_hit():
        log.warning("Daily loss limit (%.0f%%) hit; no new entries today.", s.daily_loss_limit_pct * 100)
        return

    open_count = len(broker.option_positions())
    busy = broker.busy_underlyings()
    targets = []
    for symbol in s.watchlist:
        if open_count >= s.max_open_positions:
            log.info("Max open positions (%d) reached.", s.max_open_positions)
            return
        if symbol in busy:
            continue
        try:
            kind, price, reason = strategy.signal(broker, symbol, s)
        except Exception as e:  # keep one bad symbol from killing the cycle
            log.warning("%s: could not compute signal (%s)", symbol, e)
            continue
        if not kind:
            detail = f" [{reason}]" if reason and strategy.show_blocked else ""
            log.info("%s: no signal (last %.2f)%s", symbol, price or 0, detail)
            continue

        cand, why = strategy.pick(broker.trading, broker.options, symbol, kind, price, s)
        if not cand:
            log.info("%s: %s signal but %s", symbol, kind, why)
            continue
        delta = f"{cand.delta:+.2f}" if cand.delta is not None else "n/a"
        if strategy.confirm_buys:
            targets.append((cand, f"{reason}, delta {delta}"))
            continue
        broker.buy(cand, f"{reason}, delta {delta}")
        open_count += 1

    if targets:
        confirm_and_buy(broker, s, targets, open_count)


def run_cycle(broker, s, strategy=SMA, entries=True):
    manage_exits(broker, s, strategy)
    if entries:
        look_for_entries(broker, s, strategy)
