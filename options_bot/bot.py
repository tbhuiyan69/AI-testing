"""One cycle of the bot: manage exits, then look for new entries."""
import logging
from dataclasses import dataclass
from typing import Callable

from .broker import days_to_expiry
from .contracts import pick_capped_contract, pick_contract
from .price_action import expiry_day_exit, price_action_signal
from .strategy import signal_for

log = logging.getLogger("bot")


def dte_exit(symbol, s):
    dte = days_to_expiry(symbol)
    return f"time exit, {dte} DTE" if dte is not None and dte <= s.exit_dte else None


def pct_take_profit(p, s):
    plpc = float(p.unrealized_plpc or 0)
    return f"take profit {plpc:+.0%}" if plpc >= s.take_profit_pct else None


def quick_take_profit(p, s):
    """Take profit on more than the dollar target or more than the percent target, whichever comes first."""
    pl, plpc = float(p.unrealized_pl or 0), float(p.unrealized_plpc or 0)
    if pl > s.quick_profit_dollars or plpc > s.quick_profit_pct:
        return f"take profit ${pl:+.2f} ({plpc:+.1%})"
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


SMA = Strategy(sma_signal, pick_contract, dte_exit)
PRICE_ACTION = Strategy(price_action_signal, pick_capped_contract, expiry_day_exit,
                        time_exit_at_market=True, show_blocked=True, take_profit=quick_take_profit)


def manage_exits(broker, s, strategy):
    for p in broker.option_positions():
        plpc = float(p.unrealized_plpc or 0)
        dte = days_to_expiry(p.symbol)
        time_exit = strategy.time_exit(p.symbol, s)
        take_profit = strategy.take_profit(p, s)
        if time_exit and strategy.time_exit_at_market:
            broker.close_now(p, time_exit)
        elif take_profit:
            broker.sell(p, take_profit)
        elif plpc <= -s.stop_loss_pct:
            broker.sell(p, f"stop loss {plpc:+.0%}")
        elif time_exit:
            broker.sell(p, time_exit)
        else:
            log.info("HOLD %s  P/L %+.0f%%  %s DTE", p.symbol, plpc * 100, dte)


def look_for_entries(broker, s, strategy):
    if broker.daily_loss_hit():
        log.warning("Daily loss limit (%.0f%%) hit; no new entries today.", s.daily_loss_limit_pct * 100)
        return

    open_count = len(broker.option_positions())
    busy = broker.busy_underlyings()
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
        broker.buy(cand, f"{reason}, delta {delta}")
        open_count += 1


def run_cycle(broker, s, strategy=SMA):
    manage_exits(broker, s, strategy)
    look_for_entries(broker, s, strategy)
