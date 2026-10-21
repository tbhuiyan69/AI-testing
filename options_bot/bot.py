"""One cycle of the bot: manage exits, then look for new entries."""
import logging

from .broker import days_to_expiry
from .contracts import pick_contract
from .strategy import signal_for

log = logging.getLogger("bot")


def manage_exits(broker, s):
    for p in broker.option_positions():
        plpc = float(p.unrealized_plpc or 0)
        dte = days_to_expiry(p.symbol)
        if plpc >= s.take_profit_pct:
            broker.sell(p, f"take profit {plpc:+.0%}")
        elif plpc <= -s.stop_loss_pct:
            broker.sell(p, f"stop loss {plpc:+.0%}")
        elif dte is not None and dte <= s.exit_dte:
            broker.sell(p, f"time exit, {dte} DTE")
        else:
            log.info("HOLD %s  P/L %+.0f%%  %s DTE", p.symbol, plpc * 100, dte)


def look_for_entries(broker, s):
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
            kind, price = signal_for(broker.stocks, symbol, s)
        except Exception as e:  # keep one bad symbol from killing the cycle
            log.warning("%s: could not compute signal (%s)", symbol, e)
            continue
        if not kind:
            log.info("%s: no signal (last %.2f)", symbol, price or 0)
            continue

        cand, why = pick_contract(broker.trading, broker.options, symbol, kind, price, s)
        if not cand:
            log.info("%s: %s signal but %s", symbol, kind, why)
            continue
        delta = f"{cand.delta:+.2f}" if cand.delta is not None else "n/a"
        broker.buy(cand, f"{symbol} SMA{s.fast_sma}/{s.slow_sma} cross -> {kind}, delta {delta}")
        open_count += 1


def run_cycle(broker, s):
    manage_exits(broker, s)
    look_for_entries(broker, s)
