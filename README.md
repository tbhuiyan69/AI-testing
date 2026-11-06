# AI_bots: Alpaca paper options bot

A bot that buys calls/puts on an Alpaca **paper** account based on a signal, and manages exits.
It refuses to run unless `ALPACA_PAPER_TRADE=true`.

## Run

```
pip install -r requirements.txt
python run_bot.py --once     # one cycle, dry-run: logs what it would do, places nothing
python run_bot.py            # loop every 15 min during market hours, dry-run
python run_bot.py --trade    # loop and submit real paper orders
python run_bot.py --trade2 --auto   # same, but contracts under $100 each (`BOT_AUTO_APPROVE_BELOW`) buy without asking
python run_bot.py --trade2 --skip-rsi-macd   # ignore the RSI/MACD check for this run
python run_bot.py --dry2     # price-action strategy, log only
python run_bot.py --trade2   # price-action strategy, asks yes/no before each paper buy
```

Logs go to `logs/bot.log`; every order (or dry-run order) is appended to `logs/trades.csv`.

## How it works (each cycle)

1. **Exits** for every open option position: take profit at +50%, stop at -35%, or close at 7 days to expiry.
2. **Entries** for each watchlist symbol without an existing position/open order:
   - Signal (`options_bot/strategy.py`): 10/30-day SMA crossover. Up-cross → call, down-cross → put.
   - Contract (`options_bot/contracts.py`): 21-45 DTE, open interest ≥ 100, bid/ask spread ≤ 10%, closest to 0.40 delta.
   - Order: limit at mid, sized so one trade costs ≤ $1,000.
3. **Guards**: max 3 open positions, no new entries after a 3% daily drawdown.

The bot buys 1 contract per trade (`BOT_MAX_CONTRACTS`), and skips a contract costing over $1,000.

Bot sells are sent as sell-to-close, so Alpaca rejects one that would open a short position.

Only one order-placing bot (`--trade`, `--trade2`, `--trade3`) can run on a PC at a time; a second copy exits straight away. Dry runs aren't limited.

All numbers are configurable; see `bot.env.example` (copy to `.env`).

## Price-action strategy (`--dry2`, `--trade2`)

Replaces the SMA signal, and only buys short-dated contracts within price limits. Exits are checked every minute, and new setups each time a 5-minute bar closes.

`--trade2` lists every contract it wants to buy with its price, quantity and cost, then asks yes or no for each. Anything but `y` or `yes` within 60 seconds (`BOT_APPROVAL_SECONDS`) skips the trade. Exits are not asked about. It needs an open terminal; without one, no buys go through. `--dry2` shows the same list without asking or ordering.

1. **Pattern** (`options_bot/price_action.py`): one of the 10 in `price_action_use_cases.py` fires on the last closed bar. Puts use the bearish version, calls the mirror image.
2. **Momentum**: RSI vs its 14-bar average agrees on the 1m. MACD must cross its signal line on the 5m, then the 10m, then the 15m, in the trade's direction, with the 15m cross in the last 30 minutes and within 2 hours of the 5m cross.
3. **Flow**: put/call ratio above 1 backs a put, below 1 backs a call. It's computed from open interest in the 0-5 day expiry window.
4. **Contract price**: the limit price per share must be under $0.80 for same-day expiry, at most $1.20 for 1-3 days, and at most $1.50 for 4-5 days. Contracts further out are never bought. Delta must be at least 0.15. Among those allowed, it picks the closest to 0.40 delta, with the same open-interest and spread filters.

**Exits:** there's no profit cap. Once a position's profit passes $5 (`BOT_PROFIT_LOCK_DOLLARS`), it's sold at market if the profit falls back to $5, otherwise it keeps running. The bot checks this each minute, so it isn't a stop held at the broker and a sale can land below $5. The lock is kept in `logs/profit_locks.json` across restarts. The stop is -50% (`BOT_SHORT_DTE_STOP_LOSS_PCT`), and this replaces the +50% target. The 7-day exit is also replaced: a position is sold at market at 3:15 PM ET on its expiry day, because Alpaca may liquidate from 3:30 PM and in-the-money options can auto-exercise. No same-day contracts are bought after 3:00 PM ET.

The pattern's stop level is logged with the order but not enforced. The detectors are one interpretation of written descriptions and have not been backtested.

## Zone strategy (`--dry3`, `--trade3`)

`options_bot/zone_strategy.py`. Run this or `--trade2`, never both at once, because each manages every open position.

1. **Zones** on 5-minute bars: each swing high (body top to wick high) and swing low (wick low to body bottom).
2. **Touch**: price comes back into a zone within the last 12 bars.
3. **Close outside**: another candle is the first to close outside it, in that direction. Above the zone buys a call, below it buys a put.
4. **Exits**, checked each minute: the stock reaching the zone stop or target sells at market. The target is the next zone if it's at least 1.5R away, otherwise 2R. Every open position also has a profit stop of $20, or $10 for SOFI (`BOT_ZONE_LOCK_DOLLARS`, `BOT_ZONE_LOCK_BY_TICKER`): once its profit passes that, it keeps running with no cap and sells at market if it falls back to that amount. The -50% option stop and 3:15 PM expiry-day exit apply too.

Contracts, price limits and the 1-contract limit are the same as `--trade2`. RSI, MACD and the put/call ratio are not used. With `--auto`, every target buys right away with no question, as a limit order at the mid that is never re-priced higher.

**Opening trade:** just after 9:31 AM ET each day, once per run, a ticker qualifies if its premarket price is at least 1% above the previous close and its first 1-minute candle is green, closes above the premarket high, closes near its high and has a range of at least 0.3%. The bot immediately buys 1 call at the ask, costing under $50, with no yes/no question, and sells at market at +20% or -20%. The 3:15 PM expiry-day exit still applies.

The SMA crossover is a placeholder to exercise the plumbing, not a tested edge.
Swap in your own logic by editing `signal_for` in `options_bot/strategy.py`.
