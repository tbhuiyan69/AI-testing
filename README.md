# AI_bots: Alpaca paper options bot

A bot that buys calls/puts on an Alpaca **paper** account based on a signal, and manages exits.
It refuses to run unless `ALPACA_PAPER_TRADE=true`.

## Run

```
pip install -r requirements.txt
python run_bot.py --once     # one cycle, dry-run: logs what it would do, places nothing
python run_bot.py            # loop every 15 min during market hours, dry-run
python run_bot.py --trade    # loop and submit real paper orders
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

All numbers are configurable; see `bot.env.example` (copy to `.env`).

## Price-action strategy (`--dry2`, `--trade2`)

Replaces the SMA signal, and only buys short-dated contracts within price limits. Exits are checked every minute, and new setups each time a 5-minute bar closes.

`--trade2` lists every contract it wants to buy with its price, quantity and cost, then asks yes or no for each. Anything but `y` or `yes` within 60 seconds (`BOT_APPROVAL_SECONDS`) skips the trade. Exits are not asked about. It needs an open terminal; without one, no buys go through. `--dry2` shows the same list without asking or ordering.

1. **Pattern** (`options_bot/price_action.py`): one of the 10 in `price_action_use_cases.py` fires on the last closed bar. Puts use the bearish version, calls the mirror image.
2. **Momentum**: RSI vs its 14-bar average and MACD vs signal agree on both 5m and 10m.
3. **Flow**: put/call ratio above 1 backs a put, below 1 backs a call. It's computed from open interest in the 0-5 day expiry window.
4. **Contract price**: the limit price per share must be under $0.80 for same-day expiry, at most $1.20 for 1-3 days, and at most $1.50 for 4-5 days. Contracts further out are never bought. Delta must be at least 0.15. Among those allowed, it picks the closest to 0.40 delta, with the same open-interest and spread filters.

**Exits:** take profit at $5 or more, or more than 3%, on the position, whichever comes first (`BOT_QUICK_PROFIT_DOLLARS`, `BOT_QUICK_PROFIT_PCT`), and the -35% stop as above. These replace the +50% target. The 7-day exit is also replaced: a position is sold at market at 3:15 PM ET on its expiry day, because Alpaca may liquidate from 3:30 PM and in-the-money options can auto-exercise. No same-day contracts are bought after 3:00 PM ET.

The pattern's stop level is logged with the order but not enforced. The detectors are one interpretation of written descriptions and have not been backtested.

The SMA crossover is a placeholder to exercise the plumbing, not a tested edge.
Swap in your own logic by editing `signal_for` in `options_bot/strategy.py`.
