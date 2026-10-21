# AI_bots: Alpaca paper options bot

A bot that buys calls/puts on an Alpaca **paper** account based on a signal, and manages exits.
It refuses to run unless `ALPACA_PAPER_TRADE=true`.

## Run

```
pip install -r requirements.txt
python run_bot.py --once     # one cycle, dry-run: logs what it would do, places nothing
python run_bot.py            # loop every 15 min during market hours, dry-run
python run_bot.py --trade    # loop and submit real paper orders
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

The SMA crossover is a placeholder to exercise the plumbing, not a tested edge.
Swap in your own logic by editing `signal_for` in `options_bot/strategy.py`.
