"""Entry signal: moving-average crossover on daily bars of the underlying.

This is an example strategy to show the plumbing, not a recommendation.
Replace `signal_for` with your own logic; it only has to return "call", "put" or None.
"""
from datetime import datetime, timedelta, timezone

from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def daily_closes(stock_client, symbol, lookback_days):
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.now(timezone.utc) - timedelta(days=lookback_days * 2 + 10),
        feed=DataFeed.IEX,
    )
    df = stock_client.get_stock_bars(req).df
    if df.empty:
        return None
    return df.xs(symbol, level="symbol")["close"]


def signal_for(stock_client, symbol, settings):
    """Return ("call" | "put" | None, last_price)."""
    closes = daily_closes(stock_client, symbol, settings.slow_sma)
    if closes is None or len(closes) < settings.slow_sma + 1:
        return None, None

    fast = closes.rolling(settings.fast_sma).mean()
    slow = closes.rolling(settings.slow_sma).mean()
    last_price = float(closes.iloc[-1])

    crossed_up = fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]
    crossed_down = fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]
    if crossed_up:
        return "call", last_price
    if crossed_down:
        return "put", last_price
    return None, last_price
