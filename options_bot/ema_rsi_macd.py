"""EMA 9/21 + RSI(14) + MACD(12/26/9) setup on 15-minute bars.

Long side follows ema_rsi_macd_setup_spec.md exactly (Entry A breakout, Entry B pullback/reclaim,
signal states in spec section 4). The short side (for puts) is a mirror image of those rules;
it is not in the spec.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# Spec section 7 defaults
RSI_BREAKOUT = 60
RSI_FLOOR = 40
VOL_MULT = 2.0
TOUCH_TOL = 0.002
TREND_BARS = 4
CHOP_PCT = 0.001

BULLISH_STATES = {"RECLAIM", "BREAKOUT", "PULLBACK", "TRENDING"}
BEARISH_STATES = {"REJECT", "BREAKDOWN", "RALLY", "BROKEN", "BEARISH"}


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi_wilder(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def add_indicators(df):
    df["ema9"] = ema(df.close, 9)
    df["ema21"] = ema(df.close, 21)
    df["volma"] = df.volume.rolling(20).mean()
    df["rsi"] = rsi_wilder(df.close, 14)
    macd = ema(df.close, 12) - ema(df.close, 26)
    df["macd"], df["signal"] = macd, ema(macd, 9)
    df["mhist"] = df.macd - df.signal
    return df


def add_signals(df):
    p = df.shift(1)
    vol_ok = df.volume >= VOL_MULT * df.volma

    # --- Long side (spec section 3) ---
    df["entry_a"] = (
        (df.close > df.ema9) & (df.close > df.ema21)
        & ((p.close <= p.ema21) | (df.close >= df.close.rolling(20).max()))
        & (p.rsi < RSI_BREAKOUT) & (df.rsi >= RSI_BREAKOUT)
        & (df.macd > df.signal) & (df.mhist > p.mhist)
        & vol_ok
    )
    up_trend = (df.ema9 > df.ema21).rolling(TREND_BARS).sum() == TREND_BARS
    df["pullback"] = up_trend & (df.low <= df.ema21 * (1 + TOUCH_TOL)) & (df.close >= df.ema21) & (df.rsi >= RSI_FLOOR)
    df["entry_b"] = df.pullback.shift(1, fill_value=False) & (df.close > df.ema9) & (df.macd > -0.001 * df.close)
    df["exit_long"] = (df.rsi < RSI_FLOOR) | ((df.macd < df.signal) & (df.macd < 0)) | (df.close < df.ema21)

    # --- Short side: mirror of the long rules (for puts) ---
    df["entry_a_short"] = (
        (df.close < df.ema9) & (df.close < df.ema21)
        & ((p.close >= p.ema21) | (df.close <= df.close.rolling(20).min()))
        & (p.rsi > RSI_FLOOR) & (df.rsi <= RSI_FLOOR)
        & (df.macd < df.signal) & (df.mhist < p.mhist)
        & vol_ok
    )
    down_trend = (df.ema9 < df.ema21).rolling(TREND_BARS).sum() == TREND_BARS
    df["rally"] = down_trend & (df.high >= df.ema21 * (1 - TOUCH_TOL)) & (df.close <= df.ema21) & (df.rsi <= RSI_BREAKOUT)
    df["entry_b_short"] = df.rally.shift(1, fill_value=False) & (df.close < df.ema9) & (df.macd < 0.001 * df.close)

    # --- Chop filter (spec: EMAs within 0.1% for most of last 10 bars) ---
    tangled = ((df.ema9 - df.ema21).abs() / df.close) < CHOP_PCT
    df["chop"] = tangled.rolling(10).sum() >= 6
    return df


def state_of(row):
    """Spec section 4 states, plus mirrored bearish ones. Most specific first."""
    if row.entry_b:
        return "RECLAIM"
    if row.entry_a:
        return "BREAKOUT"
    if row.entry_b_short:
        return "REJECT"
    if row.entry_a_short:
        return "BREAKDOWN"
    if row.pullback:
        return "PULLBACK"
    if row.rally:
        return "RALLY"
    if row.close > row.ema9 > row.ema21 and row.rsi > 50 and row.macd > 0:
        return "TRENDING"
    if row.close < row.ema21 and row.rsi < RSI_FLOOR:
        return "BROKEN"
    if row.close < row.ema21 and row.ema9 < row.ema21:
        return "BEARISH"
    if row.chop and RSI_FLOOR <= row.rsi <= RSI_BREAKOUT:
        return "BASE"
    return "NEUTRAL"


def fetch_15m(stock_client, symbols, days=12, regular_hours=True):
    """15m bars per symbol, indexed in US/Eastern. Regular session only by default (spec section 6),
    since IEX extended-hours bars are too thin for the volume filter."""
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(15, TimeFrameUnit.Minute),
        start=datetime.now(timezone.utc) - timedelta(days=days),
        feed=DataFeed.IEX,
    )
    df = stock_client.get_stock_bars(req).df
    if df.empty:
        return {}
    out = {}
    for sym, g in df.groupby(level="symbol"):
        g = g.droplevel("symbol").tz_convert("America/New_York")
        if regular_hours:
            g = g.between_time("09:30", "15:45")  # bar start times; last bar covers 15:45-16:00
        out[sym] = g.copy()
    return out


def evaluate(bars, lookback=4):
    """Run the setup on one symbol's bars. Returns a summary dict for the latest bar."""
    if len(bars) < 60:
        return None
    df = add_signals(add_indicators(bars))
    last = df.iloc[-1]
    recent = df.iloc[-lookback:]
    fired = [name for name in ("entry_a", "entry_b", "entry_a_short", "entry_b_short") if recent[name].any()]
    return {
        "time": df.index[-1],
        "close": last.close,
        "ema9": last.ema9,
        "ema21": last.ema21,
        "rsi": last.rsi,
        "macd": last.macd,
        "hist": last.mhist,
        "vol_x": last.volume / last.volma if last.volma else float("nan"),
        "chop": bool(last.chop),
        "state": state_of(last),
        "recent_signals": fired,
        "long_stop": min(last.low, last.ema21),
        "short_stop": max(last.high, last.ema21),
    }
