"""Live detectors for the 10 patterns in price_action_use_cases.py, on 5-minute bars.

That file describes each pattern in words, using made-up prices. The numeric thresholds here
(pivot size, ATR tolerances, retrace range) are one interpretation of those descriptions, and
none of it has been backtested. Each detector is written for the bearish (put) version; the
bullish (call) version runs the same code on a flipped price series.
"""
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone

import numpy as np
import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from .broker import parse_occ
from .contracts import MAX_CAPPED_DTE, pc_ratio
from .ema_rsi_macd import ema, rsi_wilder

BAR_MINUTES = 5
STALE_MINUTES = 15
MIN_BARS = 60
PIVOT_K = 2
SCAN_BARS = 40  # how far back a spike or break may be and still count
TOL_ATR = 0.25  # levels within this many ATRs count as the same level
QM_ZONE = 0.30  # QM zone height, as a fraction of the H-to-HH distance
QUICK_BARS = 6  # retests within this many bars of the LL are "quick"
FLAG_LOW = 0.30  # flag low, as a fraction of the bounce (247.60 in the file's 247-249 flag)
# From 3:30 PM ET on expiry day Alpaca may liquidate, and in-the-money options can auto-exercise.
EXPIRY_EXIT_TIME = time(15, 15)


@dataclass
class Setup:
    pattern: str
    kind: str
    price: float
    stop: float


@dataclass
class Bars:
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    atr: float
    piv: list

    @property
    def n(self):
        return len(self.c)

    @property
    def tol(self):
        return self.atr * TOL_ATR


def pivots(high, low, k=PIVOT_K):
    """Alternating swing points as (index, price, "H" or "L"), oldest first."""
    raw = []
    for i in range(k, len(high) - k):
        # Equal neighbours are common at swing points, so the first of a tie counts as the pivot.
        if high[i] == high[i - k:i + k + 1].max() and high[i] > high[i - 1]:
            raw.append((i, float(high[i]), "H"))
        if low[i] == low[i - k:i + k + 1].min() and low[i] < low[i - 1]:
            raw.append((i, float(low[i]), "L"))
    out = []
    for p in raw:
        if out and out[-1][2] == p[2]:
            better = p[1] > out[-1][1] if p[2] == "H" else p[1] < out[-1][1]
            if better:
                out[-1] = p
        else:
            out.append(p)
    return out


def _atr(h, l, c, n=14):
    prev = np.roll(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))[1:]
    return float(tr[-n:].mean())


def _make(o, h, l, c):
    return Bars(o, h, l, c, _atr(h, l, c), pivots(h, l))


def _rejected(x, zone_low, zone_high):
    """Last bar reached the zone, closed red in its lower half and did not close above the zone."""
    return (x.h[-1] >= zone_low and x.c[-1] < x.o[-1] and x.c[-1] <= zone_high
            and x.c[-1] < (x.h[-1] + x.l[-1]) / 2)


def _crossed_below(x, level):
    return x.c[-1] < level <= x.c[-2]


def _clusters(points, tol, min_touches):
    """Levels touched at least min_touches times: (level, index of last touch)."""
    out = []
    for i, (_, anchor) in enumerate(points):
        group = [p for p in points[i:] if abs(p[1] - anchor) <= tol]
        if len(group) >= min_touches:
            out.append((float(np.mean([p[1] for p in group])), group[-1][0]))
    return out


def _pivot_points(x, kind):
    return [(i, p) for i, p, t in x.piv if t == kind]


def _recent_qms(x):
    """Newest-first H, L, HH, LL sequences: (h1, h2, l2, index of the LL, index in the pivot list)."""
    p = x.piv
    for end in range(len(p) - 1, 2, -1):
        (_, h1, t1), (_, l1, t2), (_, h2, t3), (i_ll, l2, t4) = p[end - 3:end + 1]
        if (t1, t2, t3, t4) == ("H", "L", "H", "L") and h2 > h1 and l2 < l1:
            yield h1, h2, l2, i_ll, end


def _qm_retest(x):
    """QM Quick Retest, QM Late Retest and Continuation QM: first touch of the zone, rejected."""
    for h1, h2, _, i_ll, end in _recent_qms(x):
        zone_high = h1 + QM_ZONE * (h2 - h1)
        before_last = x.h[i_ll + 1:-1]
        if len(before_last) < 2 or before_last.max() >= h1 or not _rejected(x, h1, zone_high):
            continue
        p = x.piv
        in_downtrend = end >= 5 and p[end - 5][2] == "H" and p[end - 5][1] > h2
        if in_downtrend:
            name = "Continuation QM"
        else:
            name = "QM Quick Retest" if x.n - 1 - i_ll <= QUICK_BARS else "QM Late Retest"
        return Setup(name, "put", float(x.c[-1]), h2 + x.tol)
    return None


def _qm_reentry(x):
    """A small QM forms inside the main QM zone and its low breaks."""
    for h1, h2, _, i_ll, _ in _recent_qms(x):
        zone_high = h1 + 0.5 * (h2 - h1)
        touches = np.nonzero(x.h[i_ll + 1:] >= h1)[0]
        if not len(touches):
            continue
        start = i_ll + 1 + int(touches[0])
        if x.h[start:].max() > h2 or x.n - start < 8:
            continue
        small = pivots(x.h[start:], x.l[start:], k=1)
        if small and small[-1][2] == "H":  # a bounce after the small LL doesn't change the structure
            small = small[:-1]
        if len(small) < 4:
            continue
        (_, sh1, a), (_, sl1, b), (_, sh2, c), (_, sl2, d) = small[-4:]
        if (a, b, c, d) == ("H", "L", "H", "L") and sh2 > sh1 and sl2 < sl1 and _crossed_below(x, sl2):
            return Setup("QM Re-Entry", "put", float(x.c[-1]), max(zone_high, sh2) + x.tol)
    return None


def _fakeout(x):
    """Sweep above resistance, close back inside, break a low, then a first retest is rejected."""
    lows = _pivot_points(x, "L")
    for level, last_touch in _clusters(_pivot_points(x, "H"), x.tol, 2):
        for i in range(max(last_touch + 1, x.n - SCAN_BARS), x.n - 2):
            if not (x.h[i] > level + x.tol and x.c[i] < level):
                continue
            prior = [p for j, p in lows if j < i]
            if not prior:
                continue
            broke = np.nonzero(x.c[i + 1:-1] < prior[-1])[0]
            if not len(broke):
                continue
            j = i + 1 + int(broke[0])
            if x.h[j:-1].max() < level and _rejected(x, level, float(x.h[i])):
                return Setup("Fakeout V1 (Default)", "put", float(x.c[-1]), float(x.h[i]) + x.tol)
    return None


def _sr_flip(x):
    """Support breaks, price comes back to it for the first time and is rejected."""
    for level, last_touch in _clusters(_pivot_points(x, "L"), x.tol, 2):
        for i in range(max(last_touch + 1, x.n - SCAN_BARS), x.n - 2):
            if x.c[i] < level - x.tol:
                if x.h[i + 1:-1].max() < level - x.tol and _rejected(x, level - x.tol, level + x.atr):
                    return Setup("Fakeout V2 (SR Flip)", "put", float(x.c[-1]), level + x.atr + x.tol)
                break
    return None


def _flag(x):
    """A sharp drop, a 25-62% bounce that has topped out, then a close below the flag low."""
    p = x.piv
    for end in range(len(p) - 1, 0, -1):
        (i_h, top, t1), (i_l, bottom, t2) = p[end - 1], p[end]
        if (t1, t2) != ("H", "L") or x.n - 1 - i_l > 20:
            continue
        since = x.h[i_l + 1:-1]
        if top - bottom < 3 * x.atr or i_l - i_h > 15 or len(since) < 4 or x.l[i_l + 1:].min() < bottom:
            continue
        flag_high = float(since.max())
        retrace = (flag_high - bottom) / (top - bottom)
        topped_out = len(since) - 1 - int(since.argmax()) >= PIVOT_K
        flag_low = bottom + FLAG_LOW * (flag_high - bottom)
        if 0.25 <= retrace <= 0.62 and topped_out and _crossed_below(x, flag_low):
            return Setup("Flag B", "put", float(x.c[-1]), flag_high + x.tol)
    return None


def _mpl(x):
    """Stops hunted above a level touched 3+ times, then a bearish engulfing close below the prior low."""
    engulf = x.c[-1] < x.o[-1] and x.o[-1] >= x.c[-2] and x.c[-1] < x.l[-2]
    if not engulf:
        return None
    spike = float(x.h[-3:].max())
    for level, last_touch in _clusters(_pivot_points(x, "H"), x.tol, 3):
        between = x.h[last_touch + 1:-3]
        if (x.n - 1 - last_touch <= SCAN_BARS and spike > level + x.tol
                and (not len(between) or between.max() <= level + x.tol)):
            return Setup("MPL (Stop Hunt)", "put", float(x.c[-1]), spike + x.tol)
    return None


def _compression(x):
    """Rising lows into a flat ceiling with shrinking swings, then the last higher low breaks."""
    p = x.piv[-6:]
    if [t for _, _, t in p] != ["L", "H", "L", "H", "L", "H"]:
        return None
    l1, h1, l2, h2, l3, h3 = [v for _, v, _ in p]
    ceiling_flat = max(h1, h2, h3) - min(h1, h2, h3) <= x.atr
    shrinking = h1 - l1 > h2 - l2 > h3 - l3
    if l1 < l2 < l3 and ceiling_flat and shrinking and _crossed_below(x, l3):
        return Setup("Compression (CP)", "put", float(x.c[-1]), max(h1, h2, h3) + x.tol)
    return None


def _three_drive(x):
    """Three higher highs with a smaller second pullback, then the last pullback low breaks."""
    p = x.piv[-5:]
    if [t for _, _, t in p] != ["H", "L", "H", "L", "H"] or x.n - 1 - p[-1][0] > 12:
        return None
    d1, p1, d2, p2, d3 = [v for _, v, _ in p]
    if d1 < d2 < d3 and d1 - p1 > d2 - p2 and _crossed_below(x, p2):
        return Setup("3 Drive", "put", float(x.c[-1]), d3 + x.tol)
    return None


# Most specific first: a fakeout also has the H, L, HH, LL shape of a QM, plus a tested level.
DETECTORS = [_mpl, _qm_reentry, _fakeout, _qm_retest, _sr_flip, _flag, _compression, _three_drive]


def _first_setup(x):
    for find in DETECTORS:
        setup = find(x)
        if setup:
            return setup
    return None


def detect(df):
    """Return the Setup that fires on the last bar of df, or None. Conflicting sides cancel out."""
    if len(df) < MIN_BARS:
        return None
    o, h, l, c = [df[col].to_numpy(dtype=float) for col in ("open", "high", "low", "close")]
    put = _first_setup(_make(o, h, l, c))
    flipped = _first_setup(_make(-o, -l, -h, -c))
    if put and flipped:
        return None
    if flipped:
        return Setup(flipped.pattern, "call", -flipped.price, -flipped.stop)
    return put


def recent_setup(df, bars):
    """Newest setup on any of the last `bars` closed bars, so bars between checks aren't missed."""
    for back in range(max(bars, 1)):
        setup = detect(df.iloc[:len(df) - back])
        if setup:
            return setup
    return None


def _momentum_agrees(close, kind):
    rsi = rsi_wilder(close, 14)
    rsi_avg = rsi.rolling(14).mean()
    macd = ema(close, 12) - ema(close, 26)
    signal = ema(macd, 9)
    if kind == "put":
        return rsi.iloc[-1] < rsi_avg.iloc[-1] and macd.iloc[-1] < signal.iloc[-1]
    return rsi.iloc[-1] > rsi_avg.iloc[-1] and macd.iloc[-1] > signal.iloc[-1]


def momentum_confirms(df, kind):
    """RSI vs its average line and MACD vs its signal line must agree on the 5m and 10m."""
    grouped = df.close.resample("10min")
    complete = grouped.last()[grouped.count() == 2]
    return bool(_momentum_agrees(df.close, kind) and _momentum_agrees(complete, kind))


def fetch_5m(stock_client, symbol, days=7):
    """Closed 5m regular-session bars, indexed in US/Eastern."""
    now = datetime.now(timezone.utc)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
        start=now - timedelta(days=days),
        feed=DataFeed.IEX,
    )
    df = stock_client.get_stock_bars(req).df
    if df.empty:
        return df
    df = df.xs(symbol, level="symbol").tz_convert("America/New_York").between_time("09:30", "15:55").copy()
    return df[df.index + pd.Timedelta(minutes=BAR_MINUTES) <= now]


def expiry_day_exit(symbol, s):
    """Reason to close a position that expires today, once it's late enough, else None."""
    _, expiry = parse_occ(symbol)
    now = pd.Timestamp.now("America/New_York")
    if expiry and (expiry < now.date() or (expiry == now.date() and now.time() >= EXPIRY_EXIT_TIME)):
        return "expiry-day exit"
    return None


def price_action_signal(broker, symbol, s):
    """Return (kind or None, last price, reason) using patterns, momentum and the put/call ratio."""
    df = fetch_5m(broker.stocks, symbol)
    if df.empty:
        return None, None, "no bars"
    price = float(df.close.iloc[-1])
    age = datetime.now(timezone.utc) - df.index[-1].to_pydatetime()
    if broker.live_orders and age > timedelta(minutes=STALE_MINUTES):
        return None, price, f"latest bar is {age} old"

    setup = recent_setup(df, s.poll_minutes // BAR_MINUTES)
    if not setup:
        return None, price, None
    label = f"{setup.pattern} {setup.kind}, stop {setup.stop:.2f}"
    if not momentum_confirms(df, setup.kind):
        return None, price, f"{label}, blocked: RSI/MACD disagree"
    ratio = pc_ratio(broker.trading, symbol, price, 0, MAX_CAPPED_DTE)
    if ratio is None:
        return None, price, f"{label}, blocked: no put/call data"
    if (setup.kind == "put" and ratio <= 1) or (setup.kind == "call" and ratio >= 1):
        return None, price, f"{label}, blocked: P/C {ratio:.2f}"
    return setup.kind, price, f"{label}, P/C {ratio:.2f}"
