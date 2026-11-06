"""Strategy 3: 5-minute zone test, then a close outside the zone.

  1. Zones come from swing points: swing high = body top to wick high; swing low = wick low to body bottom.
  2. Price comes back and touches the zone.
  3. Another candle closes outside it: above the zone buys a call, below it buys a put.
  4. Stop is just outside the other side of the zone, measured on the stock price.
  5. Target is the next zone in the trade's direction if it's at least 1.5R away, otherwise 2R.
"""
import json
import logging
import os
import time as clock
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from .broker import OCC_RE, parse_occ
from .contracts import MAX_CAPPED_DTE, _closest_to_target, meets_min_delta, quote_contracts
from .price_action import STALE_MINUTES, _atr, fetch_5m

log = logging.getLogger("bot")
LEVELS_PATH = Path(__file__).resolve().parent.parent / "logs" / "zone_levels.json"


@dataclass(frozen=True)
class Strategy3Config:
    swing_bars: int = 3  # bars on each side that a swing point must beat
    lookback_bars: int = 156  # about two trading days of 5-minute bars
    touch_window: int = 12  # the touch must be within this many bars before the close outside
    stop_buffer_atr: float = 0.1
    rr_target: float = 2.0
    min_rr: float = 1.5


@dataclass(frozen=True)
class Zone:
    bottom: float
    top: float
    formed: int  # first bar index at which this swing point is confirmed

    def touched_by(self, high, low):
        return high >= self.bottom and low <= self.top


@dataclass(frozen=True)
class ZoneSignal:
    side: str
    entry: float
    stop: float
    target: float
    zone: Zone

    @property
    def kind(self):
        return "call" if self.side == "long" else "put"

    @property
    def reward_risk(self):
        return abs(self.target - self.entry) / abs(self.entry - self.stop)


def find_zones(df, swing_bars):
    """Swing-high zones (body top to wick high) and swing-low zones (wick low to body bottom)."""
    opens, highs, lows, closes = [df[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close")]
    zones = []
    for i in range(swing_bars, len(df) - swing_bars):
        before_high, after_high = highs[i - swing_bars:i], highs[i + 1:i + swing_bars + 1]
        if highs[i] > before_high.max() and highs[i] >= after_high.max():
            body_top = max(opens[i], closes[i])
            if highs[i] > body_top:
                zones.append(Zone(float(body_top), float(highs[i]), i + swing_bars))
        before_low, after_low = lows[i - swing_bars:i], lows[i + 1:i + swing_bars + 1]
        if lows[i] < before_low.min() and lows[i] <= after_low.min():
            body_bottom = min(opens[i], closes[i])
            if lows[i] < body_bottom:
                zones.append(Zone(float(lows[i]), float(body_bottom), i + swing_bars))
    return zones


def detect_zone_break(df, cfg=Strategy3Config()):
    """The best setup whose close outside the zone is the last bar of df, or None."""
    df = df.iloc[-cfg.lookback_bars:]
    if len(df) < 2 * cfg.swing_bars + 16:
        return None
    last = len(df) - 1
    bar, prev_close = df.iloc[last], float(df.close.iloc[last - 1])
    atr = _atr(*[df[c].to_numpy(dtype=float) for c in ("high", "low", "close")])
    zones = find_zones(df.iloc[:last], cfg.swing_bars)
    close = float(bar.close)
    best = None
    for z in zones:
        window = df.iloc[max(z.formed + 1, last - cfg.touch_window):last]
        if not any(z.touched_by(h, l) for h, l in zip(window.high, window.low)):
            continue
        # This must be the first close outside since the touch, in a candle of that colour.
        if close > z.top and prev_close <= z.top and close > bar.open:
            side, stop = "long", z.bottom - cfg.stop_buffer_atr * atr
            target = min((o.bottom for o in zones if o.bottom > close), default=None)
        elif close < z.bottom and prev_close >= z.bottom and close < bar.open:
            side, stop = "short", z.top + cfg.stop_buffer_atr * atr
            target = max((o.top for o in zones if o.top < close), default=None)
        else:
            continue
        risk = abs(close - stop)
        if risk <= 0:
            continue
        # Use the next zone only if it's at least min_rr away, otherwise the R-multiple target.
        if target is None or abs(target - close) / risk < cfg.min_rr:
            target = close + cfg.rr_target * risk if side == "long" else close - cfg.rr_target * risk
        signal = ZoneSignal(side, close, float(stop), float(target), z)
        if best is None or signal.reward_risk > best.reward_risk:
            best = signal
    return best if best and best.reward_risk >= cfg.min_rr else None


class ZoneLevels:
    """Stock-price stop and target for each contract, kept across bot restarts."""

    def __init__(self, path=LEVELS_PATH):
        self.path = Path(path)
        try:
            self.levels = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.levels = {}

    def get(self, contract):
        return self.levels.get(contract)

    def set(self, contract, kind, stop, target):
        today = date.today()
        self.levels = {c: v for c, v in self.levels.items() if (parse_occ(c)[1] or today) >= today}
        self.levels[contract] = {"kind": kind, "stop": stop, "target": target}
        self._save()

    def set_opening(self, contract):
        self.levels[contract] = {"kind": "opening"}
        self._save()

    def _save(self):
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        for attempt in range(3):
            try:
                self.path.parent.mkdir(exist_ok=True)
                tmp.write_text(json.dumps(self.levels), encoding="utf-8")
                os.replace(tmp, self.path)
                return
            except OSError as e:  # the levels still apply in memory for this run
                if attempt == 2:
                    log.warning("Could not save zone levels (%s); kept in memory only", e)
                else:
                    clock.sleep(0.2)


ZONE_LEVELS = ZoneLevels()
_pending = {}  # latest signal per (ticker, kind), saved against the contract once it's bought


def zone_signal(broker, symbol, s):
    """Return (kind or None, last price, reason) for the zone setup."""
    df = fetch_5m(broker.stocks, symbol)
    if df.empty:
        return None, None, "no bars"
    price = float(df.close.iloc[-1])
    age = datetime.now(timezone.utc) - df.index[-1].to_pydatetime()
    if broker.live_orders and age > timedelta(minutes=STALE_MINUTES):
        return None, price, f"latest bar is {age} old"
    signal = detect_zone_break(df)
    if not signal:
        return None, price, None
    _pending[(symbol, signal.kind)] = signal
    return signal.kind, price, (f"Zone Test + Close Outside {signal.kind}, zone {signal.zone.bottom:.2f}-"
                                f"{signal.zone.top:.2f}, stop {signal.stop:.2f}, target {signal.target:.2f}, "
                                f"RR {signal.reward_risk:.1f}")


def remember_levels(cand):
    """Save the stop and target for a contract the bot just bought."""
    match = OCC_RE.match(cand.symbol)
    kind = "call" if match["cp"] == "C" else "put"
    signal = _pending.get((match["root"], kind))
    if signal:
        ZONE_LEVELS.set(cand.symbol, kind, signal.stop, signal.target)


def zone_level_exit(broker, p, s):
    """Reason to close once the stock reaches the zone stop or target; None without saved levels."""
    levels = ZONE_LEVELS.get(p.symbol)
    if not levels:
        return None
    if levels["kind"] == "opening":
        return opening_exit(p)
    root, _ = parse_occ(p.symbol)
    try:
        req = StockLatestTradeRequest(symbol_or_symbols=root, feed=DataFeed.IEX)
        price = float(broker.stocks.get_stock_latest_trade(req)[root].price)
    except Exception as e:  # the option stop and expiry exit still apply this minute
        log.warning("%s: could not get a stock price for the zone exit (%s)", root, e)
        return None
    is_call = levels["kind"] == "call"
    if (price <= levels["stop"]) if is_call else (price >= levels["stop"]):
        return f"zone stop, {root} {price:.2f}"
    if (price >= levels["target"]) if is_call else (price <= levels["target"]):
        return f"zone target, {root} {price:.2f}"
    return None


@dataclass(frozen=True)
class OpeningConfig:
    min_gap: float = 0.01  # premarket price at least this far above the previous close
    min_body_of_range: float = 0.60  # candle closes near its high, with little wick
    min_range: float = 0.003  # the candle's high-low range, as a fraction of its open
    max_contract_cost: float = 50.0  # one contract must cost less than this
    take_profit_pct: float = 0.20
    stop_loss_pct: float = 0.20
    first_check: time = time(9, 31)
    last_check: time = time(9, 40)


def fetch_extended_1m(stock_client, symbol, days=5):
    """Closed 1-minute bars including premarket, indexed in US/Eastern."""
    now = datetime.now(timezone.utc)
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                           start=now - timedelta(days=days), feed=DataFeed.IEX)
    df = stock_client.get_stock_bars(req).df
    if df.empty:
        return df
    df = df.xs(symbol, level="symbol").tz_convert("America/New_York")
    return df[df.index + pd.Timedelta(minutes=1) <= now]


def opening_signal(df, day, cfg=OpeningConfig()):
    """Why this day's first 1-minute candle qualifies, or None. Premarket up, then an aggressive green candle."""
    if df.empty:
        return None
    today = df[df.index.date == day]
    earlier = df[df.index.date < day].between_time("09:30", "15:59")
    premarket = today.between_time("04:00", "09:29")
    first = today[today.index.time == time(9, 30)]
    if earlier.empty or premarket.empty or first.empty:
        return None
    previous_close = float(earlier.close.iloc[-1])
    premarket_high = float(premarket.high.max())
    gap = float(premarket.close.iloc[-1]) / previous_close - 1
    bar = first.iloc[0]
    candle_range = bar.high - bar.low
    if (gap < cfg.min_gap or bar.close <= bar.open or bar.close <= premarket_high or candle_range <= 0
            or (bar.close - bar.open) / candle_range < cfg.min_body_of_range
            or candle_range / bar.open < cfg.min_range):
        return None
    return f"premarket {gap:+.1%}, high {premarket_high:.2f}, first minute {bar.open:.2f} to {bar.close:.2f}"


def pick_opening_contract(broker, symbol, price, s, cfg=OpeningConfig()):
    """The call closest to the target delta that costs under the limit at the ask, or None."""
    candidates = quote_contracts(broker.trading, broker.options, symbol, "call", price, 0, MAX_CAPPED_DTE, 0.10,
                                 s.min_open_interest, s.max_spread_pct)
    allowed = [c for c in candidates if c.ask * 100 < cfg.max_contract_cost and meets_min_delta(c)]
    return _closest_to_target(allowed, price, s) if allowed else None


_opening_done = {"day": None}


def opening_check(broker, s, cfg=OpeningConfig()):
    """Once a day, just after the first 1-minute candle closes, buy calls on qualifying tickers."""
    now = pd.Timestamp.now("America/New_York")
    if _opening_done["day"] == now.date() or not cfg.first_check <= now.time() <= cfg.last_check:
        return
    _opening_done["day"] = now.date()
    if broker.daily_loss_hit():
        return
    open_count = len(broker.option_positions())
    busy = broker.busy_underlyings()
    for symbol in s.watchlist:
        if open_count >= s.max_open_positions:
            log.info("OPENING: max open positions (%d) reached.", s.max_open_positions)
            return
        if symbol in busy:
            continue
        try:
            df = fetch_extended_1m(broker.stocks, symbol)
            reason = opening_signal(df, now.date(), cfg)
            if not reason:
                continue
            cand = pick_opening_contract(broker, symbol, float(df.close.iloc[-1]), s, cfg)
            if not cand:
                log.info("OPENING: %s qualifies (%s) but no call costs under $%.0f", symbol, reason, cfg.max_contract_cost)
                continue
            broker.buy(cand, f"OPENING call, {reason}, delta {cand.delta:+.2f}", at_ask=True)
            if broker.live_orders:
                ZONE_LEVELS.set_opening(cand.symbol)
            open_count += 1
        except Exception as e:  # one bad ticker must not stop the rest
            log.warning("OPENING: %s failed (%s)", symbol, e)


def opening_exit(p, cfg=OpeningConfig()):
    """Sell an opening trade at +20% or -20%."""
    plpc = float(p.unrealized_plpc or 0)
    if plpc >= cfg.take_profit_pct:
        return f"opening take profit {plpc:+.0%}"
    if plpc <= -cfg.stop_loss_pct:
        return f"opening stop {plpc:+.0%}"
    return None
