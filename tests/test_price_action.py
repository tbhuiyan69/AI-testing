"""Run with: python -m unittest discover tests"""
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from alpaca.trading.enums import OrderSide, OrderType, PositionIntent
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from options_bot import bot, contracts, price_action as pa
from options_bot.approval import ask_yes_no
from options_bot.broker import Broker
from options_bot.profit_lock import ProfitLocks
from options_bot.zone_strategy import Strategy3Config, ZoneLevels, detect_zone_break, find_zones
from options_bot import zone_strategy as zs
from options_bot.config import Settings
from options_bot.contracts import Candidate, premium_cap, within_premium_cap
from options_bot.price_action import detect, momentum_confirms, pivots
from price_action_use_cases import CASES


def path(*legs, start=235.0, lead=60):
    """Closes running through (price, bars) legs, after a slow rise from start."""
    first = legs[0][0]
    closes = list(np.linspace(start, first, lead))
    for price, bars in legs[1:]:
        closes += list(np.linspace(closes[-1], price, bars + 1)[1:])
    return closes


def frame(closes, *last_bars):
    """OHLC with small wicks from closes; last_bars are exact (open, high, low, close) bars."""
    rows, prev = [], closes[0]
    for c in closes:
        rows.append((prev, max(prev, c) + 0.05, min(prev, c) - 0.05, c))
        prev = c
    rows += list(last_bars)
    idx = pd.date_range("2026-11-03 09:30", periods=len(rows), freq="5min", tz="America/New_York")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def flip(df):
    out = pd.DataFrame(index=df.index)
    out["open"], out["high"], out["low"], out["close"] = 1000 - df.open, 1000 - df.low, 1000 - df.high, 1000 - df.close
    return out


REJECT = (249.75, 250.30, 249.20, 249.30)
QM = [(250.0, 1), (247.5, 5), (252.0, 5), (245.8, 6)]

SCENARIOS = {
    "QM Quick Retest": frame(path(*QM, (249.7, 5)), REJECT),
    "QM Late Retest": frame(path(*QM, (248.0, 4), (247.0, 5), (248.5, 4), (249.7, 3)), REJECT),
    "Continuation QM": frame(
        path((253.0, 1), (249.0, 5), (250.0, 3), (247.5, 5), (252.0, 5), (245.8, 6), (249.7, 5)), REJECT),
    "QM Re-Entry": frame(
        path(*QM, (249.8, 5), (250.6, 2), (250.3, 2), (251.1, 3), (250.1, 3), (250.5, 2)),
        (250.5, 250.55, 249.9, 250.0)),
    "Fakeout V1 (Default)": frame(
        path((250.0, 1), (248.0, 5), (249.95, 5), (247.8, 5), (249.9, 5)),
        (249.9, 251.0, 249.4, 249.5), (249.5, 249.55, 248.2, 248.3), (248.3, 248.35, 247.2, 247.3),
        (247.3, 247.35, 246.6, 246.7), (246.7, 248.0, 246.6, 247.9), (247.9, 249.3, 247.8, 249.2),
        (249.75, 250.3, 249.2, 249.3)),
    "Fakeout V2 (SR Flip)": frame(
        path((251.0, 1), (248.0, 5), (250.0, 5), (248.05, 5), (249.5, 4), (246.0, 6), (247.2, 4)),
        (247.6, 248.2, 247.0, 247.1)),
    "Flag B": frame(
        path((252.0, 1), (247.0, 6), (247.6, 2), (248.4, 2), (249.0, 2), (248.7, 1), (248.5, 1), (248.4, 1)),
        (248.4, 248.45, 247.4, 247.5)),
    "MPL (Stop Hunt)": frame(
        path((250.0, 1), (248.5, 5), (250.05, 5), (248.5, 5), (249.95, 5), (249.2, 4), (249.9, 3)),
        (249.9, 250.9, 249.7, 250.3), (250.4, 250.45, 248.6, 248.8)),
    "Compression (CP)": frame(
        path((249.0, 1), (245.5, 5), (250.0, 6), (247.0, 5), (250.0, 5), (248.0, 5), (250.0, 4), (248.5, 3)),
        (248.5, 248.55, 247.5, 247.6)),
    "3 Drive": frame(
        path((249.0, 1), (247.0, 5), (250.0, 5), (248.8, 4), (251.0, 5), (249.2, 5)),
        (249.2, 249.25, 248.3, 248.4)),
}


class Patterns(unittest.TestCase):
    def test_scenarios_cover_every_pattern_in_the_file(self):
        self.assertEqual(set(SCENARIOS), {c.name for c in CASES})

    def test_put_fires_on_the_last_bar_with_stop_above_price(self):
        for name, df in SCENARIOS.items():
            with self.subTest(name):
                s = detect(df)
                self.assertIsNotNone(s)
                self.assertEqual((s.pattern, s.kind), (name, "put"))
                self.assertGreater(s.stop, s.price)

    def test_mirror_fires_as_call_with_stop_below_price(self):
        for name, df in SCENARIOS.items():
            with self.subTest(name):
                s = detect(flip(df))
                self.assertIsNotNone(s)
                self.assertEqual((s.pattern, s.kind), (name, "call"))
                self.assertLess(s.stop, s.price)

    def test_does_not_fire_one_bar_earlier(self):
        for name, df in SCENARIOS.items():
            with self.subTest(name):
                s = detect(df.iloc[:-1])
                self.assertFalse(s and s.pattern == name)

    def test_steady_trend_has_no_setup(self):
        self.assertIsNone(detect(frame(list(np.linspace(200, 260, 120)))))

    def test_too_few_bars_has_no_setup(self):
        self.assertIsNone(detect(frame(list(np.linspace(200, 210, 30)))))

    def test_pivots_alternate(self):
        kinds = [t for _, _, t in pivots(*[SCENARIOS["3 Drive"][c].to_numpy() for c in ("high", "low")])]
        self.assertTrue(all(a != b for a, b in zip(kinds, kinds[1:])))


def v_shape(down=120, up=60):
    return frame(list(np.linspace(300, 240, down)) + list(np.linspace(240, 270, up + 1)[1:]))


class MacdSequence(unittest.TestCase):
    start = pd.Timestamp("2026-11-05 10:00", tz="America/New_York")

    def at(self, minutes):
        return self.start + pd.Timedelta(minutes=minutes)

    def in_order(self, t5, t10, t15, now):
        return pa.crosses_in_sequence(self.at(t5), self.at(t10), self.at(t15), self.at(now))

    def test_five_then_ten_then_fifteen_passes(self):
        self.assertTrue(self.in_order(0, 10, 15, 15))
        self.assertTrue(self.in_order(0, 0, 0, 0))

    def test_wrong_order_fails(self):
        self.assertFalse(self.in_order(10, 0, 15, 15))
        self.assertFalse(self.in_order(0, 20, 15, 20))
        self.assertFalse(self.in_order(20, 10, 15, 20))

    def test_old_fifteen_minute_cross_fails(self):
        self.assertTrue(self.in_order(0, 10, 15, 45))
        self.assertFalse(self.in_order(0, 10, 15, 50))

    def test_crosses_too_far_apart_fail(self):
        self.assertTrue(self.in_order(0, 60, 120, 120))
        self.assertFalse(self.in_order(0, 60, 125, 125))

    def first_pass(self, df, kind):
        for end in range(100, len(df) + 1):
            if pa.macd_sequence_confirmed(df.iloc[:end], kind):
                return end
        return None

    def test_turn_up_confirms_a_call_after_the_low_not_at_it(self):
        df = v_shape()
        end = self.first_pass(df, "call")
        self.assertIsNotNone(end)
        self.assertGreater(end, 120)
        self.assertIsNone(self.first_pass(df, "put"))

    def test_turn_down_confirms_a_put(self):
        df = flip(v_shape())
        self.assertIsNotNone(self.first_pass(df, "put"))
        self.assertIsNone(self.first_pass(df, "call"))

    def test_only_a_five_minute_cross_so_far_does_not_pass(self):
        df = v_shape()
        five_only = next(end for end in range(121, len(df)) if pa._last_macd_cross(df.close.iloc[:end], 5)[1])
        self.assertFalse(pa.macd_sequence_confirmed(df.iloc[:five_only], "call"))

    def test_long_finished_move_no_longer_passes(self):
        self.assertFalse(pa.macd_sequence_confirmed(v_shape(up=120), "call"))

    def test_too_few_bars_does_not_pass(self):
        self.assertFalse(pa.macd_sequence_confirmed(v_shape().iloc[:20], "call"))


class Momentum(unittest.TestCase):
    def falling(self):
        return frame(list(np.linspace(240, 255, 60)) + list(np.linspace(255, 240, 30)))

    def confirms(self, df_1m, kind, sequence=True):
        with patch.object(pa, "macd_sequence_confirmed", return_value=sequence):
            return momentum_confirms(self.falling(), df_1m, kind)

    def test_falling_one_minute_rsi_backs_a_put_only(self):
        self.assertTrue(self.confirms(self.falling(), "put"))
        self.assertFalse(self.confirms(self.falling(), "call"))

    def test_rising_one_minute_rsi_backs_a_call_only(self):
        self.assertTrue(self.confirms(flip(self.falling()), "call"))
        self.assertFalse(self.confirms(flip(self.falling()), "put"))

    def test_blocked_without_the_macd_sequence(self):
        self.assertFalse(self.confirms(self.falling(), "put", sequence=False))

    def test_too_few_one_minute_bars_blocks_the_trade(self):
        self.assertFalse(self.confirms(self.falling().iloc[-29:], "put"))


def bars(*rows):
    idx = pd.date_range("2026-11-03 09:30", periods=len(rows), freq="5min", tz="America/New_York")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def rising(start, count, step=0.5):
    return [(start + i * step, start + i * step + 0.3, start + i * step - 0.1, start + i * step + 0.2) for i in range(count)]


def zone_setup():
    """A swing high with a wick zone of 110.00-111.00, a pullback, a touch, then a close above it."""
    rows = rising(100, 20)
    rows += [(109.8, 111.0, 109.6, 110.0)]  # swing high: body top 110.0, wick high 111.0
    rows += [(110.0, 110.2, 108.5, 108.6), (108.6, 108.8, 107.5, 107.6), (107.6, 107.8, 106.5, 106.6),
             (106.6, 106.8, 106.0, 106.2), (106.2, 107.5, 106.1, 107.4), (107.4, 109.0, 107.3, 108.9)]
    rows += [(108.9, 110.4, 108.8, 110.3)]  # touches the zone and closes inside it
    rows += [(110.3, 112.0, 110.2, 111.8)]  # first close above the zone, green
    return bars(*rows)


def flip_bars(df):
    out = pd.DataFrame(index=df.index)
    out["open"], out["high"], out["low"], out["close"] = 500 - df.open, 500 - df.low, 500 - df.high, 500 - df.close
    return out


class ZoneDetection(unittest.TestCase):
    def test_swing_high_zone_runs_from_body_top_to_wick_high(self):
        self.assertTrue(any(z.bottom == 110.0 and z.top == 111.0 for z in find_zones(zone_setup(), 3)))

    def test_touch_then_close_above_buys_a_call_with_stop_below_the_zone(self):
        sig = detect_zone_break(zone_setup())
        self.assertEqual((sig.side, sig.kind, sig.entry), ("long", "call", 111.8))
        self.assertLess(sig.stop, 110.0)
        self.assertGreater(sig.target, sig.entry)
        self.assertGreaterEqual(sig.reward_risk, 1.5)

    def test_touch_then_close_below_buys_a_put_with_stop_above_the_zone(self):
        sig = detect_zone_break(flip_bars(zone_setup()))
        self.assertEqual((sig.side, sig.kind), ("short", "put"))
        self.assertGreater(sig.stop, sig.entry)
        self.assertLess(sig.target, sig.entry)

    def test_target_falls_back_to_two_r_without_a_zone_beyond(self):
        sig = detect_zone_break(zone_setup())
        self.assertAlmostEqual(sig.reward_risk, 2.0)

    def test_next_zone_is_the_target_when_far_enough(self):
        far = Strategy3Config()
        df = zone_setup()
        with patch.object(zs, "find_zones", side_effect=lambda d, k: find_zones(d, k) + [zs.Zone(125.0, 126.0, 5)]):
            sig = detect_zone_break(df, far)
        self.assertEqual(sig.target, 125.0)

    def test_second_close_outside_is_not_a_new_entry(self):
        df = zone_setup()
        last = df.iloc[-1]
        again = bars(*[tuple(r) for r in df.itertuples(index=False)], (last.close, 113.0, 111.7, 112.8))
        self.assertIsNone(detect_zone_break(again))

    def test_red_close_above_is_not_an_entry(self):
        rows = [tuple(r) for r in zone_setup().itertuples(index=False)]
        rows[-1] = (112.5, 112.6, 110.2, 111.8)
        self.assertIsNone(detect_zone_break(bars(*rows)))

    def test_close_outside_without_a_touch_is_not_an_entry(self):
        rows = [tuple(r) for r in zone_setup().itertuples(index=False)]
        rows[-2] = (108.9, 109.5, 108.8, 109.4)  # stops short of the zone
        rows[-1] = (109.4, 112.0, 109.3, 111.8)
        self.assertIsNone(detect_zone_break(bars(*rows)))

    def test_too_few_bars_has_no_setup(self):
        self.assertIsNone(detect_zone_break(zone_setup().iloc[:10]))


class ZoneExits(unittest.TestCase):
    CALL = "AAA261106C00100000"
    PUT = "AAA261106P00100000"

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.path = Path(d.name) / "levels.json"
        patcher = patch.object(zs, "ZONE_LEVELS", ZoneLevels(self.path))
        patcher.start()
        self.addCleanup(patcher.stop)

    def exit_at(self, contract, price):
        stocks = SimpleNamespace(get_stock_latest_trade=lambda req: {"AAA": SimpleNamespace(price=price)})
        return zs.zone_level_exit(SimpleNamespace(stocks=stocks), SimpleNamespace(symbol=contract), None)

    def test_call_sells_at_stop_or_target_only(self):
        zs.ZONE_LEVELS.set(self.CALL, "call", 95.0, 110.0)
        self.assertIn("zone stop", self.exit_at(self.CALL, 95.0))
        self.assertIn("zone target", self.exit_at(self.CALL, 110.0))
        self.assertIsNone(self.exit_at(self.CALL, 100.0))

    def test_put_levels_are_mirrored(self):
        zs.ZONE_LEVELS.set(self.PUT, "put", 105.0, 90.0)
        self.assertIn("zone stop", self.exit_at(self.PUT, 105.0))
        self.assertIn("zone target", self.exit_at(self.PUT, 90.0))
        self.assertIsNone(self.exit_at(self.PUT, 100.0))

    def test_position_without_levels_has_no_zone_exit(self):
        self.assertIsNone(self.exit_at(self.CALL, 1.0))

    def test_levels_survive_a_restart(self):
        zs.ZONE_LEVELS.set(self.CALL, "call", 95.0, 110.0)
        self.assertEqual(ZoneLevels(self.path).get(self.CALL)["stop"], 95.0)

    def test_price_error_does_not_stop_exit_checks(self):
        zs.ZONE_LEVELS.set(self.CALL, "call", 95.0, 110.0)
        def fail(req):
            raise RuntimeError("no data")
        broker = SimpleNamespace(stocks=SimpleNamespace(get_stock_latest_trade=fail))
        self.assertIsNone(zs.zone_level_exit(broker, SimpleNamespace(symbol=self.CALL), None))

    def test_bought_contract_saves_the_latest_signal(self):
        zs._pending[("AAA", "call")] = zs.ZoneSignal("long", 100.0, 95.0, 110.0, zs.Zone(96.0, 97.0, 1))
        self.addCleanup(zs._pending.clear)
        zs.remember_levels(SimpleNamespace(symbol=self.CALL))
        self.assertEqual(zs.ZONE_LEVELS.get(self.CALL), {"kind": "call", "stop": 95.0, "target": 110.0})

    def test_strategy_sells_at_market_on_the_level_and_has_no_profit_cap(self):
        closed, sold = [], []
        position = SimpleNamespace(symbol=self.CALL, unrealized_plpc="3.00", unrealized_pl="300")
        broker = SimpleNamespace(option_positions=lambda: [position], close_now=lambda _, why: closed.append(why),
                                 sell=lambda _, why: sold.append(why))
        s = SimpleNamespace(short_dte_stop_loss_pct=0.50)
        with patch.object(bot, "PROFIT_LOCKS", ProfitLocks(Path(tempfile.mkdtemp()) / "l.json")):
            strategy = bot.Strategy(bot.ZONES.signal, bot.ZONES.pick, lambda *_: None, take_profit=bot.no_take_profit,
                                    stop_loss=bot.short_dte_stop_loss, level_exit=lambda *_: "zone target, AAA 110.00")
            bot.manage_exits(broker, s, strategy)
            self.assertEqual((closed, sold), (["zone target, AAA 110.00"], []))
            closed.clear()
            quiet = bot.Strategy(bot.ZONES.signal, bot.ZONES.pick, lambda *_: None, take_profit=bot.no_take_profit,
                                 stop_loss=bot.short_dte_stop_loss, level_exit=lambda *_: None)
            bot.manage_exits(broker, s, quiet)
        self.assertEqual((closed, sold), ([], []))

    def test_option_stop_still_applies(self):
        sold = []
        position = SimpleNamespace(symbol=self.CALL, unrealized_plpc="-0.50", unrealized_pl="-50")
        broker = SimpleNamespace(option_positions=lambda: [position], sell=lambda _, why: sold.append(why))
        strategy = bot.Strategy(bot.ZONES.signal, bot.ZONES.pick, lambda *_: None, take_profit=bot.no_take_profit,
                                stop_loss=bot.short_dte_stop_loss, level_exit=lambda *_: None)
        with patch.object(bot, "PROFIT_LOCKS", ProfitLocks(Path(tempfile.mkdtemp()) / "l.json")):
            bot.manage_exits(broker, SimpleNamespace(short_dte_stop_loss_pct=0.50), strategy)
        self.assertEqual(sold, ["stop loss -50%"])


def opening_bars(pre_close=102.0, pre_high=102.5, first=(102.6, 103.5, 102.5, 103.4), prev_close=100.0):
    day, prior = date(2026, 11, 6), date(2026, 11, 5)
    rows = [(f"{prior} 15:59", prev_close, prev_close, prev_close, prev_close),
            (f"{day} 08:00", pre_close, pre_high, pre_close - 0.2, pre_close),
            (f"{day} 09:29", pre_close, pre_close, pre_close, pre_close),
            (f"{day} 09:30", *first)]
    idx = pd.DatetimeIndex([r[0] for r in rows]).tz_localize("America/New_York")
    return pd.DataFrame([r[1:] for r in rows], columns=["open", "high", "low", "close"], index=idx), day


class OpeningTrade(unittest.TestCase):
    def test_gap_up_and_aggressive_first_candle_qualifies(self):
        df, day = opening_bars()
        self.assertIn("premarket +2.0%", zs.opening_signal(df, day))

    def test_failing_conditions_do_not_qualify(self):
        cases = {
            "small premarket move": dict(pre_close=100.5, pre_high=100.6, first=(100.6, 101.5, 100.5, 101.4)),
            "red candle": dict(first=(103.4, 103.5, 102.5, 102.6)),
            "closes below premarket high": dict(pre_high=104.0),
            "long upper wick": dict(first=(102.6, 106.0, 102.5, 103.4)),
            "tiny candle": dict(first=(102.60, 102.70, 102.59, 102.69), pre_high=102.5),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                df, day = opening_bars(**kwargs)
                self.assertIsNone(zs.opening_signal(df, day))

    def test_missing_first_candle_does_not_qualify(self):
        df, day = opening_bars()
        self.assertIsNone(zs.opening_signal(df.iloc[:-1], day))

    def test_exit_at_twenty_percent_either_way(self):
        for plpc, expected in [("0.20", "take profit"), ("0.50", "take profit"), ("-0.20", "stop"), ("-0.60", "stop")]:
            with self.subTest(plpc):
                self.assertIn(expected, zs.opening_exit(SimpleNamespace(unrealized_plpc=plpc)))
        for plpc in ("0.19", "0", "-0.19"):
            with self.subTest(plpc):
                self.assertIsNone(zs.opening_exit(SimpleNamespace(unrealized_plpc=plpc)))

    def contract(self, ask, delta=0.30):
        return Candidate("AAA261106C00100000", 100.0, date.today(), ask - 0.01, ask, delta)

    def pick(self, candidates):
        s = SimpleNamespace(min_open_interest=100, max_spread_pct=0.10, target_delta=0.40)
        with patch.object(zs, "quote_contracts", return_value=candidates):
            return zs.pick_opening_contract(SimpleNamespace(trading=None, options=None), "AAA", 100.0, s)

    def test_contract_must_cost_under_fifty_dollars_at_the_ask(self):
        cheap = self.contract(0.49)
        self.assertIs(self.pick([self.contract(0.50), cheap]), cheap)
        self.assertIsNone(self.pick([self.contract(0.50)]))
        self.assertIsNone(self.pick([self.contract(0.30, delta=0.10)]))

    def run_check(self, hour=9, minute=31, busy=(), positions=0, buy_contract=True):
        zs._opening_done["day"] = None
        self.addCleanup(zs._opening_done.update, {"day": None})
        bought = []
        broker = SimpleNamespace(stocks=None, live_orders=True, daily_loss_hit=lambda: False,
                                 option_positions=lambda: [None] * positions, busy_underlyings=lambda: set(busy),
                                 buy=lambda c, why, at_ask=False: bought.append((c.symbol, at_ask)))
        s = SimpleNamespace(watchlist=["AAA"], max_open_positions=3)
        now = pd.Timestamp(f"{date.today()} {hour:02d}:{minute:02d}", tz="America/New_York")
        levels = ZoneLevels(Path(tempfile.mkdtemp()) / "levels.json")
        with (patch.object(pd.Timestamp, "now", return_value=now), patch.object(zs, "ZONE_LEVELS", levels),
              patch.object(zs, "fetch_extended_1m", return_value=opening_bars()[0]),
              patch.object(zs, "opening_signal", return_value="why"),
              patch.object(zs, "pick_opening_contract", return_value=self.contract(0.40) if buy_contract else None)):
            zs.opening_check(broker, s)
            zs.opening_check(broker, s)
        return bought, levels

    def test_buys_once_at_the_ask_and_saves_the_opening_exit(self):
        bought, levels = self.run_check()
        self.assertEqual(bought, [("AAA261106C00100000", True)])
        self.assertEqual(levels.get("AAA261106C00100000"), {"kind": "opening"})

    def test_only_runs_in_the_opening_window(self):
        self.assertEqual(self.run_check(minute=30)[0], [])
        self.assertEqual(self.run_check(minute=41)[0], [])

    def test_respects_held_tickers_and_the_position_limit(self):
        self.assertEqual(self.run_check(busy=["AAA"])[0], [])
        self.assertEqual(self.run_check(positions=3)[0], [])

    def test_buy_at_ask_uses_the_ask_as_the_limit(self):
        b = Broker.__new__(Broker)
        b.s = SimpleNamespace(max_contracts=1, max_trade_dollars=1000.0)
        sent = []
        b._submit = lambda symbol, qty, side, limit, reason: sent.append(limit)
        b.buy(SimpleNamespace(symbol="X", ask=0.45, mid=0.40), "why", at_ask=True)
        self.assertEqual(sent, [0.45])


class FakeBroker:
    stocks = trading = options = None

    def __init__(self, live_orders=True):
        self.live_orders = live_orders
        self.bought = []

    def daily_loss_hit(self):
        return False

    def option_positions(self):
        return []

    def busy_underlyings(self):
        return set()

    def buy(self, cand, reason):
        self.bought.append((cand, reason))


class Gates(unittest.TestCase):
    """Which combinations of pattern, momentum and P/C reach an order."""

    def signal(self, ratio=2.0, momentum=True, live=True, bar_age_minutes=5, skip_momentum=False):
        df = SCENARIOS["QM Quick Retest"].copy()
        last_close = datetime.now(timezone.utc) - timedelta(minutes=bar_age_minutes)
        df.index = pd.date_range(end=last_close, periods=len(df), freq="5min")
        with (patch.object(pa, "fetch_5m", return_value=df),
              patch.object(pa, "pc_ratio", return_value=ratio),
              patch.object(pa, "fetch_1m", return_value=df),
              patch.object(pa, "momentum_confirms", return_value=momentum)):
            return pa.price_action_signal(FakeBroker(live), "AAA", SimpleNamespace(skip_momentum=skip_momentum))

    def test_all_agree_gives_a_put(self):
        kind, _, reason = self.signal()
        self.assertEqual(kind, "put")
        self.assertIn("QM Quick Retest", reason)
        self.assertIn("P/C 2.00", reason)

    def test_blocked_when_momentum_disagrees(self):
        kind, _, reason = self.signal(momentum=False)
        self.assertIsNone(kind)
        self.assertIn("RSI/MACD", reason)

    def test_skip_flag_ignores_momentum_only(self):
        self.assertEqual(self.signal(momentum=False, skip_momentum=True)[0], "put")
        self.assertIsNone(self.signal(momentum=False, skip_momentum=True, ratio=0.5)[0])

    def test_momentum_check_is_on_by_default(self):
        self.assertFalse(Settings().skip_momentum)

    def test_blocked_when_put_call_ratio_disagrees(self):
        for ratio in (0.5, 1.0):
            with self.subTest(ratio):
                self.assertIsNone(self.signal(ratio=ratio)[0])

    def test_blocked_without_put_call_data(self):
        self.assertIsNone(self.signal(ratio=None)[0])

    def test_stale_bars_block_live_orders_only(self):
        self.assertIsNone(self.signal(bar_age_minutes=60)[0])
        self.assertEqual(self.signal(bar_age_minutes=60, live=False)[0], "put")

    def test_signal_reaches_a_buy_with_its_reason(self):
        broker = FakeBroker()
        s = SimpleNamespace(watchlist=["AAA"], max_open_positions=3)
        cand = SimpleNamespace(delta=-0.40)
        strategy = bot.Strategy(
            signal=lambda *_: ("put", 250.0, "QM Quick Retest put, stop 252.20, P/C 2.00"),
            pick=lambda *_: (cand, None),
            time_exit=lambda *_: None,
        )
        bot.look_for_entries(broker, s, strategy)
        self.assertEqual(broker.bought, [(cand, "QM Quick Retest put, stop 252.20, P/C 2.00, delta -0.40")])


def position(pl, plpc, symbol="AAPL261120P00335000", qty="2"):
    return SimpleNamespace(symbol=symbol, unrealized_pl=str(pl), unrealized_plpc=str(plpc), qty=qty,
                           qty_available="0", current_price="1.25")


class ProfitLock(unittest.TestCase):
    s = SimpleNamespace(profit_lock_dollars=5.0, take_profit_pct=0.50, stop_loss_pct=0.35, exit_dte=7,
                        short_dte_stop_loss_pct=0.50)

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "locks.json"
        patcher = patch.object(bot, "PROFIT_LOCKS", ProfitLocks(self.path))
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_exits(self, p, strategy=bot.PRICE_ACTION):
        closed, sold, holds = [], [], []
        broker = SimpleNamespace(option_positions=lambda: [p] if p else [],
                                 close_now=lambda _, why: closed.append(why),
                                 sell=lambda _, why: sold.append(why))
        bot.manage_exits(broker, self.s, strategy)
        return closed, sold

    def test_below_target_keeps_running(self):
        for pl in (4.99, 5.00, 0, -50):
            with self.subTest(pl):
                self.assertEqual(self.run_exits(position(pl, 0.01)), ([], []))

    def test_above_target_locks_and_keeps_running_with_no_cap(self):
        self.assertEqual(self.run_exits(position(5.01, 0.01)), ([], []))
        self.assertEqual(self.run_exits(position(500, 0.60)), ([], []))
        self.assertEqual(self.run_exits(position(5000, 6.00)), ([], []))

    def test_falling_back_to_target_after_lock_sells_at_market(self):
        self.run_exits(position(20, 0.05))
        self.assertEqual(self.run_exits(position(5.00, 0.01)), (["profit lock $+5.00"], []))

    def test_falling_past_target_after_lock_still_sells(self):
        self.run_exits(position(20, 0.05))
        self.assertEqual(self.run_exits(position(-3, -0.01)), (["profit lock $-3.00"], []))

    def test_lock_survives_a_restart(self):
        self.run_exits(position(20, 0.05))
        with patch.object(bot, "PROFIT_LOCKS", ProfitLocks(self.path)):
            self.assertEqual(self.run_exits(position(4, 0.01)), (["profit lock $+4.00"], []))

    def test_closed_position_forgets_its_lock(self):
        self.run_exits(position(20, 0.05))
        self.run_exits(None)
        self.assertEqual(self.run_exits(position(4, 0.01)), ([], []))

    def test_unlocked_position_still_has_the_stop_loss(self):
        self.assertEqual(self.run_exits(position(-400, -0.49)), ([], []))
        self.assertEqual(self.run_exits(position(-500, -0.50)), ([], ["stop loss -50%"]))

    def test_file_error_never_stops_exit_checks_and_lock_still_holds(self):
        with (patch("options_bot.profit_lock.os.replace", side_effect=PermissionError("in use")),
              patch("options_bot.profit_lock.time.sleep")):
            self.assertEqual(self.run_exits(position(20, 0.05)), ([], []))
        self.assertEqual(self.run_exits(position(4, 0.01)), (["profit lock $+4.00"], []))

    def test_damaged_lock_file_starts_empty(self):
        self.path.write_text("not json", encoding="utf-8")
        self.assertFalse(ProfitLocks(self.path).is_locked("X"))

    def test_default_target_is_five(self):
        self.assertEqual(Settings().profit_lock_dollars, 5.0)

    def test_sma_strategy_keeps_its_own_stop_loss(self):
        self.assertEqual(self.run_exits(position(-400, -0.40), bot.SMA), ([], ["stop loss -40%"]))

    def test_sma_strategy_still_waits_for_fifty_percent(self):
        self.assertEqual(self.run_exits(position(120, 0.20), bot.SMA), ([], []))
        self.assertEqual(self.run_exits(position(500, 0.60), bot.SMA), ([], ["take profit +60%"]))
        self.assertFalse(bot.PROFIT_LOCKS.is_locked("AAPL261120P00335000"))


class OrderSize(unittest.TestCase):
    def size(self, ask, max_contracts=1, max_dollars=1000.0):
        b = Broker.__new__(Broker)
        b.s = SimpleNamespace(max_contracts=max_contracts, max_trade_dollars=max_dollars)
        return b.order_size(SimpleNamespace(ask=ask, mid=ask))[0]

    def test_buys_one_contract_however_cheap(self):
        for ask in (0.05, 0.31, 1.15, 9.99):
            with self.subTest(ask):
                self.assertEqual(self.size(ask), 1)

    def test_skips_a_contract_over_the_trade_limit(self):
        self.assertEqual(self.size(10.01), 0)

    def test_default_limit_is_one(self):
        self.assertEqual(Settings().max_contracts, 1)

    def test_buy_sends_one_contract(self):
        b = Broker.__new__(Broker)
        b.s = SimpleNamespace(max_contracts=1, max_trade_dollars=1000.0)
        sent = []
        b._submit = lambda symbol, qty, side, limit, reason: sent.append(qty)
        b.buy(SimpleNamespace(symbol="X", ask=0.31, mid=0.30), "why")
        self.assertEqual(sent, [1])


class OrderIntent(unittest.TestCase):
    def submit(self, side, limit):
        sent = []
        b = Broker.__new__(Broker)
        b.live_orders = True
        b.trading = SimpleNamespace(submit_order=lambda req: sent.append(req) or SimpleNamespace(status="new"))
        with patch.object(Broker, "_journal"):
            b._submit("X261106P00015500", 2, side, limit, "why")
        return sent[0]

    def test_limit_and_market_sells_can_only_close(self):
        for limit in (1.25, None):
            with self.subTest(limit):
                self.assertEqual(self.submit(OrderSide.SELL, limit).position_intent, PositionIntent.SELL_TO_CLOSE)

    def test_buys_can_only_open(self):
        self.assertEqual(self.submit(OrderSide.BUY, 1.25).position_intent, PositionIntent.BUY_TO_OPEN)

    def test_market_close_does_not_use_close_position(self):
        self.assertEqual(self.submit(OrderSide.SELL, None).type, OrderType.MARKET)


class ExitFailures(unittest.TestCase):
    def test_one_failed_sell_does_not_stop_the_next_position(self):
        sold = []

        def sell(p, why):
            if p.symbol == "BAD":
                raise RuntimeError("rejected")
            sold.append(p.symbol)

        positions = [SimpleNamespace(symbol=sym, unrealized_plpc="-0.50", unrealized_pl="-50") for sym in ("BAD", "GOOD")]
        broker = SimpleNamespace(option_positions=lambda: positions, sell=sell)
        s = SimpleNamespace(profit_lock_dollars=5.0, take_profit_pct=0.50, stop_loss_pct=0.35,
                            short_dte_stop_loss_pct=0.50)
        with (patch.object(bot, "PROFIT_LOCKS", ProfitLocks(Path(tempfile.mkdtemp()) / "locks.json")),
              self.assertLogs("bot", "ERROR") as logs):
            bot.manage_exits(broker, s, bot.PRICE_ACTION)
        self.assertEqual(sold, ["GOOD"])
        self.assertIn("Exit failed for BAD", logs.output[0])


class ReplacingSells(unittest.TestCase):
    def broker(self, live, open_orders):
        calls = []
        b = Broker.__new__(Broker)
        b.live_orders = live
        b.trading = SimpleNamespace(get_orders=lambda _: open_orders,
                                    cancel_order_by_id=lambda order_id: calls.append(("cancel", order_id)))
        b._submit = lambda sym, qty, side, limit, why: calls.append(("submit", sym, qty, limit, why))
        return b, calls

    def test_unfilled_sell_is_cancelled_and_replaced_even_with_no_qty_available(self):
        b, calls = self.broker(True, [SimpleNamespace(id="old")])
        with patch("options_bot.broker.time.sleep"):
            b.sell(position(-300, -0.40), "stop loss -40%")
        self.assertEqual(calls, [("cancel", "old"), ("submit", "AAPL261120P00335000", 2, 1.25, "stop loss -40%")])

    def test_dry_run_cancels_nothing(self):
        b, calls = self.broker(False, [SimpleNamespace(id="old")])
        b.sell(position(12, 0.02), "take profit")
        self.assertEqual([c[0] for c in calls], ["submit"])


class FakeKeys:
    """Stands in for msvcrt: hands out typed keys, then reports nothing pressed."""

    def __init__(self, keys):
        self.keys = list(keys)
        self.armed = False

    def kbhit(self):
        if not self.armed:  # the first check is the flush of earlier typing
            self.armed = True
            return False
        return bool(self.keys)

    def getwche(self):
        return self.keys.pop(0)

    def getwch(self):
        return self.keys.pop(0)


class AskYesNo(unittest.TestCase):
    def ask(self, keys, timeout=5, tty=True):
        with (patch.dict(sys.modules, {"msvcrt": FakeKeys(keys)}), patch("sys.stdin.isatty", return_value=tty),
              patch("builtins.print")):
            return ask_yes_no("Buy?", timeout)

    def test_only_y_or_yes_approves(self):
        for keys, expected in [("y\r", True), ("YES\r", True), ("n\r", False), ("\r", False), ("maybe\r", False)]:
            with self.subTest(keys):
                self.assertEqual(self.ask(keys), expected)

    def test_no_answer_in_time_is_no(self):
        self.assertFalse(self.ask("", timeout=0.1))

    def test_y_without_enter_is_no(self):
        self.assertFalse(self.ask("y", timeout=0.1))

    def test_no_terminal_is_no(self):
        self.assertFalse(self.ask("y\r", tty=False))

    def test_ctrl_c_stops_the_bot(self):
        with self.assertRaises(KeyboardInterrupt):
            self.ask("\x03")


class ApprovalBroker:
    trading = options = None

    def __init__(self, live=True, positions=0, qty=8, limit=1.10):
        self.live_orders, self.qty, self.limit = live, qty, limit
        self.positions = [SimpleNamespace(symbol="X")] * positions
        self.bought = []

    def daily_loss_hit(self):
        return False

    def option_positions(self):
        return self.positions

    def busy_underlyings(self):
        return set()

    def order_size(self, cand):
        return self.qty, self.limit

    def buy(self, cand, reason):
        self.bought.append(cand.symbol)


class ApprovingBuys(unittest.TestCase):
    def run_entries(self, broker, answers, max_open=3, auto=False):
        symbols = ["AAA", "BBB", "CCC"]
        s = SimpleNamespace(watchlist=symbols, max_open_positions=max_open, approval_seconds=60, auto_approve=auto,
                            auto_approve_below=100.0)
        pick = lambda _t, _o, sym, *_: (
            Candidate(sym + "CONTRACT", 100.0, date.today() + timedelta(days=3), 1.08, 1.12, 0.3), None)
        strategy = bot.Strategy(signal=lambda _b, sym, _s: ("call", 100.0, f"{sym} setup"), pick=pick,
                                time_exit=lambda *_: None, confirm_buys=True)
        with patch.object(bot, "ask_yes_no", side_effect=answers) as ask:
            bot.look_for_entries(broker, s, strategy)
        return ask

    def test_buys_only_what_is_approved(self):
        broker = ApprovalBroker()
        ask = self.run_entries(broker, [True, False, True])
        self.assertEqual(broker.bought, ["AAACONTRACT", "CCCCONTRACT"])
        self.assertEqual(ask.call_count, 3)

    def test_question_shows_contract_price_and_cost(self):
        ask = self.run_entries(ApprovalBroker(), [False, False, False])
        self.assertEqual(ask.call_args_list[0].args[0], "BUY 8 x AAACONTRACT at 1.10, cost $880.00?")

    def test_every_target_is_listed_before_the_first_question(self):
        with self.assertLogs("bot", "INFO") as logs:
            self.run_entries(ApprovalBroker(), [False, False, False])
        listed = [line for line in logs.output if "exp " in line]
        self.assertEqual(len(listed), 3)
        self.assertTrue(all("price 1.10" in line and "cost $880.00" in line for line in listed))

    def test_stops_asking_once_positions_are_full(self):
        broker = ApprovalBroker(positions=2)
        ask = self.run_entries(broker, [True])
        self.assertEqual(broker.bought, ["AAACONTRACT"])
        self.assertEqual(ask.call_count, 1)

    def test_auto_buys_contracts_under_the_limit_at_full_size_without_asking(self):
        for qty, limit in [(1, 0.99), (10, 0.95)]:
            with self.subTest(qty=qty, limit=limit):
                broker = ApprovalBroker(qty=qty, limit=limit)
                ask = self.run_entries(broker, [], auto=True)
                self.assertEqual(ask.call_count, 0)
                self.assertEqual(len(broker.bought), 3)

    def test_auto_asks_at_or_above_the_limit(self):
        for qty, limit in [(1, 1.00), (8, 1.10)]:
            with self.subTest(qty=qty, limit=limit):
                broker = ApprovalBroker(qty=qty, limit=limit)
                ask = self.run_entries(broker, [False, False, False], auto=True)
                self.assertEqual(ask.call_count, 3)
                self.assertEqual(broker.bought, [])

    def test_auto_still_respects_the_position_limit(self):
        broker = ApprovalBroker(positions=2, qty=1, limit=0.50)
        self.run_entries(broker, [], auto=True)
        self.assertEqual(broker.bought, ["AAACONTRACT"])

    def test_asking_is_the_default(self):
        self.assertFalse(Settings().auto_approve)

    def test_dry_run_never_asks(self):
        broker = ApprovalBroker(live=False)
        ask = self.run_entries(broker, [])
        self.assertEqual(ask.call_count, 0)
        self.assertEqual(len(broker.bought), 3)

    def test_too_expensive_is_not_asked(self):
        broker = ApprovalBroker(qty=0)
        ask = self.run_entries(broker, [])
        self.assertEqual(ask.call_count, 0)

    def test_exit_only_cycle_skips_entries(self):
        calls = []
        with (patch.object(bot, "manage_exits", lambda *_: calls.append("exits")),
              patch.object(bot, "look_for_entries", lambda *_: calls.append("entries"))):
            bot.run_cycle(None, None, bot.PRICE_ACTION, entries=False)
        self.assertEqual(calls, ["exits"])

    def test_sma_strategy_does_not_ask(self):
        self.assertFalse(bot.SMA.confirm_buys)
        self.assertTrue(bot.PRICE_ACTION.confirm_buys)


def candidate(dte, mid, delta=-0.30):
    return Candidate("X", 100.0, date.today() + timedelta(days=dte), mid - 0.01, mid + 0.01, delta)


class PriceLimits(unittest.TestCase):
    def test_caps_by_days_to_expiry(self):
        expected = {0: 0.79, 1: 1.20, 3: 1.20, 4: 1.50, 5: 1.50, 6: None, 30: None, -1: None}
        for dte, cap in expected.items():
            with self.subTest(dte):
                self.assertEqual(premium_cap(dte), cap)

    def test_prices_on_each_side_of_the_limits(self):
        cases = [(0, 0.79, True), (0, 0.80, False), (3, 1.20, True), (3, 1.21, False),
                 (5, 1.50, True), (5, 1.51, False), (6, 0.10, False)]
        for dte, mid, allowed in cases:
            with self.subTest(dte=dte, mid=mid):
                self.assertEqual(within_premium_cap(candidate(dte, mid)), allowed)

    def pick(self, candidates):
        settings = SimpleNamespace(min_open_interest=100, max_spread_pct=0.10, target_delta=0.40)
        with patch.object(contracts, "quote_contracts", return_value=candidates) as quote:
            result = contracts.pick_capped_contract(None, None, "AAA", "put", 100.0, settings)
        return result, quote

    def test_closer_delta_loses_if_over_the_limit(self):
        too_dear, cheap = candidate(3, 1.30, delta=-0.40), candidate(3, 1.00, delta=-0.20)
        (chosen, _), _ = self.pick([too_dear, cheap])
        self.assertIs(chosen, cheap)

    def test_no_contract_when_all_are_over_the_limit(self):
        (chosen, why), _ = self.pick([candidate(0, 0.90), candidate(5, 1.60)])
        self.assertIsNone(chosen)
        self.assertIn("price limits", why)

    def test_contracts_below_minimum_delta_are_never_chosen(self):
        for delta in (-0.01, 0.14, None):
            with self.subTest(delta):
                (chosen, why), _ = self.pick([candidate(3, 1.00, delta=delta)])
                self.assertIsNone(chosen)
                self.assertIn("minimum delta", why)

    def test_minimum_delta_is_inclusive_for_calls_and_puts(self):
        for delta in (-0.15, 0.15):
            with self.subTest(delta):
                c = candidate(3, 1.00, delta=delta)
                (chosen, _), _ = self.pick([c])
                self.assertIs(chosen, c)

    def test_only_asks_for_contracts_up_to_five_days(self):
        _, quote = self.pick([])
        self.assertEqual(quote.call_args.args[6], 5)


def eastern(hour, minute):
    return pd.Timestamp(date.today().isoformat() + f" {hour:02d}:{minute:02d}", tz="America/New_York")


class SameDayTiming(unittest.TestCase):
    def occ(self, dte):
        return f"AAA{(date.today() + timedelta(days=dte)):%y%m%d}C00100000"

    def exit_reason(self, dte, hour, minute):
        with patch.object(pd.Timestamp, "now", return_value=eastern(hour, minute)):
            return pa.expiry_day_exit(self.occ(dte), None)

    def test_expiring_position_is_closed_only_from_3_15(self):
        self.assertIsNone(self.exit_reason(0, 15, 14))
        self.assertIsNotNone(self.exit_reason(0, 15, 15))

    def test_later_expiry_is_not_closed(self):
        self.assertIsNone(self.exit_reason(1, 15, 59))
        self.assertIsNone(self.exit_reason(5, 15, 59))

    def test_position_inside_seven_days_is_not_sold_by_the_old_rule(self):
        self.assertIsNone(self.exit_reason(5, 10, 0))

    def first_dte_requested(self, hour, minute):
        settings = SimpleNamespace(min_open_interest=100, max_spread_pct=0.10, target_delta=0.40)
        with (patch.object(pd.Timestamp, "now", return_value=eastern(hour, minute)),
              patch.object(contracts, "quote_contracts", return_value=[]) as quote):
            contracts.pick_capped_contract(None, None, "AAA", "put", 100.0, settings)
        return quote.call_args.args[5]

    def test_same_day_contracts_stop_at_3_pm(self):
        self.assertEqual(self.first_dte_requested(14, 59), 0)
        self.assertEqual(self.first_dte_requested(15, 0), 1)

    def test_time_exit_uses_a_market_close_ahead_of_profit_rules(self):
        closed, sold = [], []
        position = SimpleNamespace(symbol=self.occ(0), unrealized_plpc="0.90", unrealized_pl="90")
        broker = SimpleNamespace(option_positions=lambda: [position],
                                 close_now=lambda p, why: closed.append(why),
                                 sell=lambda p, why: sold.append(why))
        s = ProfitLock.s
        with (patch.object(pd.Timestamp, "now", return_value=eastern(15, 20)),
              patch.object(bot, "PROFIT_LOCKS", ProfitLocks(Path(tempfile.mkdtemp()) / "locks.json"))):
            bot.manage_exits(broker, s, bot.PRICE_ACTION)
        self.assertEqual((closed, sold), (["expiry-day exit"], []))


if __name__ == "__main__":
    unittest.main()
