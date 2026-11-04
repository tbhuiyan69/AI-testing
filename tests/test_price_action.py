"""Run with: python -m unittest discover tests"""
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from options_bot import bot, contracts, price_action as pa
from options_bot.broker import Broker
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


class Momentum(unittest.TestCase):
    def falling(self):
        return frame(list(np.linspace(240, 255, 60)) + list(np.linspace(255, 240, 30)))

    def test_falling_price_confirms_put_only(self):
        df = self.falling()
        self.assertTrue(momentum_confirms(df, "put"))
        self.assertFalse(momentum_confirms(df, "call"))

    def test_rising_price_confirms_call_only(self):
        df = flip(self.falling())
        self.assertTrue(momentum_confirms(df, "call"))
        self.assertFalse(momentum_confirms(df, "put"))


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

    def signal(self, ratio=2.0, momentum=True, live=True, bar_age_minutes=5):
        df = SCENARIOS["QM Quick Retest"].copy()
        last_close = datetime.now(timezone.utc) - timedelta(minutes=bar_age_minutes)
        df.index = pd.date_range(end=last_close, periods=len(df), freq="5min")
        with (patch.object(pa, "fetch_5m", return_value=df),
              patch.object(pa, "pc_ratio", return_value=ratio),
              patch.object(pa, "momentum_confirms", return_value=momentum)):
            return pa.price_action_signal(FakeBroker(live), "AAA", SimpleNamespace(poll_minutes=15))

    def test_all_agree_gives_a_put(self):
        kind, _, reason = self.signal()
        self.assertEqual(kind, "put")
        self.assertIn("QM Quick Retest", reason)
        self.assertIn("P/C 2.00", reason)

    def test_blocked_when_momentum_disagrees(self):
        kind, _, reason = self.signal(momentum=False)
        self.assertIsNone(kind)
        self.assertIn("RSI/MACD", reason)

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


class FifteenMinuteChecks(unittest.TestCase):
    def quiet_bars_after(self, df, count):
        last = df.iloc[-1]
        rows = pd.DataFrame([[last.close, last.close + 0.01, last.close - 0.01, last.close]] * count, columns=df.columns,
                            index=pd.date_range(df.index[-1] + pd.Timedelta(minutes=5), periods=count, freq="5min"))
        return pd.concat([df, rows])

    def test_finds_a_setup_from_up_to_two_bars_ago(self):
        for count in (0, 1, 2):
            with self.subTest(count):
                df = self.quiet_bars_after(SCENARIOS["QM Quick Retest"], count)
                self.assertEqual(pa.recent_setup(df, 3).pattern, "QM Quick Retest")

    def test_ignores_a_setup_from_before_the_last_check(self):
        df = self.quiet_bars_after(SCENARIOS["QM Quick Retest"], 3)
        self.assertIsNone(pa.recent_setup(df, 3))

    def test_short_interval_still_checks_the_latest_bar(self):
        self.assertIsNotNone(pa.recent_setup(SCENARIOS["QM Quick Retest"], 0))


def position(pl, plpc, symbol="AAPL261120P00335000", qty="2"):
    return SimpleNamespace(symbol=symbol, unrealized_pl=str(pl), unrealized_plpc=str(plpc), qty=qty,
                           qty_available="0", current_price="1.25")


class QuickTakeProfit(unittest.TestCase):
    s = SimpleNamespace(quick_profit_dollars=10.0, quick_profit_pct=0.03, take_profit_pct=0.50, stop_loss_pct=0.35)

    def test_more_than_either_target_takes_profit(self):
        for pl, plpc in [(10.01, 0.01), (1.00, 0.031), (50.00, 0.20)]:
            with self.subTest(pl=pl, plpc=plpc):
                self.assertIn("take profit", bot.quick_take_profit(position(pl, plpc), self.s))

    def test_at_or_below_both_targets_holds(self):
        for pl, plpc in [(10.00, 0.03), (0, 0), (-5, -0.02)]:
            with self.subTest(pl=pl, plpc=plpc):
                self.assertIsNone(bot.quick_take_profit(position(pl, plpc), self.s))

    def run_exits(self, strategy, p):
        sold = []
        broker = SimpleNamespace(option_positions=lambda: [p], sell=lambda _, why: sold.append(why))
        bot.manage_exits(broker, self.s, strategy)
        return sold

    def test_price_action_sells_at_the_quick_target(self):
        sold = self.run_exits(bot.PRICE_ACTION, position(12.00, 0.02))
        self.assertEqual(sold, ["take profit $+12.00 (+2.0%)"])

    def test_sma_strategy_still_waits_for_fifty_percent(self):
        s = SimpleNamespace(**vars(self.s), exit_dte=7)
        sold = []
        broker = SimpleNamespace(option_positions=lambda: [position(12.00, 0.02)], sell=lambda _, why: sold.append(why))
        bot.manage_exits(broker, s, bot.SMA)
        self.assertEqual(sold, [])


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
        s = QuickTakeProfit.s
        with patch.object(pd.Timestamp, "now", return_value=eastern(15, 20)):
            bot.manage_exits(broker, s, bot.PRICE_ACTION)
        self.assertEqual((closed, sold), (["expiry-day exit"], []))


if __name__ == "__main__":
    unittest.main()
