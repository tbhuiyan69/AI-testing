"""Run with: python -m unittest discover tests"""
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from options_bot import bot, price_action as pa
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
            return pa.price_action_signal(FakeBroker(live), "AAA", SimpleNamespace(min_dte=21, max_dte=45))

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
        signal = lambda *_: ("put", 250.0, "QM Quick Retest put, stop 252.20, P/C 2.00")
        with patch.object(bot, "pick_contract", return_value=(cand, None)):
            bot.look_for_entries(broker, s, signal)
        self.assertEqual(broker.bought, [(cand, "QM Quick Retest put, stop 252.20, P/C 2.00, delta -0.40")])


if __name__ == "__main__":
    unittest.main()
