"""Thin wrapper around Alpaca: account checks, positions, order placement, journaling."""
import csv
import logging
import math
import re
import time
from datetime import date, datetime
from pathlib import Path

from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest

log = logging.getLogger("bot")
JOURNAL = Path(__file__).resolve().parent.parent / "logs" / "trades.csv"
OCC_RE = re.compile(r"^(?P<root>[A-Z.]+)(?P<yymmdd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


def parse_occ(symbol):
    m = OCC_RE.match(symbol)
    if not m:
        return None, None
    return m["root"], datetime.strptime(m["yymmdd"], "%y%m%d").date()


def tick_round(price):
    tick = 0.05 if price >= 3 else 0.01
    return round(math.floor(price / tick + 0.5) * tick, 2)


class Broker:
    def __init__(self, settings, live_orders):
        self.s = settings
        self.live_orders = live_orders
        self.trading = TradingClient(settings.api_key, settings.secret_key, paper=True)
        self.stocks = StockHistoricalDataClient(settings.api_key, settings.secret_key)
        self.options = OptionHistoricalDataClient(settings.api_key, settings.secret_key)

    # ---- state -------------------------------------------------------------
    def market_open(self):
        return self.trading.get_clock().is_open

    def account(self):
        return self.trading.get_account()

    def daily_loss_hit(self):
        a = self.account()
        equity, last = float(a.equity), float(a.last_equity)
        return last > 0 and (equity - last) / last <= -self.s.daily_loss_limit_pct

    def option_positions(self):
        return [p for p in self.trading.get_all_positions() if p.asset_class == AssetClass.US_OPTION]

    def busy_underlyings(self):
        """Underlyings that already have an option position or an open order."""
        busy = {parse_occ(p.symbol)[0] for p in self.option_positions()}
        for o in self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN)):
            root, _ = parse_occ(o.symbol)
            busy.add(root or o.symbol)
        busy.discard(None)
        return busy

    # ---- actions -----------------------------------------------------------
    def order_size(self, cand):
        """Contracts and limit price a buy would use; qty is 0 if one contract is over the trade limit."""
        return int(self.s.max_trade_dollars // (cand.ask * 100)), tick_round(cand.mid)

    def buy(self, cand, reason):
        qty, limit = self.order_size(cand)
        if qty < 1:
            log.info("SKIP %s: one contract (~$%.0f) exceeds max trade size $%.0f",
                     cand.symbol, cand.ask * 100, self.s.max_trade_dollars)
            return
        self._submit(cand.symbol, qty, OrderSide.BUY, limit, reason)

    def _cancel_open_orders(self, symbol):
        """An unfilled order holds the contracts, which would block any new exit on them."""
        if not self.live_orders:
            return
        orders = self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
        for order in orders:
            self.trading.cancel_order_by_id(order.id)
        if orders:
            time.sleep(1)  # cancels are asynchronous; give the contracts time to free up

    def close_now(self, position, reason):
        """Close at market."""
        self._cancel_open_orders(position.symbol)
        self._submit(position.symbol, int(float(position.qty)), OrderSide.SELL, None, reason)

    def sell(self, position, reason):
        """Sell at the current price, replacing any earlier unfilled order."""
        self._cancel_open_orders(position.symbol)
        qty = int(float(position.qty))
        if qty < 1:
            return
        price = float(position.current_price or 0)
        limit = tick_round(price) if price > 0 else None
        self._submit(position.symbol, qty, OrderSide.SELL, limit, reason)

    def _submit(self, symbol, qty, side, limit, reason):
        tag = "ORDER" if self.live_orders else "DRY-RUN"
        log.info("%s %s %d x %s @ %s  (%s)", tag, side.value.upper(), qty, symbol, limit or "MKT", reason)
        status = "dry-run"
        if self.live_orders:
            if limit is None:
                self.trading.close_position(symbol)
                status = "submitted"
            else:
                order = self.trading.submit_order(LimitOrderRequest(
                    symbol=symbol, qty=qty, side=side, limit_price=limit, time_in_force=TimeInForce.DAY,
                ))
                status = str(order.status.value if hasattr(order.status, "value") else order.status)
        self._journal(symbol, side.value, qty, limit, reason, status)

    def _journal(self, symbol, side, qty, limit, reason, status):
        JOURNAL.parent.mkdir(exist_ok=True)
        new = not JOURNAL.exists()
        with JOURNAL.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "symbol", "side", "qty", "limit", "reason", "status"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), symbol, side, qty, limit, reason, status])


def days_to_expiry(symbol):
    _, exp = parse_occ(symbol)
    return (exp - date.today()).days if exp else None
