"""Pick a liquid option contract near the target delta."""
from dataclasses import dataclass
from datetime import date, time, timedelta

import pandas as pd

from alpaca.data.enums import OptionsFeed
from alpaca.data.requests import OptionSnapshotRequest
from alpaca.trading.enums import AssetStatus, ContractType
from alpaca.trading.requests import GetOptionContractsRequest

from .broker import tick_round

# Most you may pay per share, by days to expiry: same day under $0.80, up to 3 days $1.20, up to 5 days $1.50.
PREMIUM_CAPS = ((0, 0.79), (3, 1.20), (5, 1.50))
MAX_CAPPED_DTE = PREMIUM_CAPS[-1][0]
MIN_DELTA = 0.15  # cheap contracts below this are unlikely to pay off; unknown delta doesn't pass
# Alpaca stops accepting new orders on expiring contracts at 3:30 PM ET, so stop well before that.
LAST_SAME_DAY_ENTRY = time(15, 0)


@dataclass
class Candidate:
    symbol: str
    strike: float
    expiration: date
    bid: float
    ask: float
    delta: float | None
    open_interest: int = 0
    iv: float | None = None

    @property
    def mid(self):
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self):
        return (self.ask - self.bid) / self.mid if self.mid > 0 else 1.0

    @property
    def dte(self):
        return (self.expiration - date.today()).days


def quote_contracts(trading_client, option_client, underlying, kind, price,
                    min_dte, max_dte, strike_band=0.10, min_open_interest=0, max_spread_pct=1.0):
    """All tradable contracts in the expiry/strike window that have a two-sided quote."""
    today = date.today()
    req = GetOptionContractsRequest(
        underlying_symbols=[underlying],
        status=AssetStatus.ACTIVE,
        type=ContractType.CALL if kind == "call" else ContractType.PUT,
        expiration_date_gte=today + timedelta(days=min_dte),
        expiration_date_lte=today + timedelta(days=max_dte),
        strike_price_gte=str(round(price * (1 - strike_band), 2)),
        strike_price_lte=str(round(price * (1 + strike_band), 2)),
        limit=1000,
    )
    contracts = [
        c for c in (trading_client.get_option_contracts(req).option_contracts or [])
        if c.tradable and int(c.open_interest or 0) >= min_open_interest
    ]
    if not contracts:
        return []

    by_symbol = {c.symbol: c for c in contracts}
    symbols = list(by_symbol)
    snapshots = {}
    for i in range(0, len(symbols), 100):
        req = OptionSnapshotRequest(symbol_or_symbols=symbols[i:i + 100], feed=OptionsFeed.INDICATIVE)
        snapshots.update(option_client.get_option_snapshot(req))

    out = []
    for sym, snap in snapshots.items():
        q = snap.latest_quote
        if not q or not q.bid_price or not q.ask_price:
            continue
        c = by_symbol[sym]
        cand = Candidate(
            sym, float(c.strike_price), c.expiration_date, float(q.bid_price), float(q.ask_price),
            snap.greeks.delta if snap.greeks else None, int(c.open_interest or 0), snap.implied_volatility,
        )
        if cand.spread_pct <= max_spread_pct:
            out.append(cand)
    return sorted(out, key=lambda c: (c.expiration, c.strike))


def pc_ratio(trading_client, underlying, price, min_dte, max_dte, strike_band=0.10):
    """Put/call ratio from open interest in the expiry/strike window, or None if there is no call OI."""
    today = date.today()
    totals = {ContractType.CALL: 0, ContractType.PUT: 0}
    token = None
    while True:
        resp = trading_client.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[underlying],
            status=AssetStatus.ACTIVE,
            expiration_date_gte=today + timedelta(days=min_dte),
            expiration_date_lte=today + timedelta(days=max_dte),
            strike_price_gte=str(round(price * (1 - strike_band), 2)),
            strike_price_lte=str(round(price * (1 + strike_band), 2)),
            limit=1000,
            page_token=token,
        ))
        for c in resp.option_contracts or []:
            totals[c.type] += int(c.open_interest or 0)
        token = resp.next_page_token
        if not token:
            break
    return totals[ContractType.PUT] / totals[ContractType.CALL] if totals[ContractType.CALL] else None


def pick_contract(trading_client, option_client, underlying, kind, price, settings):
    candidates = quote_contracts(
        trading_client, option_client, underlying, kind, price,
        settings.min_dte, settings.max_dte, 0.10, settings.min_open_interest, settings.max_spread_pct,
    )
    if not candidates:
        return None, "no liquid contracts matched expiry/strike/open-interest/spread filters"

    return _closest_to_target(candidates, price, settings), None


def _closest_to_target(candidates, price, settings):
    def score(c):
        # Prefer closest to target delta; fall back to strike distance if greeks are missing.
        if c.delta is not None:
            return abs(abs(c.delta) - settings.target_delta)
        return abs(c.strike - price) / price + 1.0

    return min(candidates, key=score)


def premium_cap(dte):
    """Highest allowed price per share for this many days to expiry, or None if it's too far out."""
    for max_dte, cap in PREMIUM_CAPS:
        if 0 <= dte <= max_dte:
            return cap
    return None


def within_premium_cap(c):
    cap = premium_cap(c.dte)
    return cap is not None and tick_round(c.mid) <= cap  # the mid is the limit price the order will use


def meets_min_delta(c):
    return c.delta is not None and abs(c.delta) >= MIN_DELTA


def pick_capped_contract(trading_client, option_client, underlying, kind, price, settings):
    """Like pick_contract, but only 0-5 day contracts within PREMIUM_CAPS and at or above MIN_DELTA."""
    same_day_open = pd.Timestamp.now("America/New_York").time() < LAST_SAME_DAY_ENTRY
    candidates = quote_contracts(
        trading_client, option_client, underlying, kind, price,
        0 if same_day_open else 1, MAX_CAPPED_DTE, 0.10, settings.min_open_interest, settings.max_spread_pct,
    )
    if not candidates:
        return None, "no liquid 0-5 day contracts matched"
    allowed = [c for c in candidates if within_premium_cap(c) and meets_min_delta(c)]
    if not allowed:
        return None, f"none of {len(candidates)} liquid contracts are within the price limits and minimum delta"
    return _closest_to_target(allowed, price, settings), None
