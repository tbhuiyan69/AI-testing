"""Pick a liquid option contract near the target delta."""
from dataclasses import dataclass
from datetime import date, timedelta

from alpaca.data.enums import OptionsFeed
from alpaca.data.requests import OptionSnapshotRequest
from alpaca.trading.enums import AssetStatus, ContractType
from alpaca.trading.requests import GetOptionContractsRequest


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

    def score(c):
        # Prefer closest to target delta; fall back to strike distance if greeks are missing.
        if c.delta is not None:
            return abs(abs(c.delta) - settings.target_delta)
        return abs(c.strike - price) / price + 1.0

    return min(candidates, key=score), None
