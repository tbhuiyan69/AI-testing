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

    @property
    def mid(self):
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self):
        return (self.ask - self.bid) / self.mid if self.mid > 0 else 1.0


def _list_contracts(trading_client, underlying, kind, price, settings):
    today = date.today()
    req = GetOptionContractsRequest(
        underlying_symbols=[underlying],
        status=AssetStatus.ACTIVE,
        type=ContractType.CALL if kind == "call" else ContractType.PUT,
        expiration_date_gte=today + timedelta(days=settings.min_dte),
        expiration_date_lte=today + timedelta(days=settings.max_dte),
        strike_price_gte=str(round(price * 0.90, 2)),
        strike_price_lte=str(round(price * 1.10, 2)),
        limit=500,
    )
    contracts = trading_client.get_option_contracts(req).option_contracts or []
    return [
        c for c in contracts
        if c.tradable and int(c.open_interest or 0) >= settings.min_open_interest
    ]


def pick_contract(trading_client, option_client, underlying, kind, price, settings):
    contracts = _list_contracts(trading_client, underlying, kind, price, settings)
    if not contracts:
        return None, "no contracts matched expiry/strike/open-interest filters"

    by_symbol = {c.symbol: c for c in contracts}
    symbols = list(by_symbol)
    snapshots = {}
    for i in range(0, len(symbols), 100):
        req = OptionSnapshotRequest(symbol_or_symbols=symbols[i:i + 100], feed=OptionsFeed.INDICATIVE)
        snapshots.update(option_client.get_option_snapshot(req))

    candidates = []
    for sym, snap in snapshots.items():
        q = snap.latest_quote
        if not q or not q.bid_price or not q.ask_price:
            continue
        c = by_symbol[sym]
        delta = snap.greeks.delta if snap.greeks else None
        cand = Candidate(sym, float(c.strike_price), c.expiration_date, float(q.bid_price), float(q.ask_price), delta)
        if cand.spread_pct <= settings.max_spread_pct:
            candidates.append(cand)

    if not candidates:
        return None, "no contracts with a tight enough bid/ask spread"

    def score(c):
        # Prefer closest to target delta; fall back to strike distance if greeks are missing.
        if c.delta is not None:
            return abs(abs(c.delta) - settings.target_delta)
        return abs(c.strike - price) / price + 1.0

    return min(candidates, key=score), None
