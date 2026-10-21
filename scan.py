"""Scan stocks with the EMA 9/21 + RSI + MACD 15-minute setup (ema_rsi_macd_setup_spec.md).

    python scan.py                          # scan the default universe
    python scan.py --symbols QS,IREN,AAPL   # scan your own list
    python scan.py --puts                   # also list put contracts for bearish setups
    python scan.py --puts --min-dte 7 --max-dte 30

Read-only: this never places orders.
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

from options_bot.broker import Broker
from options_bot.config import Settings
from options_bot.contracts import quote_contracts
from options_bot.ema_rsi_macd import BEARISH_STATES, BULLISH_STATES, evaluate, fetch_15m

DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "AMD", "AVGO", "NFLX", "CRM", "ORCL", "ADBE", "INTC", "MU", "QCOM", "SMCI",
    "PLTR", "COIN", "MSTR", "HOOD", "SOFI", "UBER", "SHOP", "SNOW", "CRWD", "PANW",
    "JPM", "BAC", "GS", "XOM", "CVX", "BA", "DIS", "NKE", "WMT", "COST",
    "QS", "IREN", "RIVN", "MARA", "RIOT", "CRWV", "ARM", "TSM", "BABA", "SOXL",
]

ORDER = ["RECLAIM", "BREAKOUT", "PULLBACK", "TRENDING", "BASE", "NEUTRAL",
         "REJECT", "BREAKDOWN", "RALLY", "BROKEN", "BEARISH"]


def fmt_signals(sigs):
    names = {"entry_a": "Long A", "entry_b": "Long B", "entry_a_short": "Short A", "entry_b_short": "Short B"}
    return ",".join(names[s] for s in sigs) or "-"


def print_table(rows):
    print(f"\n{'Symbol':<7}{'State':<11}{'Close':>9}{'EMA9':>9}{'EMA21':>9}{'RSI':>6}"
          f"{'MACD':>9}{'Vol x':>7}  {'Signals (last 4 bars)':<22}")
    print("-" * 92)
    for sym, r in rows:
        print(f"{sym:<7}{r['state']:<11}{r['close']:>9.2f}{r['ema9']:>9.2f}{r['ema21']:>9.2f}"
              f"{r['rsi']:>6.1f}{r['macd']:>9.3f}{r['vol_x']:>7.1f}  {fmt_signals(r['recent_signals']):<22}")


def print_puts(broker, sym, r, a):
    puts = quote_contracts(broker.trading, broker.options, sym, "put", r["close"],
                           a.min_dte, a.max_dte, a.strike_band, a.min_oi, a.max_spread)
    puts = [p for p in puts if p.delta is None or a.min_delta <= abs(p.delta) <= a.max_delta]
    print(f"\n{sym} puts  (state {r['state']}, close {r['close']:.2f}, setup stop above {r['short_stop']:.2f})")
    if not puts:
        print("  no puts matched the filters")
        return []
    print(f"  {'Contract':<22}{'Exp':>11}{'DTE':>5}{'Strike':>9}{'Bid':>8}{'Ask':>8}"
          f"{'Mid':>8}{'Sprd%':>7}{'Delta':>7}{'IV':>7}{'OI':>8}")
    for p in puts:
        delta = f"{p.delta:+.2f}" if p.delta is not None else "  n/a"
        iv = f"{p.iv:.0%}" if p.iv else "n/a"
        print(f"  {p.symbol:<22}{p.expiration.isoformat():>11}{p.dte:>5}{p.strike:>9.2f}{p.bid:>8.2f}"
              f"{p.ask:>8.2f}{p.mid:>8.2f}{p.spread_pct:>7.1%}{delta:>7}{iv:>7}{p.open_interest:>8}")
    return puts


def main():
    ap = argparse.ArgumentParser(description="EMA/RSI/MACD 15m setup scanner")
    ap.add_argument("--symbols", help="comma-separated tickers (default: built-in universe)")
    ap.add_argument("--all", action="store_true", help="show every symbol, not just ones in a setup state")
    ap.add_argument("--puts", action="store_true", help="list put contracts for bearish-state symbols")
    ap.add_argument("--puts-for", help="list puts for these tickers regardless of state")
    ap.add_argument("--min-dte", type=int, default=7)
    ap.add_argument("--max-dte", type=int, default=45)
    ap.add_argument("--strike-band", type=float, default=0.07, help="strikes within ±this fraction of price")
    ap.add_argument("--min-delta", type=float, default=0.20)
    ap.add_argument("--max-delta", type=float, default=0.60)
    ap.add_argument("--min-oi", type=int, default=100)
    ap.add_argument("--max-spread", type=float, default=0.15, help="max (ask-bid)/mid")
    a = ap.parse_args()

    s = Settings()
    s.validate()
    broker = Broker(s, live_orders=False)
    universe = [x.strip().upper() for x in a.symbols.split(",")] if a.symbols else DEFAULT_UNIVERSE

    bars = fetch_15m(broker.stocks, universe)
    results = []
    for sym in universe:
        r = evaluate(bars[sym]) if sym in bars else None
        if r:
            results.append((sym, r))
    missing = [x for x in universe if x not in {s for s, _ in results}]

    results.sort(key=lambda x: (ORDER.index(x[1]["state"]), x[0]))
    shown = results if a.all else [x for x in results if x[1]["state"] not in ("NEUTRAL", "BASE")]
    last_bar = max((r["time"] for _, r in results), default=None)
    print(f"EMA9/21 + RSI14 + MACD(12,26,9) on 15m bars (IEX feed). Latest bar: {last_bar}")
    print_table(shown)

    bull = [x for x, r in results if r["state"] in BULLISH_STATES]
    bear = [x for x, r in results if r["state"] in BEARISH_STATES]
    print(f"\nBullish ({len(bull)}): {', '.join(bull) or '-'}")
    print(f"Bearish ({len(bear)}): {', '.join(bear) or '-'}")
    if missing:
        print(f"No/insufficient data: {', '.join(missing)}")

    Path("logs").mkdir(exist_ok=True)
    out = Path("logs") / f"scan_{datetime.now():%Y%m%d_%H%M}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "state", "close", "ema9", "ema21", "rsi", "macd", "hist", "vol_x", "signals", "bar_time"])
        for sym, r in results:
            w.writerow([sym, r["state"], round(r["close"], 2), round(r["ema9"], 2), round(r["ema21"], 2),
                        round(r["rsi"], 1), round(r["macd"], 4), round(r["hist"], 4), round(r["vol_x"], 2),
                        fmt_signals(r["recent_signals"]), r["time"]])
    print(f"Saved {out}")

    put_syms = [x.strip().upper() for x in a.puts_for.split(",")] if a.puts_for else (bear if a.puts else [])
    by_sym = dict(results)
    put_rows = []
    for sym in put_syms:
        if sym in by_sym:
            r = by_sym[sym]
            for p in print_puts(broker, sym, r, a):
                put_rows.append([sym, r["state"], fmt_signals(r["recent_signals"]), round(r["close"], 2),
                                 round(r["short_stop"], 2), p.symbol, p.expiration, p.dte, p.strike, p.bid, p.ask,
                                 round(p.mid, 2), round(p.spread_pct, 3),
                                 None if p.delta is None else round(p.delta, 3),
                                 None if p.iv is None else round(p.iv, 3), p.open_interest])
    if put_rows:
        pout = Path("logs") / f"puts_{datetime.now():%Y%m%d_%H%M}.csv"
        with pout.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["underlying", "state", "signals", "close", "setup_stop", "contract", "expiration", "dte",
                        "strike", "bid", "ask", "mid", "spread_pct", "delta", "iv", "open_interest"])
            w.writerows(put_rows)
        print(f"\nSaved {len(put_rows)} puts to {pout}")


if __name__ == "__main__":
    main()
