#!/usr/bin/env python3
"""
Top 10 Institutional Price Action patterns (from the QM / Quasimodo cheat sheet)
with a worked use case each.

Usage:
    python3 price_action_use_cases.py              # list the 10 patterns
    python3 price_action_use_cases.py 3            # show use case #3
    python3 price_action_use_cases.py all          # show all 10
    python3 price_action_use_cases.py export       # write use_cases.csv (and .xlsx if openpyxl is installed)

NOTES
- The 10 were chosen by judgment (clear structure, clear invalid level, common in
  retail price-action teaching). The cheat sheet itself has NO win-rate data and
  these were NOT backtested. Test on your own charts before risking money.
- All prices are made-up examples. Each is written as the bearish (PUT) version;
  the bullish (CALL) version is the mirror image.
- Confirmation rule (your setup): RSI vs its average line AND MACD vs signal line
  must agree on the 5m and 10m. P/C > 1 backs a put, P/C < 1 backs a call.
"""
import csv
import sys
from dataclasses import dataclass, field


@dataclass
class UseCase:
    name: str
    idea: str
    sequence: list
    entry: float
    stop: float
    target1: float
    target2: float
    trigger: str
    invalid: str
    mirror: str
    checks: list = field(default_factory=lambda: [
        "RSI below its average line (5m and 10m)",
        "MACD below signal line (5m and 10m)",
        "P/C ratio above 1 (flow agrees)",
    ])

    def risk(self):
        return round(self.stop - self.entry, 2)

    def r_multiple(self, target):
        return round((self.entry - target) / self.risk(), 2)


CASES = [
    UseCase(
        "QM Quick Retest",
        "Uptrend breaks structure, then price returns to the QM level right away.",
        ["H 250.00", "L 247.50", "HH 252.00", "LL 245.80 (breaks the L)",
         "QML = 250.00, supply zone 250.00-250.60", "Price retests the zone and is rejected"],
        250.40, 252.00, 245.80, 243.00,
        "First touch of the supply zone + bearish rejection candle.",
        "Close above the HH (252.00).",
        "Bullish QM: H, L, LL, HH breaks the prior high, then retest of the demand zone -> CALL.",
    ),
    UseCase(
        "QM Late Retest",
        "Same QM break, but price builds a range before returning to the zone.",
        ["H 250.00", "L 247.50", "HH 252.00", "LL 245.80",
         "Range 246.50-248.50 for several candles", "Price finally rallies into 250.00-250.60 supply"],
        250.30, 252.00, 245.80, 243.50,
        "Touch of the supply zone after the range, with rejection.",
        "Close above the HH (252.00), or a clean close above the zone.",
        "Mirror: range above the demand zone, then a late drop into it -> CALL.",
    ),
    UseCase(
        "QM Re-Entry",
        "A small QM forms inside the main QM zone, giving a tighter entry.",
        ["Main QML 250.00, zone 250.00-251.00", "Price enters the zone",
         "Inside the zone: small H 250.80, L 250.30, HH 251.10, LL 250.10", "Small QML 250.60"],
        250.55, 251.30, 247.00, 245.00,
        "Break of the small structure (LL 250.10) from the small QML.",
        "Close above the main zone top (251.00-251.30).",
        "Mirror inside a demand zone -> CALL.",
    ),
    UseCase(
        "Continuation QM",
        "A QM that forms inside an existing downtrend, so you trade with the trend.",
        ["Downtrend in place (lower highs, lower lows)", "Pullback: H 250.00",
         "Higher high 251.50 fails", "LL 246.00 breaks the prior low", "QML 250.00"],
        250.30, 251.80, 246.00, 243.00,
        "Retest of the QML zone in the direction of the trend.",
        "Close above 251.80 (trend structure broken).",
        "Mirror in an uptrend -> CALL.",
    ),
    UseCase(
        "Fakeout V1 (Default)",
        "Price sweeps above resistance, traps buyers, then closes back inside and breaks lower.",
        ["Resistance 250.00 (R1, R2 touches)", "Spike to 250.90 above resistance",
         "Close back below 250.00", "Break below the recent low 247.80", "Supply zone 250.00-250.90"],
        250.20, 251.00, 246.50, 244.00,
        "Retest of the zone after the close back inside.",
        "Close above the fakeout high (251.00).",
        "Mirror: sweep below support, close back above, then retest -> CALL.",
    ),
    UseCase(
        "Fakeout V2 (SR Flip)",
        "Broken support flips to resistance, price retests it, and fails.",
        ["Support 248.00 holds twice", "Support breaks, close 246.80",
         "Price rallies back to 248.00 (now resistance)", "Rejected at 248.00-248.40"],
        248.20, 249.00, 245.00, 243.00,
        "Rejection candle at the flipped level.",
        "Close above 249.00 (flip failed).",
        "Mirror: broken resistance becomes support -> CALL.",
    ),
    UseCase(
        "Flag B",
        "Small bounce (flag) after a drop; the flag limit acts as supply.",
        ["Impulse down 252.00 -> 247.00", "Small bounce 247.00 -> 249.00 forms the flag",
         "Flag limit / supply 249.00-249.40", "Price rejects the limit"],
        249.10, 249.80, 245.50, 243.50,
        "Rejection at the flag limit, or break of the flag low (247.60).",
        "Close above 249.80.",
        "Mirror: bull flag on a rally -> CALL.",
    ),
    UseCase(
        "MPL (Stop Hunt)",
        "Price hunts stops above repeated highs, then engulfs the prior low.",
        ["Resistance 250.00 touched 3 times (R R R)", "Spike to 250.70 takes the stops",
         "Bearish engulfing candle closes below the previous low 249.10", "Supply zone 250.00-250.70"],
        249.80, 250.90, 246.50, 244.50,
        "Engulfing candle after the stop hunt.",
        "Close above the stop-hunt high (250.90).",
        "Mirror: hunt below repeated lows, bullish engulf -> CALL.",
    ),
    UseCase(
        "Compression (CP)",
        "Price climbs in tightening steps into a zone, then clears the demand under it.",
        ["Higher lows pushing into 250.00 resistance", "Each push is smaller (compression)",
         "Break of the last higher low (248.80) clears demand", "Supply zone at 250.00-250.50"],
        249.20, 250.60, 245.50, 243.00,
        "Break of the last higher low after the compression.",
        "Close above 250.60.",
        "Mirror: compression into support, then a break upward -> CALL.",
    ),
    UseCase(
        "3 Drive",
        "Three pushes up, each with a smaller pullback, then exhaustion and reversal.",
        ["Drive 1 high 249.00", "Drive 2 high 250.00", "Drive 3 high 251.00",
         "Reversal candle at drive 3", "Break of drive 3 pullback low 249.80"],
        249.70, 251.20, 246.00, 243.50,
        "Break of the pullback low after the third drive.",
        "Close above the drive-3 high (251.20).",
        "Mirror: three drives down, then a reversal up -> CALL.",
    ),
]


def show(n):
    c = CASES[n - 1]
    print(f"\n#{n}  {c.name}  (PUT example)")
    print("-" * 60)
    print("Idea:", c.idea)
    print("Sequence:")
    for i, s in enumerate(c.sequence, 1):
        print(f"  {i}. {s}")
    print(f"Trigger : {c.trigger}")
    print(f"Entry   : {c.entry:.2f}   Stop: {c.stop:.2f}   Risk/share: {c.risk():.2f}")
    print(f"Target 1: {c.target1:.2f}  ({c.r_multiple(c.target1)}R)")
    print(f"Target 2: {c.target2:.2f}  ({c.r_multiple(c.target2)}R)")
    print(f"Invalid : {c.invalid}")
    print("Checks  :")
    for k in c.checks:
        print("  [ ]", k)
    print("Mirror  :", c.mirror)


def export():
    rows = [["#", "Pattern", "Idea", "Sequence", "Trigger", "Entry", "Stop", "Risk",
             "Target1", "R1", "Target2", "R2", "Invalid", "Mirror (CALL)"]]
    for i, c in enumerate(CASES, 1):
        rows.append([i, c.name, c.idea, " | ".join(c.sequence), c.trigger, c.entry, c.stop,
                     c.risk(), c.target1, c.r_multiple(c.target1), c.target2,
                     c.r_multiple(c.target2), c.invalid, c.mirror])
    with open("use_cases.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print("Wrote use_cases.csv")
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Use Cases"
        for r in rows:
            ws.append(r)
        wb.save("use_cases.xlsx")
        print("Wrote use_cases.xlsx")
    except ImportError:
        print("openpyxl not installed; skipped .xlsx (pip install openpyxl)")


def main():
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "list"
    if arg == "list":
        for i, c in enumerate(CASES, 1):
            print(f"{i:>2}. {c.name:<24} {c.idea}")
        print("\nRun: python3 price_action_use_cases.py <number|all|export>")
    elif arg == "all":
        for i in range(1, len(CASES) + 1):
            show(i)
    elif arg == "export":
        export()
    elif arg.isdigit() and 1 <= int(arg) <= len(CASES):
        show(int(arg))
    else:
        print("Unknown option. Use: list | 1-10 | all | export")


if __name__ == "__main__":
    main()
