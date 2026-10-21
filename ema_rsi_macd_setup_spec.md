# Setup Spec: EMA 9/21 Trend + RSI/MACD Momentum (15-min)

Taken from the QS chart in Legend's "Chart Layout" on Oct 20, 2026 (15-minute bars, overnight session on).
This is a written description of a setup so it can be coded. It's not a trade recommendation.

---

## 1. Indicators (exactly as on the chart)

| Pane | Indicator | Parameters |
|---|---|---|
| Price | EMA fast | length 9, source close |
| Price | EMA slow | length 21, source close |
| Price | Volume + Volume MA | MA length 20 |
| Lower 1 | RSI | length 14, Wilder's smoothing, source close; shaded band 40–60 (overbought/oversold lines 70/30) |
| Lower 2 | MACD | fast 12, slow 26, signal 9 (EMA); histogram = MACD − signal |

**Timeframe:** 15m for signals. Optional higher-timeframe filter: daily (see §6).

---

## 2. What the chart shows (the pattern this spec is based on)

1. **Downtrend / base (01:00 → ~09:30):** price under both EMAs, EMA9 < EMA21, RSI stays inside 40–60, MACD slightly below zero. Price bottoms near $3.84.
2. **Trigger bar (~10:00):** a large green candle on high volume (about 3× or more the 20-bar volume MA) closes **above both EMAs**. RSI goes from about 50 to above 60 (peaking near 70). MACD crosses **above signal and above 0**, and the histogram expands.
3. **EMA cross / trend confirm (~10:30–11:00):** EMA9 crosses above EMA21. Price runs to about $4.07 and then consolidates.
4. **Pullback test (~16:00):** a red candle on a volume spike pulls price back to **EMA21 (~$3.94)**. It holds: the candle wicks but doesn't close below EMA21. RSI dips to the mid-40s/50s without breaking 40. MACD histogram turns red, but the MACD line stays near 0.
5. **Continuation (17:00 → 23:00):** price reclaims EMA9 and grinds up to $4.05 with the EMAs stacked (price > EMA9 > EMA21). RSI holds about 55–60. The MACD histogram flips back to small green.

There's a manual price alert at **$3.99** and a reference line at **$3.93** (≈ EMA21 / pullback low).

---

## 3. Rules

### Long Entry A: Momentum Breakout (step 2)
All on the **close** of a 15m bar:
- `close > EMA9` and `close > EMA21`
- the previous bar's close was `<= EMA21` (it crossed above), **or** the close is the highest of the last N bars (N=20)
- `RSI(14)` crosses above **60**
- `MACD > signal` and `histogram > histogram[1]` (expanding); a MACD > 0 bonus
- `volume >= 2.0 × VolMA20`

### Long Entry B: Pullback to EMA21 (step 4), the higher-quality entry
Requires the trend to already be established:
- `EMA9 > EMA21` for at least 4 bars
- the bar's `low <= EMA21 × 1.002` (touches or tags EMA21) **and** `close >= EMA21`
- `RSI(14)` stays `>= 40` during the pullback
- `MACD line >= -tolerance` (it hasn't collapsed: near or above 0)
- **Trigger:** the next bar closes back above EMA9 (reclaim)

### Stop
- Entry A: below the trigger bar's low, or below EMA21, whichever is lower
- Entry B: below the pullback bar's low (in the example, about $3.92–3.93)
- Optional: `entry − 1.5 × ATR(14)`

### Exits (first one hit)
- **Target 1:** the prior swing high (example: $4.07), or 1.5R → take partial
- **Trail:** after T1, exit on a 15m close below EMA9 (aggressive) or EMA21 (conservative)
- **Momentum fail:** `RSI < 40` **or** a bearish MACD cross below 0 → exit the rest
- **Time stop (optional):** no new high within X bars

### No-trade filters
- EMA9 and EMA21 tangled: `abs(EMA9 − EMA21) / close < 0.1%` for most of the last 10 bars (chop; this was the case 01:00–09:00)
- Volume on the trigger bar `< VolMA20`
- Overnight/extended session with a very wide spread (e.g., `(ask − bid)/mid > 0.5%`). Note that in overnight trading QS showed a bid of $4.03 and an ask of $4.05 (≈ 0.5%).

---

## 4. Signal states (useful for coding and for a dashboard)

| State | Condition |
|---|---|
| `BEARISH` | close < EMA21 and EMA9 < EMA21 |
| `BASE` | EMAs within 0.1%, RSI in 40–60, \|MACD\| ≈ 0 |
| `BREAKOUT` | Entry A conditions true |
| `TRENDING` | close > EMA9 > EMA21, RSI > 50, MACD > 0 |
| `PULLBACK` | EMA9 > EMA21, low touches EMA21, RSI ≥ 40 |
| `RECLAIM` | PULLBACK on the previous bar, and close > EMA9 now (Entry B fires) |
| `BROKEN` | close < EMA21 and RSI < 40 |

---

## 5. Reference pseudocode (Python / pandas)

```python
import pandas as pd

def ema(s, n):  return s.ewm(span=n, adjust=False).mean()

def rsi_wilder(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

def add_indicators(df):
    df["ema9"]  = ema(df.close, 9)
    df["ema21"] = ema(df.close, 21)
    df["volma"] = df.volume.rolling(20).mean()
    df["rsi"]   = rsi_wilder(df.close, 14)
    macd        = ema(df.close, 12) - ema(df.close, 26)
    df["macd"], df["signal"] = macd, ema(macd, 9)
    df["hist"]  = df.macd - df.signal
    return df

def signals(df, vol_mult=2.0):
    p = df.shift(1)
    trend_bars = (df.ema9 > df.ema21).rolling(4).sum() == 4

    df["entry_a"] = (
        (df.close > df.ema9) & (df.close > df.ema21) &
        ((p.close <= p.ema21) | (df.close >= df.close.rolling(20).max())) &
        (p.rsi < 60) & (df.rsi >= 60) &
        (df.macd > df.signal) & (df.hist > p.hist) &
        (df.volume >= vol_mult * df.volma)
    )

    pullback = trend_bars & (df.low <= df.ema21 * 1.002) & (df.close >= df.ema21) & (df.rsi >= 40)
    df["pullback"] = pullback
    df["entry_b"] = pullback.shift(1, fill_value=False) & (df.close > df.ema9) & (df.macd > -0.001 * df.close)  # MACD near/above 0

    df["exit"] = (df.rsi < 40) | ((df.macd < df.signal) & (df.macd < 0)) | (df.close < df.ema21)
    return df
```

Data source options: Robinhood `get_equity_historicals` (15minute bars, `bounds="extended"` if you want overnight), or compute directly with `get_equity_technical_indicators` (rsi / macd / ema, interval `15minute`).

---

## 6. Optional upgrades

- **Higher-timeframe filter:** only take longs when daily close > daily EMA21, **or** (mean-reversion variant) daily RSI < 30 plus a 15m Entry A, the "oversold bounce" version. IREN fits that variant right now (daily RSI ≈ 25).
- **Session flag:** tag each bar as overnight, pre, regular or post. Require regular hours for entries, or demand a higher volume multiple at night.
- **Scanner (Legend "Claude_scanner"):** filter on RSI(14, 15m) crossing above 60, price > EMA21, and volume > 2× average to surface Entry A candidates.
- **Alerts:** RSI above 60 / below 40 on 15m, a MACD/signal cross on 15m, and a price alert at the pullback level (EMA21).

---

## 7. Parameters to tune

| Param | Default | Range to test |
|---|---|---|
| EMA fast / slow | 9 / 21 | 8–12 / 20–34 |
| RSI breakout level | 60 | 55–65 |
| RSI floor | 40 | 35–45 |
| Volume multiple | 2.0× | 1.5–3.0× |
| EMA21 touch tolerance | 0.2% | 0.1–0.5% |
| Trend-bars before pullback | 4 | 3–8 |
