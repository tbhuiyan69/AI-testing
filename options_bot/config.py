"""Bot settings. Every value can be overridden with an environment variable or a .env file."""
import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# Stock tickers from the Robinhood "Claude_scanner" watchlist. Its SPX and MGTN indexes are left out
# because they have no stock bars or stock options to trade here.
SCANNER_SYMBOLS = [
    "NBIS", "APLD", "CRWV", "QMCO", "KO", "AMZN", "META", "AMD", "QCOM",
    "NVDA", "TSLA", "HIMS", "SOFI", "AAPL", "IREN", "XOM", "QQQ", "SPY",
]


def _env(name, default, cast=str):
    raw = os.getenv(name)
    return default if raw is None or raw == "" else cast(raw)


def _bool(raw):
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _list(raw):
    return [s.strip().upper() for s in str(raw).split(",") if s.strip()]


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: _env("ALPACA_API_KEY", ""))
    secret_key: str = field(default_factory=lambda: _env("ALPACA_SECRET_KEY", ""))
    paper: bool = field(default_factory=lambda: _env("ALPACA_PAPER_TRADE", True, _bool))

    # Universe and signal
    watchlist: list = field(default_factory=lambda: _env("BOT_WATCHLIST", SCANNER_SYMBOLS, _list))
    fast_sma: int = field(default_factory=lambda: _env("BOT_FAST_SMA", 10, int))
    slow_sma: int = field(default_factory=lambda: _env("BOT_SLOW_SMA", 30, int))

    # Contract selection
    min_dte: int = field(default_factory=lambda: _env("BOT_MIN_DTE", 21, int))
    max_dte: int = field(default_factory=lambda: _env("BOT_MAX_DTE", 45, int))
    target_delta: float = field(default_factory=lambda: _env("BOT_TARGET_DELTA", 0.40, float))
    max_spread_pct: float = field(default_factory=lambda: _env("BOT_MAX_SPREAD_PCT", 0.10, float))
    min_open_interest: int = field(default_factory=lambda: _env("BOT_MIN_OPEN_INTEREST", 100, int))

    # Risk
    max_trade_dollars: float = field(default_factory=lambda: _env("BOT_MAX_TRADE_DOLLARS", 1000.0, float))
    max_open_positions: int = field(default_factory=lambda: _env("BOT_MAX_OPEN_POSITIONS", 3, int))
    daily_loss_limit_pct: float = field(default_factory=lambda: _env("BOT_DAILY_LOSS_LIMIT_PCT", 0.03, float))

    # Exits
    take_profit_pct: float = field(default_factory=lambda: _env("BOT_TAKE_PROFIT_PCT", 0.50, float))
    stop_loss_pct: float = field(default_factory=lambda: _env("BOT_STOP_LOSS_PCT", 0.35, float))
    exit_dte: int = field(default_factory=lambda: _env("BOT_EXIT_DTE", 7, int))
    auto_approve: bool = False  # set by run_bot.py --auto; applies to contracts costing under auto_approve_below
    auto_approve_below: float = field(default_factory=lambda: _env("BOT_AUTO_APPROVE_BELOW", 100.0, float))
    approval_seconds: int = field(default_factory=lambda: _env("BOT_APPROVAL_SECONDS", 60, int))
    profit_lock_dollars: float = field(default_factory=lambda: _env("BOT_PROFIT_LOCK_DOLLARS", 5.0, float))

    # Loop
    poll_minutes: int = field(default_factory=lambda: _env("BOT_POLL_MINUTES", 15, int))

    def validate(self):
        if not self.api_key or not self.secret_key:
            raise SystemExit("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set (env vars or .env).")
        if not self.paper:
            raise SystemExit("This bot only runs against a paper account. Set ALPACA_PAPER_TRADE=true.")
        if self.fast_sma >= self.slow_sma:
            raise SystemExit("BOT_FAST_SMA must be smaller than BOT_SLOW_SMA.")
