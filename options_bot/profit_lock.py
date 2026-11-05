"""Remember which positions have passed the profit target, so a bot restart doesn't forget their lock."""
import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger("bot")
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "logs" / "profit_locks.json"
SAVE_ATTEMPTS = 3


class ProfitLocks:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)
        try:
            self.symbols = set(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            self.symbols = set()

    def _save(self):
        """Failing to save must never stop exit checks; the lock still holds in memory for this run."""
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        for attempt in range(SAVE_ATTEMPTS):
            try:
                self.path.parent.mkdir(exist_ok=True)
                tmp.write_text(json.dumps(sorted(self.symbols)), encoding="utf-8")
                os.replace(tmp, self.path)
                return
            except OSError as e:  # OneDrive or another copy of the bot can hold the file briefly
                if attempt == SAVE_ATTEMPTS - 1:
                    log.warning("Could not save profit locks (%s); kept in memory only", e)
                else:
                    time.sleep(0.2)

    def is_locked(self, symbol):
        return symbol in self.symbols

    def lock(self, symbol):
        self.symbols.add(symbol)
        self._save()

    def keep_only(self, open_symbols):
        """Forget closed positions, so buying the same contract later starts fresh."""
        kept = self.symbols & set(open_symbols)
        if kept != self.symbols:
            self.symbols = kept
            self._save()
