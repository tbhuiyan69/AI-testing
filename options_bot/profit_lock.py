"""Remember which positions have passed the profit target, so a bot restart doesn't forget their lock."""
import json
import os
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "logs" / "profit_locks.json"


class ProfitLocks:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)
        try:
            self.symbols = set(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            self.symbols = set()

    def _save(self):
        self.path.parent.mkdir(exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(self.symbols)), encoding="utf-8")
        os.replace(tmp, self.path)

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
