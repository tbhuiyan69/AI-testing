"""Allow only one order-placing bot at a time, so two copies can't double-buy or double-sell.

The operating system releases the lock when the process ends, even after a crash or a closed
terminal, so there's no stale lock to clean up. It lives in the temp folder rather than the
project, so OneDrive can't interfere and every copy of the code on this PC shares it.
"""
import os
import tempfile
from pathlib import Path

LOCK_PATH = Path(tempfile.gettempdir()) / "ai_bots_trading.lock"
_held = []  # keeps the file open, and so locked, for the life of the process


def acquire_single_instance(path=LOCK_PATH):
    """True if this process now holds the lock, False if another bot already does."""
    handle = open(path, "a+b")
    try:
        try:
            import msvcrt
        except ImportError:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        return False
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()).encode())
    handle.flush()
    _held.append(handle)
    return True
