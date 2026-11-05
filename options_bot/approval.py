"""Ask a yes/no question in the terminal, with a time limit so the bot can't hang on it."""
import sys
import time


def _read_windows(msvcrt, timeout):
    while msvcrt.kbhit():  # ignore anything typed before the question appeared
        msvcrt.getwch()
    answer, deadline = "", time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not msvcrt.kbhit():
            time.sleep(0.05)
            continue
        ch = msvcrt.getwche()
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch in ("\x00", "\xe0"):  # arrow and function keys arrive as two characters
            msvcrt.getwch()
        elif ch in ("\r", "\n"):
            print()
            return answer
        elif ch == "\b":
            answer = answer[:-1]
            print(" \b", end="", flush=True)
        else:
            answer += ch
    return None


def _read_posix(timeout):
    import select

    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.readline() if ready else None


def ask_yes_no(question, timeout):
    """True only for a typed y or yes. No answer in time, or no terminal, counts as no."""
    print(f"{question} [y/N] ({timeout}s): ", end="", flush=True)
    if not sys.stdin.isatty():
        print("no terminal, skipped")
        return False
    try:
        import msvcrt
    except ImportError:
        answer = _read_posix(timeout)
    else:
        answer = _read_windows(msvcrt, timeout)
    if answer is None:
        print("no answer, skipped")
        return False
    return answer.strip().lower() in ("y", "yes")
