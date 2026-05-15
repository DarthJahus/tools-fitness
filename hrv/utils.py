import csv
from datetime import datetime
from typing import Optional, TextIO

_log_file: Optional[TextIO] = None


def init_log(path: str) -> None:
    """Open the log file. Call once from main before any log() call."""
    global _log_file
    _log_file = open(path, "w", encoding="utf-8")


def close_log() -> None:
    global _log_file
    if _log_file is not None:
        try:
            _log_file.close()
        except Exception:
            pass
        _log_file = None


def log(msg: str) -> None:
    print(msg, flush=True)
    if _log_file is not None:
        try:
            _log_file.write(msg + "\n")
            _log_file.flush()
        except Exception:
            pass


def parse_ts(s: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse timestamp: {s!r}")


def read_csv(filepath: str) -> list:
    with open(filepath, newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter=";")
        rows = [x for x in r if not (x and x[0].startswith("#"))]
    return rows[1:]
