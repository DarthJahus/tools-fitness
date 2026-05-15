import csv
from datetime import datetime
from typing import TextIO

ECG_SAMPLE_RATE = 130  # Hz — Polar H10 via Polar Sensor Logger

LOG_FILE: TextIO = None


def log(log_file, msg):
    global LOG_FILE
    LOG_FILE = log_file
    print(msg, flush=True)
    if log_file is not None:
        try:
            log_file.write(msg + "\n")
            log_file.flush()
        except Exception:
            pass


def parse_ts(s):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except:
            pass
    raise ValueError(f"Cannot parse timestamp: {s!r}")


def read_csv(filepath):
    with open(filepath, newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter=";")
        rows = [x for x in r if not (x and x[0].startswith("#"))]
    return rows[1:]
