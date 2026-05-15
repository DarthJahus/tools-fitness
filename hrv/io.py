import os
import glob
import numpy as np
from utils import log, read_csv, parse_ts, LOG_FILE

ECG_SAMPLE_RATE = 130  # Hz — Polar H10 via Polar Sensor Logger

# ─────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────

def find_file(path, keyword, required=True):
    files = glob.glob(os.path.join(path, f"*{keyword}*.txt"))
    if files:
        return files[0]
    if required:
        raise FileNotFoundError(
            f"No file matching '*{keyword}*.txt' found in {path!r}"
        )
    return None

# ─────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────

def load_ecg(file):
    log(LOG_FILE, f"[ecg] {file}")
    rows = read_csv(file)
    ts, ecg = [], []
    skipped = 0
    for r in rows:
        if len(r) < 4:
            skipped += 1
            continue
        try:
            ts.append(parse_ts(r[0]))
            ecg.append(float(r[3]))
        except (ValueError, IndexError):
            skipped += 1
    if skipped:
        log(LOG_FILE, f"[ecg] {skipped} lines skipped (parse errors)")
    return np.array(ts), np.array(ecg)


def load_marker(file):
    if not file:
        log(LOG_FILE, "[marker] none found")
        return None, None

    log(LOG_FILE, f"[marker] {file}")
    rows = read_csv(file)

    start = stop = None
    for r in rows:
        if len(r) < 2: continue
        ts = parse_ts(r[0])
        label = r[1].strip().upper()

        if label == "MARKER_START": start = ts
        if label == "MARKER_STOP":  stop  = ts

    if start and stop:
        log(LOG_FILE, f"[marker] window: {start} → {stop}")
    else:
        log(LOG_FILE, "[marker] invalid or incomplete")

    return start, stop


