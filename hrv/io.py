import os
import glob
import numpy as np
from hrv.utils import log, read_csv, parse_ts


def find_file(path, keyword, required=True):
    files = glob.glob(os.path.join(path, f"*{keyword}*.txt"))
    if files:
        return files[0]
    if required:
        raise FileNotFoundError(
            f"No file matching '*{keyword}*.txt' found in {path!r}"
        )
    return None


def load_ecg(file):
    log(f"[ecg] {file}")
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
        log(f"[ecg] {skipped} lines skipped (parse errors)")
    return np.array(ts), np.array(ecg)


def load_marker(file):
    if not file:
        log("[marker] none found")
        return None, None

    log(f"[marker] {file}")
    rows = read_csv(file)

    start = stop = None
    for r in rows:
        if len(r) < 2: continue
        ts = parse_ts(r[0])
        label = r[1].strip().upper()

        if label == "MARKER_START": start = ts
        if label == "MARKER_STOP":  stop  = ts

    if start and stop:
        log(f"[marker] window: {start} → {stop}")
    else:
        log("[marker] invalid or incomplete")

    return start, stop
