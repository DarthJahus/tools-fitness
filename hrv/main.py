import os
import numpy as np
from datetime import datetime
import argparse
from hrv.core import (
    compute_rr_from_ecg,
    run_mode_night,
    run_mode_readiness,
    run_mode_exercise
)
from hrv.io import find_file, load_ecg, load_marker
from hrv.utils import parse_ts, init_log, close_log, log
from constants import ECG_SAMPLE_RATE, RR_MIN_MS, RR_MAX_MS


def run(path, window_min, use_marker, hrv_detail, no_sleep, no_gru, custom_marker=None, output=None, mode="night"):
    if output:
        os.makedirs(output, exist_ok=True)
        ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(output, f"hrv_{mode}_{ts_label}.txt")
        init_log(log_path)
        log(f"[output] logging to {log_path}")
    try:
        _run_inner(path, window_min, use_marker, hrv_detail, no_sleep, no_gru, custom_marker, mode)
    finally:
        close_log()
        

def _run_inner(path, window_min, use_marker, hrv_detail, no_sleep, no_gru, custom_marker, mode):
    log(f"[init] discovering files for mode: {mode.upper()}")
    ecg_file    = find_file(path, "ECG", required=True)
    marker_file = find_file(path, "MARKER", required=False)

    ts_ecg, ecg = load_ecg(ecg_file)

    # ── marker resolution
    if custom_marker:
        m_start, m_stop = custom_marker
        log(f"[mode] custom marker: {m_start} → {m_stop}")
    elif use_marker:
        log("[mode] marker ENABLED (default)")
        m_start, m_stop = load_marker(marker_file)
    else:
        log("[mode] marker DISABLED (--no-marker)")
        m_start, m_stop = None, None

    peaks, rr = compute_rr_from_ecg(ecg, ECG_SAMPLE_RATE)
    rr_ts = ts_ecg[peaks][1:]

    # Filter physically impossible RR values (artefacts, BLE dropouts)
    valid_mask = (rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)
    n_removed = np.sum(~valid_mask)
    if n_removed:
        log(f"[core] {n_removed} RR intervals outside [{RR_MIN_MS}, {RR_MAX_MS}] ms removed (artefacts)")
    rr    = rr[valid_mask]
    rr_ts = rr_ts[valid_mask]

    # Filter by custom/file marker window if provided before operational processing
    if m_start and m_stop:
        mask = (rr_ts >= m_start) & (rr_ts <= m_stop)
        rr = rr[mask]
        rr_ts = rr_ts[mask]

    # Execute designated execution stream based on mode selection
    if mode == "night":
        run_mode_night(
            rr, rr_ts, ecg, ts_ecg, window_min, hrv_detail,
            no_sleep, no_gru, m_start, m_stop
        )
    elif mode == "readiness":
        run_mode_readiness(rr, rr_ts)
    elif mode == "exercise":
        run_mode_exercise(rr, rr_ts, window_min)


def main():
    description = """
Polar H10 HRV Advanced Analyzer (Supports Night, Readiness, and Exercise modes)

Definitions:
- RR interval: time between heart beats (ms)
- HRV: variability of RR intervals
- RMSSD: short-term HRV (parasympathetic activity)
- SDNN: global HRV variability
"""

    p = argparse.ArgumentParser(description=description)

    p.add_argument(
        "--mode",
        choices=["night", "readiness", "exercise"],
        default="night",
        help="Operational routine context (default: night)"
    )
    p.add_argument("--path",    required=True, help="Folder containing ECG/ACC/MARKER files")
    p.add_argument("--window",  type=int, default=5, help="Sliding window (minutes)")
    p.add_argument("--no-marker",   action="store_true", help="Ignore marker file → analyze full recording")
    p.add_argument("--no-gru",      action="store_true", help="Use rule-based hypnogram, without TensorFlow/GRU")
    p.add_argument("--no-sleep", action="store_true", help="Disable hypnogram staging parsing loops")
    p.add_argument("--hrv-detail",   action="store_true", help="Windowed HRV detail tracking statistics aggregation")
    p.add_argument("--output", metavar="DIR", default=None, help="Directory to write a timestamped .txt log")
    p.add_argument(
        "--custom-marker",
        nargs=2,
        metavar=("START", "STOP"),
        help="Manual marker window, e.g. --custom-marker 2026-04-15T08:30:00 2026-04-15T08:34:00"
    )

    args = p.parse_args()

    if args.mode == "readiness" and not args.custom_marker:
        p.error("--mode readiness explicitly requires an execution frame bounded via --custom-marker")

    custom_marker = None
    if args.custom_marker:
        try:
            custom_marker = (parse_ts(args.custom_marker[0]), parse_ts(args.custom_marker[1]))
        except ValueError as e:
            p.error(f"--custom-marker: {e}")

    run(
        path=args.path,
        window_min=args.window,
        use_marker=not args.no_marker,
        hrv_detail=args.hrv_detail,
        no_gru=args.no_gru,
        no_sleep=args.no_sleep,
        custom_marker=custom_marker,
        output=args.output,
        mode=args.mode
    )

if __name__ == "__main__":
    main()
