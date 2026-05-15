import os
import argparse
import numpy as np
from datetime import datetime, timedelta
from core import compute_rr_from_ecg, run_mode_night, run_mode_readiness, run_mode_exercise
from file_io import find_file, load_ecg, load_marker
from utils import init_log, close_log, log, parse_ts
from constants import ECG_SAMPLE_RATE, RR_MIN_MS, RR_MAX_MS


def _shared_parser() -> argparse.ArgumentParser:
    """Common arguments inherited by all subcommands."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--path",    required=True, help="Folder containing ECG/MARKER files")
    p.add_argument("--output",  metavar="DIR", default=None,
                   help="Directory to write a timestamped .txt log")
    p.add_argument("--no-marker", action="store_true",
                   help="Ignore marker file — analyze full recording")
    p.add_argument(
        "--custom-marker",
        nargs=2,
        metavar=("START", "STOP_OR_MIN"),
        help="Marker window: START and either a stop timestamp or a duration in minutes "
             "(e.g. --custom-marker 2026-05-15T09:49:00 2026-05-15T09:54:00  "
             "or  --custom-marker 2026-05-15T09:49:00 5)"
    )
    return p


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="main.py",
        description="Polar H10 HRV Advanced Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Definitions:\n"
            "  RR interval : time between heart beats (ms)\n"
            "  HRV         : variability of RR intervals\n"
            "  RMSSD       : short-term HRV (parasympathetic activity)\n"
            "  SDNN        : global HRV variability\n"
        )
    )

    shared = _shared_parser()
    sub = root.add_subparsers(dest="mode", required=True, metavar="MODE",
                              help="night | readiness | exercise")

    # ── night ────────────────────────────────────────────────
    night = sub.add_parser("night", parents=[shared],
                           help="Full overnight recovery analysis (default)")
    night.add_argument("--window",     type=int, default=5,
                       help="Sliding HRV window in minutes (default: 5)")
    night.add_argument("--no-gru",     action="store_true",
                       help="Rule-based sleep staging — skips TensorFlow/GRU")
    night.add_argument("--no-sleep",   action="store_true",
                       help="Disable sleep staging entirely")
    night.add_argument("--hrv-detail", action="store_true",
                       help="5-min windowed RMSSD/SDNN/pNN50/DFA α1, aggregated by hour")

    # ── readiness ────────────────────────────────────────────
    sub.add_parser("readiness", parents=[shared],
                   help="Short resting window (3–10 min) — requires --custom-marker")

    # ── exercise ─────────────────────────────────────────────
    exercise = sub.add_parser("exercise", parents=[shared],
                              help="Exercise session analysis (10-min segments)")
    exercise.add_argument("--window", type=int, default=10,
                          help="Segment duration in minutes (default: 10)")

    return root


def resolve_custom_marker(raw: list | None, p: argparse.ArgumentParser):
    """
    Parse --custom-marker START STOP_OR_MIN.
    The second argument is tried as an ISO timestamp first, then as float minutes.
    Returns (start, stop) as datetime objects, or None.
    """
    if not raw:
        return None

    start_str, second_str = raw

    try:
        start = parse_ts(start_str)
    except ValueError as e:
        p.error(f"--custom-marker START: {e}")

    try:
        stop = parse_ts(second_str)
    except ValueError:
        try:
            stop = start + timedelta(minutes=float(second_str))
        except ValueError:
            p.error(
                f"--custom-marker: second argument must be a stop timestamp or a duration "
                f"in minutes, got {second_str!r}"
            )

    return start, stop


def parse_args():
    p = build_parser()
    args = p.parse_args()

    if args.mode == "readiness" and not args.custom_marker:
        p.error("readiness mode requires --custom-marker (START and stop timestamp or duration)")

    custom_marker = resolve_custom_marker(args.custom_marker, p)

    return args, custom_marker

def run(args, custom_marker):
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(args.output, f"hrv_{args.mode}_{ts_label}.txt")
        init_log(log_path)
        log(f"[output] logging to {log_path}")

    try:
        _run_inner(args, custom_marker)
    finally:
        close_log()


def _run_inner(args, custom_marker):
    log(f"[init] discovering files for mode: {args.mode.upper()}")
    ecg_file    = find_file(args.path, "ECG", required=True)
    marker_file = find_file(args.path, "MARKER", required=False)

    ts_ecg, ecg = load_ecg(ecg_file)

    # ── marker resolution ─────────────────────────────────────
    if custom_marker:
        m_start, m_stop = custom_marker
        log(f"[mode] custom marker: {m_start} → {m_stop}")
    elif not args.no_marker:
        log("[mode] marker ENABLED (default)")
        m_start, m_stop = load_marker(marker_file)
    else:
        log("[mode] marker DISABLED (--no-marker)")
        m_start, m_stop = None, None

    # ── signal processing ─────────────────────────────────────
    peaks, rr = compute_rr_from_ecg(ecg, ECG_SAMPLE_RATE)
    rr_ts = ts_ecg[peaks][1:]

    valid_mask = (rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)
    n_removed  = int(np.sum(~valid_mask))
    if n_removed:
        log(f"[core] {n_removed} RR intervals outside [{RR_MIN_MS}, {RR_MAX_MS}] ms removed (artefacts)")
    rr    = rr[valid_mask]
    rr_ts = rr_ts[valid_mask]

    if m_start and m_stop:
        mask  = (rr_ts >= m_start) & (rr_ts <= m_stop)
        rr    = rr[mask]
        rr_ts = rr_ts[mask]

    # ── mode dispatch ─────────────────────────────────────────
    if args.mode == "night":
        run_mode_night(
            rr, rr_ts, ecg, ts_ecg,
            window_min=args.window,
            hrv_detail=args.hrv_detail,
            no_sleep=args.no_sleep,
            no_gru=args.no_gru,
            m_start=m_start,
            m_stop=m_stop
        )
    elif args.mode == "readiness":
        run_mode_readiness(rr, rr_ts)
    elif args.mode == "exercise":
        run_mode_exercise(rr, rr_ts, window_min=args.window)

def main():
    args, custom_marker = parse_args()
    run(args, custom_marker)


if __name__ == "__main__":
    main()
