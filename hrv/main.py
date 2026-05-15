import os
from typing import TextIO
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from hrv.hrv import compute_rr_from_ecg, detect_ectopics, plot_ectopics, sliding_hrv, hrv_by_hour, plot_ecg_segment, \
    sleep_staging_rule_based, sleep_staging, run_hrv_detail
from hrv.io import find_file, load_ecg, load_marker
from utils import log, parse_ts

ECG_SAMPLE_RATE = 130  # Hz — Polar H10 via Polar Sensor Logger


LOG_FILE: TextIO = None

# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

def run(path, window_min, use_marker, hrv_detail, no_sleep, no_gru, custom_marker=None, output=None):

    if output:
        global LOG_FILE
        os.makedirs(output, exist_ok=True)
        ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(output, f"hrv_{ts_label}.txt")
        LOG_FILE = open(log_path, "w", encoding="utf-8")
        log(LOG_FILE, f"[output] logging to {log_path}")
    _run_inner(path, window_min, use_marker, hrv_detail, no_sleep, no_gru, custom_marker)


def _run_inner(path, window_min, use_marker, hrv_detail, no_sleep, no_gru, custom_marker):
    log("[init] discovering files")
    ecg_file    = find_file(path, "ECG", required=True)
    marker_file = find_file(path, "MARKER", required=False)

    ts_ecg, ecg = load_ecg(ecg_file)

    # ── marker resolution
    if custom_marker:
        m_start, m_stop = custom_marker
        log(LOG_FILE, f"[mode] custom marker: {m_start} → {m_stop}")
    elif use_marker:
        log(LOG_FILE, "[mode] marker ENABLED (default)")
        m_start, m_stop = load_marker(marker_file)
    else:
        log(LOG_FILE, "[mode] marker DISABLED (--no-marker)")
        m_start, m_stop = None, None

    peaks, rr = compute_rr_from_ecg(ecg, ECG_SAMPLE_RATE)
    rr_ts = ts_ecg[peaks][1:]

    # Filter physically impossible RR values (artefacts, BLE dropouts)
    RR_MIN_MS, RR_MAX_MS = 200, 3000
    valid_mask = (rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)
    n_removed = np.sum(~valid_mask)
    if n_removed:
        log(f"[core] {n_removed} RR intervals outside [{RR_MIN_MS}, {RR_MAX_MS}] ms removed (artefacts)")
    rr    = rr[valid_mask]
    rr_ts = rr_ts[valid_mask]

    # ── ectopiques
    ectopics = detect_ectopics(peaks, rr, rr_ts)
    plot_ectopics(rr_ts, rr, ectopics)

    # ── sliding HRV
    times, rmssd, hr = sliding_hrv(rr_ts, rr, window_min)
    log(LOG_FILE, f"[core] windows: {len(times)}")

    hrv_by_hour(rr, rr_ts, ectopics)

    if len(hr) > 10:
        corr = np.corrcoef(hr, rmssd)[0, 1]
        log(LOG_FILE, f"[core] HR vs RMSSD corr: {corr:.2f}")

    # ── full night stats (ectopics excluded)
    if ectopics and ectopics["count"] > 0:
        ect_idx = set(ectopics["indices"].tolist())
        clean_mask = np.array([i not in ect_idx for i in range(len(rr))])
        rr_clean = rr[clean_mask]
        log(f"[core] full-night metrics computed on {len(rr_clean)} RR ({len(rr) - len(rr_clean)} ectopics removed)")
    else:
        rr_clean = rr

    diff       = np.diff(rr_clean)
    rmssd_full = np.sqrt(np.mean(diff**2))
    sdnn_full  = np.std(rr_clean)
    hr_full    = 60000 / np.mean(rr_clean)

    log(LOG_FILE, "[core] FULL NIGHT:")
    log(LOG_FILE, f"  RMSSD: {rmssd_full:.1f} ms")
    log(LOG_FILE, f"  SDNN : {sdnn_full:.1f} ms")
    log(LOG_FILE, f"  HR   : {hr_full:.1f} bpm")

    # ── ECG segment plot
    plot_ecg_segment(ts_ecg, ecg, m_start, m_stop)

    # ── Poincaré (ectopics excluded)
    log(LOG_FILE, "[core] Poincaré plot")
    rr1 = rr_clean[:-1]
    rr2 = rr_clean[1:]

    plt.figure(figsize=(5, 5))
    plt.scatter(rr1, rr2, s=2)
    plt.xlabel("RR(n)")
    plt.ylabel("RR(n+1)")
    plt.title("Poincaré plot")
    plt.tight_layout()
    plt.show()

    sd1   = np.std((rr2 - rr1) / np.sqrt(2))
    sd2   = np.std((rr2 + rr1) / np.sqrt(2))
    ratio = sd1 / sd2 if sd2 > 0 else 0

    log(LOG_FILE, "\n[interpretation] Poincaré:")
    log(LOG_FILE, f"  SD1 (court terme): {sd1:.1f}")
    log(LOG_FILE, f"  SD2 (long terme):  {sd2:.1f}")
    log(LOG_FILE, f"  ratio SD1/SD2:     {ratio:.2f}")

    if ratio > 0.5:
        log(LOG_FILE, "  → variabilité court terme élevée (bonne récupération)")
    elif ratio > 0.3:
        log(LOG_FILE, "  → variabilité modérée")
    else:
        log(LOG_FILE, "  → variabilité faible (activation / fatigue possible)")

    log(LOG_FILE, "\n[lecture générale]:")
    log(LOG_FILE, "  SD1 = variabilité battement à battement")
    log(LOG_FILE, "  SD2 = tendance globale / régulation lente")
    log(LOG_FILE, "  nuage large = variabilité élevée")
    log(LOG_FILE, "  nuage serré = rigidité cardiaque")

    # ── sleep staging
    if not no_sleep:
        if no_gru:
            sleep_staging_rule_based(rr, rr_ts, times, rmssd, hr, ectopics)
        else:
            sleep_staging(ecg, ECG_SAMPLE_RATE, ts_ecg, times, rmssd, hr, ectopics)

    # ── hrv detail (optional)
    if hrv_detail:
        run_hrv_detail(rr, rr_ts, m_start, m_stop)


def main():
    description = """
Sleep HRV analysis (Polar Sensor Logger ECG export — Polar H10, 130 Hz)

Definitions:
- RR interval: time between heart beats (ms)
- HRV: variability of RR intervals
- RMSSD: short-term HRV (parasympathetic activity)
- SDNN: global HRV variability

Useful reading:
- https://en.wikipedia.org/wiki/Heart_rate_variability
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/
"""

    p = argparse.ArgumentParser(description=description)

    p.add_argument("--path",    required=True, help="Folder containing ECG/ACC/MARKER files")
    p.add_argument("--window",  type=int, default=5, help="Sliding window (minutes)")
    p.add_argument("--no-marker",   action="store_true", help="Ignore marker file → analyze full recording")
    p.add_argument("--no-gru",      action="store_true", help="Use rule-based hypnogram, without TensorFlow/GRU")
    p.add_argument("--no-sleep", action="store_true")
    p.add_argument("--hrv-detail",   action="store_true", help="Windowed HRV detail: RMSSD/SDNN/pNN50/DFAα1 by 5-min windows, aggregated by hour")
    p.add_argument("--output", metavar="DIR", default=None,
                   help="Directory to write a timestamped .txt log (default: stdout only)")
    p.add_argument(
        "--custom-marker",
        nargs=2,
        metavar=("START", "STOP"),
        help="Manual marker window, e.g. --custom-marker 2026-04-15T08:30:00 2026-04-15T08:34:00"
    )

    args = p.parse_args()

    custom_marker = None
    if args.custom_marker:
        try:
            custom_marker = (parse_ts(args.custom_marker[0]), parse_ts(args.custom_marker[1]))
        except ValueError as e:
            p.error(f"--custom-marker: {e}")

    run(
        args.path,
        args.window,
        use_marker=not args.no_marker,
        hrv_detail=args.hrv_detail,
        no_gru=args.no_gru,
        no_sleep=args.no_sleep,
        custom_marker=custom_marker,
        output=args.output,
    )

if __name__ == "__main__":
    main()

