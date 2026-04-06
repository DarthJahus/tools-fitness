import os
import glob
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

import neurokit2 as nk

# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────

def log(msg):
    print(msg, flush=True)

def parse_ts(s):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except:
            pass
    raise ValueError(s)

def read_csv(filepath):
    with open(filepath, newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter=";")
        rows = [x for x in r if not (x and x[0].startswith("#"))]
    return rows[1:]

# ─────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────

def find_file(path, keyword):
    files = glob.glob(os.path.join(path, f"*{keyword}*.txt"))
    return files[0] if files else None

# ─────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────

def load_ecg(file):
    log(f"[ecg] {file}")
    rows = read_csv(file)
    ts, ecg = [], []
    for r in rows:
        if len(r) < 4: continue
        try:
            ts.append(parse_ts(r[0]))
            ecg.append(float(r[3]))
        except:
            continue
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

# ─────────────────────────────────────────────────────────────
# CORE
# ─────────────────────────────────────────────────────────────

def plot_ecg_segment(ts_ecg, ecg, m_start, m_stop):
    if not (m_start and m_stop):
        log("[ecg] no marker → skip ECG segment")
        return

    mid = m_start + (m_stop - m_start)/2
    start = mid - timedelta(seconds=30)
    end   = mid + timedelta(seconds=30)

    log(f"[ecg] plotting 60s segment around {mid}")

    mask = (ts_ecg >= start) & (ts_ecg <= end)

    if np.sum(mask) < 10:
        log("[ecg] insufficient data for segment")
        return

    plt.figure(figsize=(12,4))
    plt.plot(ts_ecg[mask], ecg[mask])
    plt.title(f"ECG segment (centered) {mid.strftime('%H:%M:%S')}")
    plt.xlabel("Time")
    plt.ylabel("ECG")
    plt.tight_layout()
    plt.show()

def compute_rr_from_ecg(ecg, sr):
    log("[core] cleaning ECG")
    ecg_clean = nk.ecg_clean(ecg, sampling_rate=sr)

    log("[core] detecting R peaks")
    _, info = nk.ecg_peaks(ecg_clean, sampling_rate=sr, correct_artifacts=True)

    peaks = info["ECG_R_Peaks"]
    rr = np.diff(peaks) / sr * 1000

    log(f"[core] peaks: {len(peaks)}")
    log(f"[core] rr intervals: {len(rr)}")

    return peaks, rr

def sliding_hrv(timestamps, rr, window_min):
    window = timedelta(minutes=window_min)
    step   = timedelta(minutes=1)

    times, rmssd, hr = [], [], []

    pos = 0
    while pos < len(timestamps):
        start = timestamps[pos]
        end   = start + window

        idx = np.where((timestamps >= start) & (timestamps <= end))[0]
        if len(idx) < 20:
            break

        w = rr[idx]

        diff = np.diff(w)
        rmssd.append(np.sqrt(np.mean(diff**2)))
        hr.append(60000 / np.mean(w))
        times.append(start)

        target = start + step
        while pos < len(timestamps) and timestamps[pos] < target:
            pos += 1

    return times, rmssd, hr

# ─────────────────────────────────────────────────────────────

def run(path, window_min, use_marker, use_full):

    log("[init] discovering files")

    ecg_file = find_file(path, "ECG")
    marker_file = find_file(path, "MARKER")

    ts_ecg, ecg = load_ecg(ecg_file)

    if use_marker:
        log("[mode] marker ENABLED (default)")
        m_start, m_stop = load_marker(marker_file)
    else:
        log("[mode] marker DISABLED (--no-marker)")
        m_start, m_stop = None, None

    SR = 130

    peaks, rr = compute_rr_from_ecg(ecg, SR)
    rr_ts = ts_ecg[peaks][1:]

    # ── sliding HRV
    times, rmssd, hr = sliding_hrv(rr_ts, rr, window_min)
    log(f"[core] windows: {len(times)}")

    # ── correlation
    if len(hr) > 10:
        corr = np.corrcoef(hr, rmssd)[0,1]
        log(f"[core] HR vs RMSSD corr: {corr:.2f}")

    # ─────────────────────────
    # FULL NIGHT FAST
    # ─────────────────────────
    diff = np.diff(rr)
    rmssd_full = np.sqrt(np.mean(diff**2))
    sdnn_full  = np.std(rr)
    hr_full    = 60000 / np.mean(rr)

    log("[core] FULL NIGHT:")
    log(f"  RMSSD: {rmssd_full:.1f} ms")
    log(f"  SDNN : {sdnn_full:.1f} ms")
    log(f"  HR   : {hr_full:.1f} bpm")

    # ─────────────────────────
    # GRAPH OVERVIEW (FIXED)
    # ─────────────────────────
    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.plot(times, rmssd, label="RMSSD (ms)", color="tab:blue")
    ax1.set_ylabel("RMSSD (ms)")

    ax2 = ax1.twinx()
    ax2.plot(times, hr, linestyle="--", label="HR (bpm)", color="tab:red")
    ax2.set_ylabel("HR (bpm)")

    # marker overlay VISUEL uniquement
    if m_start:
        ax1.axvline(m_start, linestyle="--", color="green", label="Marker START")
    if m_stop:
        ax1.axvline(m_stop, linestyle="--", color="orange", label="Marker STOP")

    # fusion légendes
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels)

    title = "Sleep overview (FULL NIGHT)"
    if m_start and m_stop:
        title += f"\nMarker: {m_start.strftime('%H:%M')} → {m_stop.strftime('%H:%M')}"

    plt.title(title)
    plt.tight_layout()
    plt.show()

    log("\n[graph] Sleep overview:")
    log("  RMSSD ↑ = récupération parasympathique")
    log("  HR ↓ = repos profond")
    log("  HR↑ + RMSSD↓ = stress / activation")

    # ECG
    plot_ecg_segment(ts_ecg, ecg, m_start, m_stop)

    # ───────────────────────
    # POINCARÉ + INTERPRÉTATION
    # ─────────────────────────
    log("[core] Poincaré plot")

    rr1 = rr[:-1]
    rr2 = rr[1:]

    plt.figure(figsize=(5,5))
    plt.scatter(rr1, rr2, s=2)
    plt.xlabel("RR(n)")
    plt.ylabel("RR(n+1)")
    plt.title("Poincaré plot")
    plt.tight_layout()
    plt.show()

    # interprétation simple
    sd1 = np.std((rr2 - rr1) / np.sqrt(2))
    sd2 = np.std((rr2 + rr1) / np.sqrt(2))

    log("\n[interpretation] Poincaré (TES DONNÉES):")
    log(f"  SD1 (court terme): {sd1:.1f}")
    log(f"  SD2 (long terme): {sd2:.1f}")

    ratio = sd1 / sd2 if sd2 > 0 else 0
    log(f"  ratio SD1/SD2: {ratio:.2f}")

    if ratio > 0.5:
        log("  → variabilité court terme élevée (bonne récupération)")
    elif ratio > 0.3:
        log("  → variabilité modérée")
    else:
        log("  → variabilité faible (activation / fatigue possible)")

    log("\n[lecture générale]:")
    log("  SD1 = variabilité battement à battement")
    log("  SD2 = tendance globale / régulation lente")
    log("  nuage large = variabilité élevée")
    log("  nuage serré = rigidité cardiaque")

    # ─────────────────────────
    # NEUROKIT (OPTIONNEL)
    # ─────────────────────────
    if use_full:
        log("[core] neurokit FULL mode (marker-limited)")

        if m_start and m_stop:
            mask = (rr_ts >= m_start) & (rr_ts <= m_stop)
            rr_nk = rr[mask]
        else:
            log("[core] no valid marker → using full RR (slow)")
            rr_nk = rr

        peaks_idx = np.round(np.cumsum(rr_nk)).astype(int)
        nk.hrv(peaks_idx, sampling_rate=1000, show=True)

# ─────────────────────────────────────────────────────────────

def main():
    description = """
Sleep HRV analysis (Garmin ECG export)

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

    p.add_argument("--path", required=True, help="Folder containing ECG/ACC/MARKER files")
    p.add_argument("--window", type=int, default=5, help="Sliding window (minutes)")

    p.add_argument("--no-marker", action="store_true",
                   help="Ignore marker file → analyze full recording")

    p.add_argument("--full", action="store_true",
                   help="Run NeuroKit full HRV analysis (slow)")

    args = p.parse_args()

    run(
        args.path,
        args.window,
        use_marker=not args.no_marker,
        use_full=args.full
    )

if __name__ == "__main__":
    main()
