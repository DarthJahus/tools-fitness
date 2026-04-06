import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="H10 data analyzer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rr",     metavar="FILE")
    group.add_argument("--ecg",    metavar="FILE")
    group.add_argument("--acc",    metavar="FILE")
    group.add_argument("--hr",     metavar="FILE")
    parser.add_argument("--marker", metavar="FILE", help="Optional marker file")
    return parser.parse_args()

def parse_ts(s):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse: {s}")

def read_csv(filepath, delimiter=";"):
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        lines = [r for r in reader if not (r and r[0].startswith("#"))]
    print(f"  [read_csv] {filepath}")
    print(f"  [read_csv] header : {lines[0]}")
    print(f"  [read_csv] {len(lines)-1} data rows")
    return lines[0], lines[1:]

def read_marker(filepath):
    print(f"\n[marker] Reading {filepath}")
    _, rows = read_csv(filepath)
    start = stop = None
    for row in rows:
        if len(row) < 2:
            continue
        ts    = parse_ts(row[0].strip())
        label = row[1].strip()
        if "START" in label:
            start = ts
            print(f"  [marker] START → {start}")
        elif "STOP" in label:
            stop = ts
            print(f"  [marker] STOP  → {stop}")
    if start and stop:
        print(f"  [marker] Duration: {stop - start}")
    return start, stop

# ── RR pipeline ──────────────────────────────────────────────────────────────

def pipeline_rr(filepath, marker_file=None):
    import neurokit2 as nk

    print(f"\n[rr] Reading {filepath}")
    header, rows = read_csv(filepath)

    timestamps, rr = [], []
    skipped = 0
    for row in rows:
        if len(row) < 2:
            skipped += 1
            continue
        try:
            timestamps.append(parse_ts(row[0].strip()))
            rr.append(float(row[1]))
        except ValueError:
            skipped += 1

    print(f"  [rr] Parsed   : {len(rr)} beats")
    print(f"  [rr] Skipped  : {skipped} rows")
    if timestamps:
        print(f"  [rr] Start    : {timestamps[0]}")
        print(f"  [rr] End      : {timestamps[-1]}")
        duration_s = sum(rr) / 1000.0
        print(f"  [rr] Total RR duration : {duration_s/3600:.2f}h ({duration_s:.0f}s)")
        print(f"  [rr] Mean HR  : {60000/np.mean(rr):.1f} bpm")

    timestamps = np.array(timestamps)
    rr = np.array(rr)

    # ── HRV window ──
    marker_start = marker_stop = None
    if marker_file:
        marker_start, marker_stop = read_marker(marker_file)

    if marker_start and marker_stop:
        mask = (timestamps >= marker_start) & (timestamps <= marker_stop)
        hrv_rr = rr[mask]
        print(f"\n[rr] HRV window: marker ({marker_start} → {marker_stop})")
        print(f"  [rr] Beats in window : {len(hrv_rr)}")
    else:
        hrv_rr, total = [], 0
        for val in rr:
            hrv_rr.append(val)
            total += val
            if total >= 300000:
                break
        hrv_rr = np.array(hrv_rr)
        print(f"\n[rr] HRV window: first 5 min (no marker)")
        print(f"  [rr] Beats in window : {len(hrv_rr)}")

    rr_time   = np.cumsum(hrv_rr) / 1000.0
    peaks_idx = np.round(rr_time * 1000).astype(int)

    print(f"\n[rr] Running nk.hrv()…")
    hrv = nk.hrv(peaks_idx, sampling_rate=1000, show=True)
    print("\n=== HRV RESULTS ===")
    print(hrv.T.to_string())

    # ── Night overview ──
    print(f"\n[rr] Building night overview (5-min window, 1-min step)…")
    window_ms  = 300_000
    step_ms    =  60_000
    cumsum_rr  = np.cumsum(rr)

    win_times, win_rmssd, win_hr = [], [], []
    pos = 0

    while pos < len(rr):
        start_ms = cumsum_rr[pos] - rr[pos]
        end_ms   = start_ms + window_ms
        idx = np.where((cumsum_rr >= start_ms) & (cumsum_rr <= end_ms))[0]
        if len(idx) < 10:
            print(f"  [rr] Stopping at pos={pos} (only {len(idx)} beats in window)")
            break
        w = rr[idx]
        win_rmssd.append(np.sqrt(np.mean(np.diff(w)**2)))
        win_hr.append(60000.0 / np.mean(w))
        win_times.append(timestamps[idx[0]])
        target = start_ms + step_ms
        while pos < len(rr) and cumsum_rr[pos] < target:
            pos += 1

    print(f"  [rr] Windows computed : {len(win_times)}")
    if win_times:
        print(f"  [rr] Overview start   : {win_times[0]}")
        print(f"  [rr] Overview end     : {win_times[-1]}")

    fig, ax1 = plt.subplots(figsize=(14, 5))
    fig.suptitle("Night overview — sliding 5-min window, 1-min step")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("RMSSD (ms)", color="tab:blue")
    ax1.plot(win_times, win_rmssd, color="tab:blue", linewidth=1.2, label="RMSSD")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    if marker_start:
        ax1.axvline(marker_start, color="green", linestyle="--", linewidth=1, label="Marker start")
    if marker_stop:
        ax1.axvline(marker_stop,  color="green", linestyle=":",  linewidth=1, label="Marker stop")

    ax2 = ax1.twinx()
    ax2.set_ylabel("HR (bpm)", color="tab:red")
    ax2.plot(win_times, win_hr, color="tab:red", linewidth=1.2, alpha=0.7, label="HR")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

# ── ECG pipeline ─────────────────────────────────────────────────────────────

def pipeline_ecg(filepath, marker_file=None):
    import neurokit2 as nk

    SAMPLING_RATE = 130

    print(f"\n[ecg] Reading {filepath}")
    header, rows = read_csv(filepath)

    timestamps, ecg = [], []
    skipped = 0
    for row in rows:
        if len(row) < 4:
            skipped += 1
            continue
        try:
            timestamps.append(parse_ts(row[0].strip()))
            ecg.append(float(row[3]))
        except ValueError:
            skipped += 1

    ecg        = np.array(ecg, dtype=float)
    timestamps = np.array(timestamps)

    print(f"  [ecg] Parsed   : {len(ecg):,} samples")
    print(f"  [ecg] Skipped  : {skipped} rows")
    if len(timestamps):
        print(f"  [ecg] Start    : {timestamps[0]}")
        print(f"  [ecg] End      : {timestamps[-1]}")
        print(f"  [ecg] Duration : {len(ecg)/SAMPLING_RATE/3600:.2f}h")
    print(f"  [ecg] Signal range : {ecg.min():.0f} → {ecg.max():.0f} µV")

    # ── Window ──
    marker_start = marker_stop = None
    if marker_file:
        marker_start, marker_stop = read_marker(marker_file)

    if marker_start and marker_stop:
        mask = (timestamps >= marker_start) & (timestamps <= marker_stop)
        plot_ecg = ecg[mask]
        plot_ts  = timestamps[mask]
        title_suffix = f"marker window ({marker_start.strftime('%H:%M')} → {marker_stop.strftime('%H:%M')})"
        print(f"\n[ecg] Using marker window: {len(plot_ecg):,} samples ({len(plot_ecg)/SAMPLING_RATE:.1f}s)")
    else:
        n = SAMPLING_RATE * 30
        plot_ecg = ecg[:n]
        plot_ts  = timestamps[:n]
        title_suffix = "first 30 seconds"
        print(f"\n[ecg] No marker — using first 30s ({n} samples)")

    # skip startup artifact
    skip = SAMPLING_RATE * 2
    plot_ecg = plot_ecg[skip:]
    plot_ts  = plot_ts[skip:]
    print(f"  [ecg] After skipping 2s artifact : {len(plot_ecg):,} samples")
    print(f"  [ecg] Signal range after skip    : {plot_ecg.min():.0f} → {plot_ecg.max():.0f} µV")

    print(f"  [ecg] Cleaning signal…")
    ecg_cleaned = nk.ecg_clean(plot_ecg, sampling_rate=SAMPLING_RATE)
    print(f"  [ecg] Detecting R-peaks…")
    peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=SAMPLING_RATE)
    r_peaks = info["ECG_R_Peaks"]
    print(f"  [ecg] R-peaks detected : {len(r_peaks):,}")
    if len(r_peaks) > 1:
        mean_rr = np.mean(np.diff(r_peaks)) / SAMPLING_RATE * 1000
        print(f"  [ecg] Mean RR from peaks : {mean_rr:.0f} ms ({60000/mean_rr:.1f} bpm)")

    t = np.arange(len(plot_ecg)) / SAMPLING_RATE
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(f"ECG — {title_suffix} with R-peaks")
    ax.plot(t, ecg_cleaned, linewidth=0.6, color="tab:blue", label="ECG")
    ax.scatter(r_peaks / SAMPLING_RATE, ecg_cleaned[r_peaks],
               color="red", zorder=5, s=20, label="R-peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend()
    plt.tight_layout()
    plt.show()

# ── ACC pipeline ─────────────────────────────────────────────────────────────

def pipeline_acc(filepath):
    print(f"\n[acc] Reading {filepath}")
    header, rows = read_csv(filepath)

    timestamps, X, Y, Z = [], [], [], []
    skipped = 0
    for row in rows:
        if len(row) < 5:
            skipped += 1
            continue
        try:
            timestamps.append(parse_ts(row[0].strip()))
            X.append(float(row[2]))
            Y.append(float(row[3]))
            Z.append(float(row[4]))
        except ValueError:
            skipped += 1

    print(f"  [acc] Parsed   : {len(timestamps):,} samples")
    print(f"  [acc] Skipped  : {skipped} rows")
    if timestamps:
        print(f"  [acc] Start    : {timestamps[0]}")
        print(f"  [acc] End      : {timestamps[-1]}")
        print(f"  [acc] X range  : {min(X):.0f} → {max(X):.0f} mg")
        print(f"  [acc] Y range  : {min(Y):.0f} → {max(Y):.0f} mg")
        print(f"  [acc] Z range  : {min(Z):.0f} → {max(Z):.0f} mg")

    step = 50
    print(f"  [acc] Downsampling x{step} for plot ({len(timestamps)//step:,} points)")
    ts = timestamps[::step]
    Xd = np.array(X[::step])
    Yd = np.array(Y[::step])
    Zd = np.array(Z[::step])

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Accelerometer — night (downsampled)")
    for ax, data, label, color in zip(
        axes,
        [Xd, Yd, Zd],
        ["X [mg]", "Y [mg]", "Z [mg]"],
        ["tab:blue", "tab:orange", "tab:green"]
    ):
        ax.plot(ts, data, linewidth=0.5, color=color)
        ax.set_ylabel(label)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

# ── HR pipeline ──────────────────────────────────────────────────────────────

def pipeline_hr(filepath, marker_file=None):
    print(f"\n[hr] Reading {filepath}")
    header, rows = read_csv(filepath)

    timestamps, hr = [], []
    skipped = 0
    for row in rows:
        if len(row) < 2:
            skipped += 1
            continue
        try:
            timestamps.append(parse_ts(row[0].strip()))
            hr.append(float(row[1]))
        except ValueError:
            skipped += 1

    print(f"  [hr] Parsed   : {len(hr):,} samples")
    print(f"  [hr] Skipped  : {skipped} rows")
    if timestamps:
        print(f"  [hr] Start    : {timestamps[0]}")
        print(f"  [hr] End      : {timestamps[-1]}")
        print(f"  [hr] HR range : {min(hr):.0f} → {max(hr):.0f} bpm")
        print(f"  [hr] Mean HR  : {np.mean(hr):.1f} bpm")

    marker_start = marker_stop = None
    if marker_file:
        marker_start, marker_stop = read_marker(marker_file)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title("Heart Rate — night")
    ax.plot(timestamps, hr, linewidth=0.8, color="tab:red")
    ax.set_ylabel("HR (bpm)")
    ax.set_xlabel("Time")
    if marker_start:
        ax.axvline(marker_start, color="green", linestyle="--", linewidth=1, label="Marker start")
    if marker_stop:
        ax.axvline(marker_stop,  color="green", linestyle=":",  linewidth=1, label="Marker stop")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    m = args.marker
    if args.rr:
        pipeline_rr(args.rr, m)
    elif args.ecg:
        pipeline_ecg(args.ecg, m)
    elif args.acc:
        pipeline_acc(args.acc)
    elif args.hr:
        pipeline_hr(args.hr, m)

if __name__ == "__main__":
    main()
