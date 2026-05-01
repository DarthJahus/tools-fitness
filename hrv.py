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

def find_gaps(rr_ts, threshold_s=30):
    """Return list of (gap_start, gap_end, duration_s) for gaps > threshold_s."""
    timestamps = [t.timestamp() for t in rr_ts]
    diffs = np.diff(timestamps)
    gap_idx = np.where(diffs > threshold_s)[0]
    return [(rr_ts[i], rr_ts[i + 1], diffs[i]) for i in gap_idx]


def resolve_marker_rr(rr, rr_ts, m_start, m_stop):
    """
    Return the RR slice matching the marker window, or None if the window
    falls in a recording gap. Logs a clear diagnostic in that case.
    """
    mask = np.array([(t >= m_start) and (t <= m_stop) for t in rr_ts])
    rr_nk = rr[mask]

    if len(rr_nk) > 0:
        return rr_nk

    # Zero matches — figure out why
    log(f"[marker] WARNING: window [{m_start} → {m_stop}] matches 0 RR intervals")
    log(f"[marker]   recording range: {rr_ts[0]} → {rr_ts[-1]}")

    gaps = find_gaps(rr_ts)
    covering_gap = None
    for g_start, g_end, g_dur in gaps:
        if g_start <= m_start and g_end >= m_stop:
            covering_gap = (g_start, g_end, g_dur)
            break

    if covering_gap:
        g_start, g_end, g_dur = covering_gap
        log(f"[marker]   → marker falls inside a recording gap of {g_dur:.0f}s")
        log(f"[marker]     gap: {g_start} → {g_end}")
        log("[marker]   no data available for this window — skipping nk.hrv")
        return None
    else:
        log("[marker]   → cause unknown (marker outside recording range?)")
        log("[marker]   falling back to full RR")
        return rr


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


def detect_ectopics(ecg, peaks, rr, rr_ts, sr, window_beats=30, premature_ratio=0.80):
    """
    Détection des extrasystoles depuis les intervalles RR.
    Classification SVEB/VEB par critère de pause compensatoire.
    Détection des couplets/triplets/runs.
    """
    n = len(rr)
    if n < window_beats * 2:
        log("[ectopic] pas assez d'intervalles RR")
        return None

    # Médiane locale glissante (suit la dérive nocturne de la FC)
    half_w = window_beats // 2
    local_median = np.array([
        np.median(rr[max(0, i - half_w):min(n, i + half_w)])
        for i in range(n)
    ])

    # Détection : RR court = battement prématuré
    ectopic_mask = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if rr[i] < premature_ratio * local_median[i]:
            ectopic_mask[i] = True

    ectopic_indices = np.where(ectopic_mask)[0]

    if len(ectopic_indices) == 0:
        log("[ectopic] aucune extrasystole détectée")
        return {"count": 0, "sveb": 0, "veb": 0,
                "couplets": 0, "triplets": 0, "runs": 0,
                "indices": [], "types": [], "timestamps": []}

    # Classification SVEB / VEB par pause compensatoire
    types = []
    for i in ectopic_indices:
        if i + 1 < n:
            compensatory_sum = rr[i] + rr[i + 1]
            if compensatory_sum >= 1.8 * local_median[i]:
                types.append("VEB")   # pause complète → ventriculaire
            else:
                types.append("SVEB")  # pause incomplète → supraventriculaire
        else:
            types.append("unknown")

    # Détection couplets / triplets / runs (ectopiques consécutifs)
    couplets = triplets = runs = 0
    i = 0
    while i < len(ectopic_indices):
        run_len = 1
        while (i + run_len < len(ectopic_indices) and
               ectopic_indices[i + run_len] == ectopic_indices[i + run_len - 1] + 1):
            run_len += 1
        if run_len == 2:
            couplets += 1
        elif run_len == 3:
            triplets += 1
        elif run_len > 3:
            runs += 1
            run_start = rr_ts[ectopic_indices[i]]
            run_end = rr_ts[min(ectopic_indices[i + run_len - 1], len(rr_ts) - 1)]
            run_dur_s = (run_end - run_start).total_seconds()
            run_rr = rr[ectopic_indices[i]:ectopic_indices[i] + run_len]
            run_bpm = 60000 / np.mean(run_rr)
            log(f"  Run @{run_start.strftime('%H:%M:%S')} — {run_len} beats — {run_bpm:.0f} bpm — {run_dur_s:.1f}s")
        i += run_len

    timestamps = [rr_ts[i] for i in ectopic_indices if i < len(rr_ts)]
    total = len(ectopic_indices)
    n_sveb = types.count("SVEB")
    n_veb  = types.count("VEB")

    log(f"\n[ectopic] ── EXTRASYSTOLES ──────────────────────")
    log(f"  Total        : {total}")
    log(f"  SVEB (supraven.) : {n_sveb} ({n_sveb/total*100:.1f}%)")
    log(f"  VEB  (ventric.)  : {n_veb}  ({n_veb/total*100:.1f}%)")
    log(f"  Couplets     : {couplets}")
    log(f"  Triplets     : {triplets}")
    log(f"  Runs (>3)    : {runs}")

    if len(rr_ts) > 1:
        duration_h = (rr_ts[-1] - rr_ts[0]).total_seconds() / 3600
        if duration_h > 0:
            log(f"  Fréquence    : {total / duration_h:.1f} /heure")

    log(f"────────────────────────────────────────────────")

    return {
        "count": total, "sveb": n_sveb, "veb": n_veb,
        "couplets": couplets, "triplets": triplets, "runs": runs,
        "indices": ectopic_indices, "types": types, "timestamps": timestamps,
    }


def plot_ectopics(rr_ts, rr, ectopics):
    if not ectopics or ectopics["count"] == 0:
        return

    indices = ectopics["indices"]
    types   = ectopics["types"]
    ts_ect  = ectopics["timestamps"]
    rr_ect  = rr[indices]

    colors = {"SVEB": "#e67e22", "VEB": "#e74c3c", "unknown": "#95a5a6"}
    c = [colors.get(t, "#95a5a6") for t in types]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Distribution des extrasystoles")

    # Tachogramme RR + ectopiques marqués
    ax1.plot(rr_ts, rr, color="#2980b9", linewidth=0.6, alpha=0.7, label="RR (ms)")
    ax1.scatter(ts_ect, rr_ect, c=c, s=18, zorder=5, label="Ectopique")
    ax1.set_ylabel("RR (ms)")

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="#e67e22", label=f"SVEB ({ectopics['sveb']})"),
        Patch(color="#e74c3c", label=f"VEB ({ectopics['veb']})"),
    ]
    ax1.legend(handles=legend_patches + ax1.get_lines()[:1], fontsize=8)

    # Densité horaire des ectopiques
    if len(ts_ect) > 1:
        import matplotlib.dates as mdates
        ax2.hist(
            mdates.date2num(ts_ect),
            bins=max(1, int((rr_ts[-1] - rr_ts[0]).total_seconds() / 600)),  # bins de 10 min
            color="#c0392b", alpha=0.7
        )
        ax2.xaxis_date()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.set_ylabel("Ectopiques / 10 min")
        ax2.set_xlabel("Temps")

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


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


def plot_ecg_segment(ts_ecg, ecg, m_start, m_stop):
    if not (m_start and m_stop):
        log("[ecg] no marker → skip ECG segment")
        return

    mid   = m_start + (m_stop - m_start) / 2
    start = mid - timedelta(seconds=30)
    end   = mid + timedelta(seconds=30)

    log(f"[ecg] plotting 60s segment around {mid}")

    mask = (ts_ecg >= start) & (ts_ecg <= end)
    if np.sum(mask) < 10:
        log("[ecg] insufficient data for segment")
        return

    plt.figure(figsize=(12, 4))
    plt.plot(ts_ecg[mask], ecg[mask])
    plt.title(f"ECG segment (centered) {mid.strftime('%H:%M:%S')}")
    plt.xlabel("Time")
    plt.ylabel("ECG")
    plt.tight_layout()
    plt.show()


def run_neurokit(rr, rr_ts, m_start, m_stop):
    log("[nk] neurokit FULL mode")

    if m_start and m_stop:
        rr_nk = resolve_marker_rr(rr, rr_ts, m_start, m_stop)
        if rr_nk is None:
            return  # gap — already logged, nothing to do
    else:
        log("[nk] no marker → using full RR (slow)")
        rr_nk = rr

    log(f"[nk] running on {len(rr_nk)} RR intervals")
    peaks_idx = np.concatenate([[0], np.round(np.cumsum(rr_nk)).astype(int)])
    nk.hrv(peaks_idx, sampling_rate=1000, show=True)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

def run(path, window_min, use_marker, use_full, no_gru, custom_marker=None):

    log("[init] discovering files")
    ecg_file    = find_file(path, "ECG")
    marker_file = find_file(path, "MARKER")

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

    SR = 130

    peaks, rr = compute_rr_from_ecg(ecg, SR)
    rr_ts = ts_ecg[peaks][1:]

    # ── ectopiques
    ectopics = detect_ectopics(ecg, peaks, rr, rr_ts, SR)
    plot_ectopics(rr_ts, rr, ectopics)

    # ── sliding HRV
    times, rmssd, hr = sliding_hrv(rr_ts, rr, window_min)
    log(f"[core] windows: {len(times)}")

    if len(hr) > 10:
        corr = np.corrcoef(hr, rmssd)[0, 1]
        log(f"[core] HR vs RMSSD corr: {corr:.2f}")

    # ── full night stats
    diff       = np.diff(rr)
    rmssd_full = np.sqrt(np.mean(diff**2))
    sdnn_full  = np.std(rr)
    hr_full    = 60000 / np.mean(rr)

    log("[core] FULL NIGHT:")
    log(f"  RMSSD: {rmssd_full:.1f} ms")
    log(f"  SDNN : {sdnn_full:.1f} ms")
    log(f"  HR   : {hr_full:.1f} bpm")

    # ── ECG segment plot
    plot_ecg_segment(ts_ecg, ecg, m_start, m_stop)

    # ── Poincaré
    log("[core] Poincaré plot")
    rr1 = rr[:-1]
    rr2 = rr[1:]

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

    log("\n[interpretation] Poincaré:")
    log(f"  SD1 (court terme): {sd1:.1f}")
    log(f"  SD2 (long terme):  {sd2:.1f}")
    log(f"  ratio SD1/SD2:     {ratio:.2f}")

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

    # ── sleep staging
    if no_gru:
        sleep_staging_rule_based(rr, rr_ts, times, rmssd, hr)
    else:
        sleep_staging(ecg, SR, ts_ecg, times, rmssd, hr)

    # ── neurokit full (optional)
    if use_full:
        run_neurokit(rr, rr_ts, m_start, m_stop)

# ─────────────────────────────────────────────────────────────

def sleep_staging_rule_based(rr, rr_ts, times, rmssd, hr):
    from collections import Counter
    from matplotlib.patches import Patch

    EPOCH_S  = 30
    epoch_td = timedelta(seconds=EPOCH_S)

    log("\n[sleep] rule-based sleep staging (30s epochs)")
    log(f"[sleep] {len(rr):,} RR intervals available")

    epochs_ts, labels = [], []
    t     = rr_ts[0]
    t_end = rr_ts[-1]

    while t + epoch_td <= t_end:
        mask = (rr_ts >= t) & (rr_ts < t + epoch_td)
        w = rr[mask]
        if len(w) < 5:
            t += epoch_td
            continue
        diff    = np.diff(w)
        rmssd_e = np.sqrt(np.mean(diff**2)) if len(diff) > 0 else 0
        hr_e    = 60000 / np.mean(w)
        if hr_e > 65 and rmssd_e < 25:
            label = "WAKE"
        elif rmssd_e > 70:
            label = "N3"
        elif rmssd_e > 45:
            label = "REM"
        elif rmssd_e > 25:
            label = "N2"
        else:
            label = "N1"
        epochs_ts.append(t)
        labels.append(label)
        t += epoch_td

    stage_labels_order = ["WAKE", "N1", "N2", "REM", "N3"]
    counts = Counter(labels)
    total  = len(labels)
    log(f"\n[sleep] {total} epochs × 30s = {total*30/3600:.1f}h")
    for stage in stage_labels_order:
        count = counts.get(stage, 0)
        log(f"  {stage:5s}: {count:4d} = {count*30/60:5.1f} min ({count/total*100:.1f}%)")

    stage_colors = {
        "WAKE": "#e74c3c",
        "REM":  "#9b59b6",
        "N1":   "#3498db",
        "N2":   "#2ecc71",
        "N3":   "#1a5276",
    }

    epochs_ts_np = np.array(epochs_ts)

    def get_stage_for_time(t):
        idx = np.searchsorted(epochs_ts_np, t, side="right") - 1
        idx = max(0, min(idx, len(labels) - 1))
        return labels[idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Sleep overview — RMSSD/HR + hypnogram (rule-based)")

    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i + 1]
        stage  = get_stage_for_time(t0)
        color  = stage_colors.get(stage, "#888888")
        ax1.fill_betweenx([0, max(rmssd) * 1.1], t0, t1, color=color, alpha=0.15)

    ax1.plot(times, rmssd, color="tab:blue", linewidth=1.2, label="RMSSD (ms)", zorder=3)
    ax1b = ax1.twinx()
    ax1b.plot(times, hr, color="tab:red", linewidth=1.0, alpha=0.7,
              linestyle="--", label="HR (bpm)", zorder=2)
    ax1.set_ylabel("RMSSD (ms)", color="tab:blue")
    ax1b.set_ylabel("HR (bpm)", color="tab:red")

    patches = [Patch(color=stage_colors[l], label=l) for l in stage_labels_order]
    ax1.legend(handles=patches + ax1.get_lines() + ax1b.get_lines(),
               loc="upper right", fontsize=8)

    stage_y = {"WAKE": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0}
    y_vals  = [stage_y[l] for l in labels]

    ax2.step(epochs_ts, y_vals, where="post", color="tab:blue", linewidth=0.8)
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(["N3", "N2", "N1", "REM", "WAKE"])
    ax2.set_ylabel("Stage")
    ax2.set_xlabel("Time")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def sleep_staging(ecg, sr, ts_ecg, times, rmssd, hr):
    import sleepecg
    import logging
    import warnings

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    log("\n[sleep] detecting heartbeats (Pan-Tompkins)…")
    beats = sleepecg.detect_heartbeats(ecg, sr)
    log(f"[sleep] beats detected: {len(beats):,}")

    beat_times = beats / sr

    record = sleepecg.SleepRecord(
        sleep_stage_duration=30,
        recording_start_time=ts_ecg[0],
        heartbeat_times=beat_times,
    )

    log("[sleep] loading classifier wrn-gru-mesa…")
    clf = sleepecg.load_classifier("wrn-gru-mesa", "SleepECG")
    log(f"[sleep] stages_mode: {clf.stages_mode}")

    log("[sleep] classifying stages…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stages_pred = sleepecg.stage(clf, record, return_mode="int")

    STAGE_MAPS = {
        "wake-rem-nrem":      {0: "WAKE", 1: "REM", 2: "NREM"},
        "wake-sleep":         {0: "WAKE", 1: "SLEEP"},
        "wake-rem-light-deep":{0: "WAKE", 1: "REM", 2: "LIGHT", 3: "DEEP"},
    }

    stage_map = STAGE_MAPS.get(clf.stages_mode)
    if stage_map is None:
        log(f"[sleep] unknown stages_mode '{clf.stages_mode}', using raw indices")
        stage_map = {i: str(i) for i in range(10)}

    stage_labels = list(stage_map.values())
    log(f"[sleep] stages mapped: {stage_labels}")

    from collections import Counter
    labels_str = [stage_map.get(int(s), "?") for s in stages_pred]
    counts = Counter(labels_str)
    total  = len(labels_str)
    log(f"\n[sleep] {total} epochs × 30s = {total * 30 / 3600:.1f}h")
    for label in stage_labels:
        count = counts.get(label, 0)
        log(f"  {label:6s}: {count:4d} = {count * 30 / 60:5.1f} min ({count / total * 100:.1f}%)")

    epoch_times = [ts_ecg[0] + timedelta(seconds=i * 30) for i in range(len(stages_pred))]

    stage_colors = {
        "WAKE":  "#e74c3c",
        "REM":   "#9b59b6",
        "NREM":  "#2980b9",
        "LIGHT": "#3498db",
        "DEEP":  "#1a5276",
        "SLEEP": "#27ae60",
        "N1":    "#5dade2",
        "N2":    "#2ecc71",
        "N3":    "#1a5276",
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Sleep overview — RMSSD/HR + hypnogram")
    epoch_times_np = np.array(epoch_times)

    def get_stage_for_time(t):
        idx = np.searchsorted(epoch_times_np, t, side="right") - 1
        idx = max(0, min(idx, len(labels_str) - 1))
        return labels_str[idx]

    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i + 1]
        stage  = get_stage_for_time(t0)
        color  = stage_colors.get(stage, "#aaaaaa")
        ax1.fill_betweenx([0, max(rmssd) * 1.1], t0, t1, color=color, alpha=0.2)

    ax1.plot(times, rmssd, color="white", linewidth=1.2, label="RMSSD (ms)", zorder=3)
    ax1b = ax1.twinx()
    ax1b.plot(times, hr, color="#f39c12", linewidth=1.0, alpha=0.9,
              linestyle="--", label="HR (bpm)", zorder=2)
    ax1.set_ylabel("RMSSD (ms)")
    ax1b.set_ylabel("HR (bpm)")

    from matplotlib.patches import Patch
    patches = [Patch(color=stage_colors.get(l, "#aaa"), label=l) for l in stage_labels]
    ax1.legend(handles=patches + ax1.get_lines() + ax1b.get_lines(),
               loc="upper right", fontsize=8)

    stage_y = {l: i for i, l in enumerate(reversed(stage_labels))}
    y_vals  = [stage_y.get(l, 0) for l in labels_str]

    ax2.step(epoch_times, y_vals, where="post", color="#3498db", linewidth=0.8)
    ax2.set_yticks(list(stage_y.values()))
    ax2.set_yticklabels(list(stage_y.keys()))

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

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

    p.add_argument("--path",    required=True, help="Folder containing ECG/ACC/MARKER files")
    p.add_argument("--window",  type=int, default=5, help="Sliding window (minutes)")
    p.add_argument("--no-marker",   action="store_true", help="Ignore marker file → analyze full recording")
    p.add_argument("--no-gru",      action="store_true", help="Use rule-based hypnogram, without TensorFlow/GRU")
    p.add_argument("--full",        action="store_true", help="Run NeuroKit full HRV analysis (slow)")
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
        use_full=args.full,
        no_gru=args.no_gru,
        custom_marker=custom_marker,
    )

if __name__ == "__main__":
    main()
