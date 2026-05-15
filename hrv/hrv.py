import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import neurokit2 as nk
from collections import Counter
from matplotlib.patches import Patch

ECG_SAMPLE_RATE = 130  # Hz — Polar H10 via Polar Sensor Logger


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
    rr_ts_np = np.array(rr_ts, dtype="datetime64[us]")
    m_start_np = np.datetime64(m_start, "us")
    m_stop_np  = np.datetime64(m_stop,  "us")
    mask = (rr_ts_np >= m_start_np) & (rr_ts_np <= m_stop_np)
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


def detect_ectopics(peaks, rr, rr_ts, window_beats=30, premature_ratio=0.80):
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

    # Build a set for O(1) membership tests
    ectopic_set = set(ectopic_indices.tolist())

    # Classification SVEB / VEB par pause compensatoire.
    # Les beats faisant partie d'une séquence consécutive (couplet, triplet, run)
    # n'ont pas de pause compensatoire lisible → classés "grouped".
    types = []
    for i in ectopic_indices:
        next_also_ectopic = (i + 1) in ectopic_set
        if next_also_ectopic:
            types.append("grouped")
        elif i + 1 < n:
            compensatory_sum = rr[i] + rr[i + 1]
            if compensatory_sum >= 1.8 * local_median[i]:
                types.append("VEB")
            else:
                types.append("SVEB")
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

    total    = len(ectopic_indices)
    n_sveb   = types.count("SVEB")
    n_veb    = types.count("VEB")
    n_grouped= types.count("grouped")

    log(f"\n[ectopic] ── EXTRASYSTOLES ──────────────────────")
    log(f"  Total        : {total}")
    log(f"  SVEB (supraven.) : {n_sveb} ({n_sveb/total*100:.1f}%)")
    log(f"  VEB  (ventric.)  : {n_veb}  ({n_veb/total*100:.1f}%)")
    log(f"  Grouped (run) : {n_grouped} ({n_grouped/total*100:.1f}%)")
    log(f"  Couplets     : {couplets}")
    log(f"  Triplets     : {triplets}")
    log(f"  Runs (>3)    : {runs}")

    if len(rr_ts) > 1:
        duration_h = (rr_ts[-1] - rr_ts[0]).total_seconds() / 3600
        if duration_h > 0:
            log(f"  Fréquence    : {total / duration_h:.1f} /heure")

    log(f"────────────────────────────────────────────────")

    timestamps = [rr_ts[i] for i in ectopic_indices if i < len(rr_ts)]
    return {
        "count": total, "sveb": n_sveb, "veb": n_veb, "grouped": n_grouped,
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

    # Identifier les indices appartenant à un run (>3 ectopiques consécutifs)
    indices_list = list(ectopics["indices"])
    ectopic_set_plot = set(indices_list)
    run_indices = set()
    i = 0
    while i < len(indices_list):
        run_len = 1
        while (i + run_len < len(indices_list) and
               indices_list[i + run_len] == indices_list[i + run_len - 1] + 1):
            run_len += 1
        if run_len > 3:
            for j in range(run_len):
                run_indices.add(indices_list[i + j])
        i += run_len

    colors = {"SVEB": "#e67e22", "VEB": "#e74c3c", "unknown": "#95a5a6", "grouped": "#95a5a6"}
    c = [
        "#8e44ad" if indices_list[k] in run_indices else colors.get(types[k], "#95a5a6")
        for k in range(len(indices_list))
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Distribution des extrasystoles")

    # Tachogramme RR + ectopiques marqués
    ax1.plot(rr_ts, rr, color="#2980b9", linewidth=0.6, alpha=0.7, label="RR (ms)")
    ax1.scatter(ts_ect, rr_ect, c=c, s=18, zorder=5, label="Ectopique")
    ax1.set_ylabel("RR (ms)")

    legend_patches = [
        Patch(color="#e67e22", label=f"SVEB ({ectopics['sveb']})"),
        Patch(color="#e74c3c", label=f"VEB ({ectopics['veb']})"),
        Patch(color="#8e44ad", label=f"Run >3 ({ectopics['runs']})"),
    ]
    ax1.legend(handles=legend_patches + ax1.get_lines()[:1], fontsize=8)

    # Densité horaire des ectopiques
    if len(ts_ect) > 1:
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
            # Insufficient beats in this window (gap / dropout).
            # Advance pos to the first timestamp beyond the window to skip the gap.
            next_pos = np.searchsorted(timestamps, end, side="right")
            if next_pos <= pos:
                next_pos = pos + 1  # safety: always advance at least one step
            pos = next_pos
            continue

        w = rr[idx]

        diff = np.diff(w)
        rmssd.append(np.sqrt(np.mean(diff**2)))
        hr.append(60000 / np.mean(w))
        times.append(start)

        target = start + step
        while pos < len(timestamps) and timestamps[pos] < target:
            pos += 1

    return times, rmssd, hr


def hrv_by_hour(rr, rr_ts, ectopics=None):
    """
    Aggregate RMSSD and HR by hour.
    Returns a list of dicts: {hour, rmssd, hr, ectopic_count}
    Logs the table.
    """
    if len(rr_ts) == 0:
        return []

    ect_ts_set = set()
    if ectopics and ectopics["count"] > 0:
        ect_ts_set = set(t.strftime("%Y-%m-%dT%H") for t in ectopics["timestamps"])

    rows = []
    first_h = rr_ts[0].replace(minute=0, second=0, microsecond=0)
    last_h  = rr_ts[-1].replace(minute=0, second=0, microsecond=0)
    h = first_h
    while h <= last_h:
        h_end = h + timedelta(hours=1)
        mask  = np.array([(t >= h) and (t < h_end) for t in rr_ts])
        w = rr[mask]
        if len(w) < 5:
            h += timedelta(hours=1)
            continue
        diff = np.diff(w)
        rmssd_h = float(np.sqrt(np.mean(diff**2))) if len(diff) else float("nan")
        hr_h    = float(60000 / np.mean(w))
        ect_count = sum(
            1 for t in (ectopics["timestamps"] if ectopics else [])
            if h <= t < h_end
        )
        rows.append({"hour": h.strftime("%H:00"), "rmssd": rmssd_h,
                     "hr": hr_h, "ectopic_count": ect_count})
        h += timedelta(hours=1)

    log("\n[hrv] ── HRV BY HOUR ────────────────────────────────")
    log(f"  {'Hour':>6}  {'RMSSD':>7}  {'HR':>7}  {'Ectopics':>9}")
    for r in rows:
        log(f"  {r['hour']:>6}  {r['rmssd']:>7.1f}  {r['hr']:>7.1f}  {r['ectopic_count']:>9}")
    log("────────────────────────────────────────────────────────")
    return rows

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


def run_hrv_detail(rr, rr_ts, m_start, m_stop):
    """
    Windowed HRV detail (5-min windows): RMSSD, SDNN, pNN50, DFA α1.
    Replaces the monolithic nk.hrv(show=True) which is unusable on long recordings.
    Results are aggregated by hour and plotted as DFA α1 evolution over the night.
    """
    log("[hrv-detail] windowed HRV analysis (5-min windows)")

    if m_start and m_stop:
        rr_nk = resolve_marker_rr(rr, rr_ts, m_start, m_stop)
        rr_ts_nk = np.array([t for t, v in zip(rr_ts, rr)
                              if t >= m_start and t <= m_stop and v in rr_nk])
        if rr_nk is None:
            return
    else:
        log("[hrv-detail] no marker → using full recording")
        rr_nk    = rr
        rr_ts_nk = np.array(rr_ts)

    WINDOW_S = 300   # 5 minutes
    STEP_S   = 60    # 1 minute stride
    SR_FAKE  = 1000  # ms-resolution virtual sampling rate for NeuroKit

    window_td = timedelta(seconds=WINDOW_S)
    step_td   = timedelta(seconds=STEP_S)

    results = []  # list of dicts per window
    pos = 0
    while pos < len(rr_ts_nk):
        t_start = rr_ts_nk[pos]
        t_end   = t_start + window_td
        mask    = (rr_ts_nk >= t_start) & (rr_ts_nk < t_end)
        w       = rr_nk[mask]
        if len(w) < 30:
            next_pos = np.searchsorted(rr_ts_nk, t_end, side="right")
            pos = max(pos + 1, next_pos)
            continue

        # Build fake peaks array from RR intervals (ms)
        peaks_idx = np.concatenate([[0], np.round(np.cumsum(w)).astype(int)])
        try:
            time_metrics     = nk.hrv_time(peaks_idx, sampling_rate=SR_FAKE, show=False)
            nonlinear_metrics= nk.hrv_nonlinear(peaks_idx, sampling_rate=SR_FAKE, show=False)
            rmssd = float(time_metrics.get("HRV_RMSSD", [np.nan])[0])
            sdnn  = float(time_metrics.get("HRV_SDNN",  [np.nan])[0])
            pnn50 = float(time_metrics.get("HRV_pNN50", [np.nan])[0])
            dfa1  = float(nonlinear_metrics.get("HRV_DFA_alpha1", [np.nan])[0])
            results.append({"t": t_start, "rmssd": rmssd, "sdnn": sdnn,
                             "pnn50": pnn50, "dfa1": dfa1})
        except Exception as e:
            log(f"[hrv-detail] window {t_start.strftime('%H:%M')} error: {e}")

        next_pos = np.searchsorted(rr_ts_nk, t_start + step_td, side="right")
        pos = max(pos + 1, next_pos)

    if not results:
        log("[hrv-detail] no valid windows computed")
        return

    # ── Aggregate by hour ────────────────────────────────────
    log("\n[hrv-detail] ── HRV DETAIL BY HOUR ─────────────────")
    log(f"  {'Hour':>6}  {'RMSSD':>7}  {'SDNN':>7}  {'pNN50':>7}  {'DFAα1':>7}")
    by_hour = {}
    for r in results:
        h = r["t"].strftime("%H:00")
        by_hour.setdefault(h, []).append(r)
    for h in sorted(by_hour):
        hrs = by_hour[h]
        def mean_or_nan(key):
            vals = [x[key] for x in hrs if not np.isnan(x[key])]
            return np.mean(vals) if vals else float("nan")
        log(f"  {h:>6}  {mean_or_nan('rmssd'):>7.1f}  {mean_or_nan('sdnn'):>7.1f}"
            f"  {mean_or_nan('pnn50'):>7.1f}  {mean_or_nan('dfa1'):>7.3f}")
    log("────────────────────────────────────────────────────────")

    # ── Plot DFA α1 over time ────────────────────────────────
    times_plot = [r["t"] for r in results]
    dfa1_plot  = [r["dfa1"] for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("HRV detail — 5-min windows")

    axes[0].plot(times_plot, [r["rmssd"] for r in results], color="tab:blue")
    axes[0].set_ylabel("RMSSD (ms)")

    axes[1].plot(times_plot, [r["pnn50"] for r in results], color="tab:green")
    axes[1].set_ylabel("pNN50 (%)")

    axes[2].plot(times_plot, dfa1_plot, color="tab:orange")
    axes[2].axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="DFA α1 = 1.0")
    axes[2].set_ylabel("DFA α1")
    axes[2].legend(fontsize=8)
    axes[2].set_xlabel("Time")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def ectopics_by_stage(ectopics, epochs_ts, labels):
    """
    Count how many ectopics fall in each sleep stage.
    epochs_ts: list of datetime, labels: list of str (same length).
    """
    if not ectopics or ectopics["count"] == 0:
        return
    epochs_np = np.array(epochs_ts)
    stage_counts = {}
    for ts in ectopics["timestamps"]:
        idx = np.searchsorted(epochs_np, ts, side="right") - 1
        idx = max(0, min(idx, len(labels) - 1))
        stage = labels[idx]
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    parts = "  ".join(f"{s}: {stage_counts.get(s, 0)}" for s in sorted(set(labels)))
    log(f"\n[ectopic] distribution par stade : {parts}")


def sleep_staging_rule_based(rr, rr_ts, times, rmssd, hr, ectopics=None):
    # ── Thresholds (empirical, not validated against PSG) ──────
    WAKE_HR_THRESHOLD    = 65   # bpm: above this → likely awake
    WAKE_RMSSD_THRESHOLD = 25   # ms:  below this + high HR → likely awake
    N3_RMSSD_THRESHOLD   = 70   # ms:  above → deep sleep (high parasympathetic)
    REM_RMSSD_THRESHOLD  = 45   # ms:  above → REM/light (moderate vagal, ambiguous)
    N2_RMSSD_THRESHOLD   = 25   # ms:  above → light sleep; else → N1
    # ──────────────────────────────────────────────────────────

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
        if hr_e > WAKE_HR_THRESHOLD and rmssd_e < WAKE_RMSSD_THRESHOLD:
            label = "WAKE"
        elif rmssd_e > N3_RMSSD_THRESHOLD:
            label = "N3"
        elif rmssd_e > REM_RMSSD_THRESHOLD:
            label = "REM"
        elif rmssd_e > N2_RMSSD_THRESHOLD:
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

    # F4: ectopics per sleep stage
    ectopics_by_stage(ectopics, epochs_ts, labels)


def sleep_staging(ecg, sr, ts_ecg, times, rmssd, hr, ectopics=None):
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

    # F4: ectopics per sleep stage
    ectopics_by_stage(ectopics, epoch_times, labels_str)

# ─────────────────────────────────────────────────────────────
