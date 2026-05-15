import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import neurokit2 as nk
from collections import Counter
from matplotlib.patches import Patch
from utils import log
from constants import (
    EPOCH_S,
    WAKE_HR_THRESHOLD,
    N3_RMSSD_THRESHOLD,
    REM_RMSSD_THRESHOLD,
    N2_RMSSD_THRESHOLD,
    WAKE_RMSSD_THRESHOLD, ECG_SAMPLE_RATE, WINDOW_S, STEP_S, SR_FAKE, STAGE_MAPS
)


def compute_time_metrics(rr_clean):
    """
    Computes standard time-domain metrics from clean NN intervals.
    Returns a dict containing core parameters.
    """
    if len(rr_clean) < 2:
        return {
            "rmssd": 0.0, "sdnn": 0.0, "mean_rr": 0.0,
            "pnn50": 0.0, "pnn200": 0.0, "brady_pct": 0.0, "tachy_pct": 0.0
        }

    diffs = np.diff(rr_clean)
    rmssd = np.sqrt(np.mean(diffs ** 2))
    sdnn = np.std(rr_clean)
    mean_rr = np.mean(rr_clean)

    # Calculate instant HR for every clean beat
    instant_hr = 60000.0 / rr_clean
    brady_beats = np.sum(instant_hr < 50.0)
    tachy_beats = np.sum(instant_hr > 150.0)

    total_clean_beats = len(rr_clean)
    brady_pct = (brady_beats / total_clean_beats) * 100.0
    tachy_pct = (tachy_beats / total_clean_beats) * 100.0

    # pNN50 and pNN200 metrics
    abs_diffs = np.abs(diffs)
    pnn50 = (np.sum(abs_diffs > 50.0) / len(diffs)) * 100.0 if len(diffs) > 0 else 0.0
    pnn200 = (np.sum(abs_diffs > 200.0) / len(diffs)) * 100.0 if len(diffs) > 0 else 0.0

    return {
        "rmssd": rmssd,
        "sdnn": sdnn,
        "mean_rr": mean_rr,
        "pnn50": pnn50,
        "pnn200": pnn200,
        "brady_pct": brady_pct,
        "tachy_pct": tachy_pct
    }


def compute_frequency_metrics(rr_clean):
    """
    Applies 4 Hz cubic spline interpolation and Welch periodogram spectral estimation.
    Bands: LF (0.04 - 0.15 Hz), HF (0.15 - 0.4 Hz).
    """
    from scipy.interpolate import CubicSpline
    from scipy.signal import welch

    if len(rr_clean) < 10:
        return {
            "lf_hf": 0.0, "lf_power": 0.0, "hf_power": 0.0,
            "lf_peak": 0.0, "hf_peak": 0.0, "total_power": 0.0
        }

    # Build continuous timeline from interval durations (seconds)
    rr_sec = rr_clean / 1000.0
    timeline = np.cumsum(rr_sec) - rr_sec[0]

    # Define uniform resampling lattice at 4 Hz
    fs_resample = 4.0
    t_uniform = np.arange(0, timeline[-1], 1.0 / fs_resample)

    if len(t_uniform) < 4:
        return {
            "lf_hf": 0.0, "lf_power": 0.0, "hf_power": 0.0,
            "lf_peak": 0.0, "hf_peak": 0.0, "total_power": 0.0
        }

    # De-trend signal by removing local mean
    rr_detrended = rr_sec - np.mean(rr_sec)

    cs = CubicSpline(timeline, rr_detrended)
    rr_uniform: np.ndarray = cs(t_uniform)

    # Run Welch method spectrum estimate (default nperseg logic balanced for short datasets)
    nperseg = min(256, len(rr_uniform))
    freqs: np.ndarray
    psd: np.ndarray
    freqs, psd = welch(rr_uniform, fs=fs_resample, nperseg=nperseg)

    # Define analytical frequency windows
    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
    total_mask = (freqs >= 0.00) & (freqs <= 0.40)

    # Resolution width for step integration (converting to ms^2)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0

    lf_power = np.sum(psd[lf_mask]) * df * 1000000.0
    hf_power = np.sum(psd[hf_mask]) * df * 1000000.0
    total_power = np.sum(psd[total_mask]) * df * 1000000.0

    lf_hf = lf_power / hf_power if hf_power > 0.0 else 0.0

    # Find peak power frequencies within bands
    lf_psd = psd[lf_mask]
    lf_freqs = freqs[lf_mask]
    lf_peak = lf_freqs[np.argmax(lf_psd)] if len(lf_psd) > 0 else 0.0

    hf_psd = psd[hf_mask]
    hf_freqs = freqs[hf_mask]
    hf_peak = hf_freqs[np.argmax(hf_psd)] if len(hf_psd) > 0 else 0.0

    return {
        "lf_hf": lf_hf,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_peak": lf_peak,
        "hf_peak": hf_peak,
        "total_power": total_power
    }


def compute_hr_zones(rr_intervals):
    """
    Buckets instant HR calculated from each RR interval into specific zones.
    Returns a dict containing cumulative duration (minutes) and percentage for each zone.
    """
    # Zone definitions (absolute boundaries in bpm)
    # Zone 1: 101–117 bpm, Zone 2: 118–132 bpm, Zone 3: 133–148 bpm
    # Zone 4: 149–161 bpm, Zone 5: >161 bpm, Rest: <101 bpm

    zones_stats = {
        "Rest": {"count": 0, "duration_ms": 0.0},
        "Zone 1": {"count": 0, "duration_ms": 0.0},
        "Zone 2": {"count": 0, "duration_ms": 0.0},
        "Zone 3": {"count": 0, "duration_ms": 0.0},
        "Zone 4": {"count": 0, "duration_ms": 0.0},
        "Zone 5": {"count": 0, "duration_ms": 0.0}
    }

    if len(rr_intervals) == 0:
        return {k: {"minutes": 0.0, "pct": 0.0} for k in zones_stats}

    instant_hrs = 60000.0 / rr_intervals

    for hr, rr_ms in zip(instant_hrs, rr_intervals):
        if hr < 101.0:
            key = "Rest"
        elif hr <= 117.0:
            key = "Zone 1"
        elif hr <= 132.0:
            key = "Zone 2"
        elif hr <= 148.0:
            key = "Zone 3"
        elif hr <= 161.0:
            key = "Zone 4"
        else:
            key = "Zone 5"

        zones_stats[key]["count"] += 1
        zones_stats[key]["duration_ms"] += rr_ms

    total_duration_ms = sum(z["duration_ms"] for z in zones_stats.values())
    results = {}

    for key, data in zones_stats.items():
        minutes = data["duration_ms"] / 60000.0
        pct = (data["duration_ms"] / total_duration_ms * 100.0) if total_duration_ms > 0 else 0.0
        results[key] = {"minutes": minutes, "pct": pct}

    return results


def run_mode_readiness(rr, rr_ts):
    """
    Executes short-term rest readiness assessment protocol (3–10 minutes window).
    """
    log("\n" + "─" * 60)
    log(" [MODE: READINESS MEASUREMENT]")
    log("─" * 60)

    # Clean data via ectopic removal loop
    # Re-detect or extract ectopics for safety over exact window context
    ectopics = detect_ectopics(rr, rr_ts, window_beats=20)

    ect_count = 0
    if ectopics and ectopics["count"] > 0:
        ect_idx = set(ectopics["indices"].tolist())
        clean_mask = np.array([i not in ect_idx for i in range(len(rr))])
        rr_clean = rr[clean_mask]
        ect_count = len(rr) - len(rr_clean)
        log(f"[WARNING] {ect_count} ectopic beat(s) detected and removed prior to HRV calculation.")
    else:
        rr_clean = rr
        log("[INFO] No ectopic intervals found. Processing native stream dataset.")

    if len(rr_clean) < 5:
        log("[ERROR] Insufficient stable RR intervals available for readiness processing.")
        return

    # Process metrics
    t_metrics = compute_time_metrics(rr_clean)
    f_metrics = compute_frequency_metrics(rr_clean)

    # Calculate geometric Poincaré statistics
    rr1 = rr_clean[:-1]
    rr2 = rr_clean[1:]
    sd1 = np.std((rr2 - rr1) / np.sqrt(2)) if len(rr1) > 1 else 0.0
    sd2 = np.std((rr2 + rr1) / np.sqrt(2)) if len(rr1) > 1 else 0.0
    sd_ratio = sd1 / sd2 if sd2 > 0.0 else 0.0

    ln_rmssd = np.log(t_metrics["rmssd"]) if t_metrics["rmssd"] > 0.0 else 0.0
    instant_hrs = 60000.0 / rr_clean
    hr_mean = np.mean(instant_hrs)
    hr_min = np.min(instant_hrs)
    hr_max = np.max(instant_hrs)

    # LF/HF Balanced autonomic ratio interpretation
    lf_hf_val = f_metrics["lf_hf"]
    if lf_hf_val < 2.0:
        interp = "Balanced autonomic regulation (Parasympathetic / Sympathetic equilibrium)"
    elif lf_hf_val <= 4.0:
        interp = "Mild sympathetic dominance / Elevated system strain"
    else:
        interp = "Marked sympathetic dominance / Acute autonomic stress"

    # Terminal breakdown dashboard presentation
    log("\n── TIME DOMAIN & METRICS ──")
    log(f"  Mean RR      : {t_metrics['mean_rr']:.1f} ms")
    log(f"  HR Mean      : {hr_mean:.1f} bpm (Min: {hr_min:.1f} | Max: {hr_max:.1f})")
    log(f"  RMSSD        : {t_metrics['rmssd']:.1f} ms")
    log(f"  LN(RMSSD)    : {ln_rmssd:.2f}")
    log(f"  SDNN         : {t_metrics['sdnn']:.1f} ms")
    log(f"  pNN50        : {t_metrics['pnn50']:.1f} %")
    log(f"  pNN200       : {t_metrics['pnn200']:.1f} %")

    log("\n── FREQUENCY DOMAIN (WELCH PSD) ──")
    log(f"  Total Power  : {f_metrics['total_power']:.1f} ms²")
    log(f"  LF Power     : {f_metrics['lf_power']:.1f} ms²")
    log(f"  HF Power     : {f_metrics['hf_power']:.1f} ms²")
    log(f"  LF Peak      : {f_metrics['lf_peak']:.3f} Hz")
    log(f"  HF Peak      : {f_metrics['hf_peak']:.3f} Hz")
    log(f"  LF/HF Ratio  : {lf_hf_val:.2f}")
    log(f"  Interpretation: {interp}")

    log("\n── POINCARÉ GEOMETRY ──")
    log(f"  SD1 (Short)  : {sd1:.1f} ms")
    log(f"  SD2 (Long)   : {sd2:.1f} ms")
    log(f"  SD1/SD2 Ratio: {sd_ratio:.2f}")

    # Section 4 Required rapid-read output layout block
    log("\n[summary]")
    log(f"Readiness Status: RMSSD {t_metrics['rmssd']:.1f}ms | HR {hr_mean:.1f}bpm | LF/HF {lf_hf_val:.2f}")
    log(f"Autonomic Balance: {interp}")
    log(f"Ectopics Count: {ect_count} removed before metrics analysis loops.")
    log("─" * 60)


def run_mode_exercise(rr, rr_ts, window_min=10):
    """
    Exercise performance session analysis over uniform time segments.
    Provides per-segment breakdowns and a global session summary.
    """
    log("\n" + "─" * 60)
    log(" [MODE: EXERCISE PERFORMANCE SESSION]")
    log("─" * 60)

    if len(rr_ts) < 2:
        log("[ERROR] Insufficient dataset intervals for activity analysis.")
        return

    # ── Global ectopic filtering ──────────────────────────────
    ectopics = detect_ectopics(rr, rr_ts, window_beats=30)
    ect_set = set(ectopics["indices"].tolist()) if (ectopics and ectopics["count"] > 0) else set()

    clean_mask  = np.array([i not in ect_set for i in range(len(rr))])
    rr_clean    = rr[clean_mask]
    rr_ts_clean = rr_ts[clean_mask]

    global_ect_count = len(rr) - len(rr_clean)
    global_ect_rate  = (global_ect_count / len(rr) * 100.0) if len(rr) > 0 else 0.0

    # ── Per-segment loop ──────────────────────────────────────
    seg_duration = timedelta(minutes=window_min)
    slice_start  = rr_ts_clean[0]
    end_time     = rr_ts_clean[-1]

    segment_metrics = []
    sliding_rmssds  = []

    log(f"\n  {'Segment Range':<20} | {'RMSSD':>8} | {'Avg HR':>7} | {'Ectopics':>8} | {'Primary Zone':<12}")
    log("  " + "─" * 68)

    while slice_start < end_time:
        slice_end = slice_start + seg_duration
        mask  = (rr_ts_clean >= slice_start) & (rr_ts_clean < slice_end)
        w_rr  = rr_clean[mask]

        if len(w_rr) >= 10:
            diffs    = np.diff(w_rr)
            w_rmssd  = np.sqrt(np.mean(diffs ** 2)) if len(diffs) > 0 else 0.0
            w_hr_avg = np.mean(60000.0 / w_rr)
            sliding_rmssds.append(w_rmssd)

            # Local ectopic count: beats present in raw window but absent after cleaning
            w_raw_count = int(np.sum((rr_ts >= slice_start) & (rr_ts < slice_end)))
            w_ect       = max(0, w_raw_count - len(w_rr))

            # Dominant zone for this segment
            seg_zones = compute_hr_zones(w_rr)
            seg_top   = max(seg_zones.items(), key=lambda x: x[1]["pct"])[0]

            label = f"{slice_start.strftime('%H:%M')}–{slice_end.strftime('%H:%M')}"
            log(f"  {label:<20} | {w_rmssd:>6.1f} ms | {w_hr_avg:>4.1f} bpm | {w_ect:>8d} | {seg_top:<12}")

            segment_metrics.append({
                "label": label, "rmssd": w_rmssd, "hr_avg": w_hr_avg, "ect": w_ect
            })

        slice_start = slice_end

    # ── Session global aggregates ─────────────────────────────
    if len(rr_clean) == 0:
        log("[ERROR] No clean intervals remaining after ectopic removal.")
        return

    global_hr_inst = 60000.0 / rr_clean
    global_hr_avg  = float(np.mean(global_hr_inst))
    global_hr_max  = float(np.max(global_hr_inst))

    rmssd_min = min(sliding_rmssds) if sliding_rmssds else 0.0
    rmssd_max = max(sliding_rmssds) if sliding_rmssds else 0.0

    tachy_beats = int(np.sum(global_hr_inst > 150.0))
    brady_beats = int(np.sum(global_hr_inst < 50.0))
    total_beats = len(rr_clean)
    tachy_pct   = (tachy_beats / total_beats * 100.0) if total_beats > 0 else 0.0
    brady_pct   = (brady_beats / total_beats * 100.0) if total_beats > 0 else 0.0

    global_zones = compute_hr_zones(rr_clean)

    # ── HR zone distribution ──────────────────────────────────
    log("\n── HEART RATE ZONE BREAKDOWN ──")
    for zone, stats in global_zones.items():
        bar = "█" * int(stats["pct"] / 5.0)
        log(f"  {zone:<8} : {stats['minutes']:>5.1f} min ({stats['pct']:>5.1f}%) {bar}")

    log("\n── EXERCISE METRICS SUMMARY ──")
    log(f"  HR Mean/Max   : {global_hr_avg:.1f} / {global_hr_max:.1f} bpm")
    log(f"  RMSSD Range   : {rmssd_min:.1f} ms ── {rmssd_max:.1f} ms")
    log(f"  Beats >150bpm : {tachy_beats} ({tachy_pct:.1f}%)")
    log(f"  Beats <50bpm  : {brady_beats} ({brady_pct:.1f}%)")
    log(f"  Total Ectopic : {global_ect_count} ({global_ect_rate:.2f}% of beats)")

    # ── Summary block ─────────────────────────────────────────
    top_zone     = max(global_zones.items(), key=lambda x: x[1]["pct"])[0]
    top_zone_pct = global_zones[top_zone]["pct"]

    log("\n[summary]")
    log(f"Workout Status: Avg HR {global_hr_avg:.1f}bpm | Peak {global_hr_max:.1f}bpm | RMSSD Max {rmssd_max:.1f}ms")
    log(f"Zone Dominance: {top_zone} ({top_zone_pct:.1f}%) — RMSSD Range {rmssd_min:.1f}–{rmssd_max:.1f}ms")
    log(f"Ectopics Profile: {global_ect_count} ({global_ect_rate:.2f}%) removed from timeline analysis.")
    log("─" * 60)


def run_mode_night(rr, rr_ts, ecg, ts_ecg, window_min=5, hrv_detail=False, no_sleep=False, no_gru=False, m_start=None, m_stop=None):
    """
    Executes standard overnight physiological sleep analysis protocol.
    Aggregates indicators by hour and tracks autonomic recovery.
    """
    log("\n" + "─" * 60)
    log(" [MODE: OVERNIGHT RECOVERY ANALYSIS]")
    log("─" * 60)

    if len(rr_ts) < 2:
        log("[ERROR] Insufficient dataset intervals for overnight parsing loops.")
        return

    # Filter out ectopics globally to create a pristine dataset for baseline tracking
    ectopics = detect_ectopics(rr, rr_ts, window_beats=30)
    ect_set = set(ectopics["indices"].tolist()) if (ectopics and ectopics["count"] > 0) else set()

    clean_mask = np.array([i not in ect_set for i in range(len(rr))])
    rr_clean = rr[clean_mask]
    rr_ts_clean = rr_ts[clean_mask]

    ect_count = len(rr) - len(rr_clean)
    if ect_count > 0:
        log(f"[INFO] Overnight metrics calculated on {len(rr_clean)} NN intervals ({ect_count} ectopics isolated).")
    else:
        log("[INFO] Processing clean baseline data stream.")

    # Core windowed and hourly statistical breakdowns
    times, rmssd, hr = sliding_hrv(rr_ts_clean, rr_clean, window_min)
    log(f"[core] Sliding windows processed: {len(times)}")

    # Hourly tables execution
    hrv_by_hour(rr_clean, rr_ts_clean, ectopics)

    if len(hr) > 10:
        corr = np.corrcoef(hr, rmssd)[0, 1]
        log(f"[core] Global HR vs RMSSD correlation factor: {corr:.2f}")

    # Calculate global indicators from clean intervals
    t_metrics = compute_time_metrics(rr_clean)
    hr_mean = 60000.0 / np.mean(rr_clean)

    # Track minimum and maximum RMSSD values from the sliding windows
    hrv_min = min(rmssd) if rmssd else t_metrics["rmssd"]
    hrv_max = max(rmssd) if rmssd else t_metrics["rmssd"]

    log("\n── OVERNIGHT RECOVERY METRICS ──")
    log(f"  RMSSD Average: {t_metrics['rmssd']:.1f} ms (Min Window: {hrv_min:.1f} | Max Window: {hrv_max:.1f})")
    log(f"  SDNN Global  : {t_metrics['sdnn']:.1f} ms")
    log(f"  HR Average   : {hr_mean:.1f} bpm")
    log(f"  pNN50        : {t_metrics['pnn50']:.1f} %")
    log(f"  pNN200       : {t_metrics['pnn200']:.1f} %")
    log(f"  Tachycardia  : {t_metrics['tachy_pct']:.1f} % of total session beats")
    log(f"  Bradycardia  : {t_metrics['brady_pct']:.1f} % of total session beats")

    # Spectral Analysis over the entire clean resting recording context
    f_metrics = compute_frequency_metrics(rr_clean)
    log(f"  Overnight LF/HF Ratio: {f_metrics['lf_hf']:.2f}")

    # Process heart rate zones mapping for the sleep timeline
    overnight_zones = compute_hr_zones(rr_clean)
    log("\n── SLEEP ZONE DISTRIBUTION ──")
    for zone, stats in overnight_zones.items():
        if stats["pct"] > 0.05:
            log(f"  {zone:<8} : {stats['minutes']:>5.1f} min ({stats['pct']:>5.1f}%)")

    # Segment plot rendering if explicit markers are passed down
    if m_start and m_stop:
        plot_ecg_segment(ts_ecg, ecg, m_start, m_stop)

    # Sleep classification execution path
    if not no_sleep:
        if no_gru:
            sleep_staging_rule_based(rr_clean, rr_ts_clean, times, rmssd, hr, ectopics)
        else:
            sleep_staging(ecg, ECG_SAMPLE_RATE, ts_ecg, times, rmssd, hr, ectopics)

    # Secondary window detailed analytics parsing
    if hrv_detail:
        run_hrv_detail(rr_clean, rr_ts_clean, m_start, m_stop)

    # Section 4 Required rapid-read output layout block
    log("\n[summary]")
    log(f"Night Summary: Mean RMSSD {t_metrics['rmssd']:.1f}ms (Range: {hrv_min:.1f}-{hrv_max:.1f}ms) | Avg HR {hr_mean:.1f}bpm")
    log(f"Autonomic Tone: LF/HF {f_metrics['lf_hf']:.2f} | pNN50 {t_metrics['pnn50']:.1f}% | Global SDNN {t_metrics['sdnn']:.1f}ms")
    log(f"Ectopics Profile: {ect_count} anomaly intervals isolated across sleep timeline.")
    log("─" * 60)


def plot_ectopics(rr_ts, rr, ectopics):
    if not ectopics or ectopics["count"] == 0:
        return

    indices = ectopics["indices"]
    types   = ectopics["types"]
    ts_ect  = ectopics["timestamps"]
    rr_ect  = rr[indices]

    indices_list = list(ectopics["indices"])
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
        "#8e44ad" if indices_list[k] in run_indices else colors.get(str(types[k]), "#95a5a6")
        for k in range(len(indices_list))
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Ectopic Beats Distribution & Density Timeline")

    ax1.plot(rr_ts, rr, color="#2980b9", linewidth=0.6, alpha=0.7, label="RR (ms)")
    ax1.scatter(ts_ect, rr_ect, c=c, s=18, zorder=5, label="Ectopic")
    ax1.set_ylabel("RR (ms)")

    legend_patches = [
        Patch(color="#e67e22", label=f"SVEB ({ectopics['sveb']})"),
        Patch(color="#e74c3c", label=f"VEB ({ectopics['veb']})"),
        Patch(color="#8e44ad", label=f"Run >3 ({ectopics['runs']})"),
    ]
    ax1.legend(handles=legend_patches + ax1.get_lines()[:1], fontsize=8)

    if len(ts_ect) > 1:
        ax2.hist(
            mdates.date2num(ts_ect),
            bins=max(1, int((rr_ts[-1] - rr_ts[0]).total_seconds() / 600)),
            color="#c0392b", alpha=0.7
        )
        ax2.xaxis_date()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.set_ylabel("Ectopics / 10 min")
        ax2.set_xlabel("Time")

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()



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


def detect_ectopics(rr, rr_ts, window_beats=30, premature_ratio=0.80):
    """
    Detect ectopic beats from RR intervals.
    SVEB/VEB classification via compensatory pause criterion.
    Couplet / triplet / run (>3) grouping.
    """
    n = len(rr)
    if n < window_beats * 2:
        log("[ectopic] insufficient RR intervals for ectopic detection")
        return None

    # Sliding local median — tracks overnight HR drift
    half_w = window_beats // 2
    local_median = np.array([
        np.median(rr[max(0, i - half_w):min(n, i + half_w)])
        for i in range(n)
    ])

    # Premature beat: RR shorter than threshold relative to local baseline
    ectopic_mask = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if rr[i] < premature_ratio * local_median[i]:
            ectopic_mask[i] = True

    ectopic_indices = np.where(ectopic_mask)[0]

    if len(ectopic_indices) == 0:
        log("[ectopic] no ectopic beats detected")
        return {"count": 0, "sveb": 0, "veb": 0,
                "couplets": 0, "triplets": 0, "runs": 0,
                "indices": np.array([], dtype=int), "types": [], "timestamps": []}

    ectopic_set = set(ectopic_indices.tolist())

    # SVEB / VEB classification via compensatory pause.
    # Beats within a consecutive run lack a readable compensatory pause → "grouped".
    types = []
    for i in ectopic_indices:
        if (i + 1) in ectopic_set:
            types.append("grouped")
        elif i + 1 < n:
            compensatory_sum = rr[i] + rr[i + 1]
            types.append("VEB" if compensatory_sum >= 1.8 * local_median[i] else "SVEB")
        else:
            types.append("unknown")

    # Couplet / triplet / run counting
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
            run_end   = rr_ts[min(int(ectopic_indices[i + run_len - 1]), len(rr_ts) - 1)]
            run_dur_s = (run_end - run_start).total_seconds()
            run_bpm   = 60000 / np.mean(rr[ectopic_indices[i]:ectopic_indices[i] + run_len])
            log(f"  Run @{run_start.strftime('%H:%M:%S')} — {run_len} beats — {run_bpm:.0f} bpm — {run_dur_s:.1f}s")
        i += run_len

    total    = len(ectopic_indices)
    n_sveb   = types.count("SVEB")
    n_veb    = types.count("VEB")
    n_grouped = types.count("grouped")

    log(f"\n[ectopic] ── ECTOPIC BEATS ──────────────────────────")
    log(f"  Total            : {total}")
    log(f"  SVEB (supravent.): {n_sveb} ({n_sveb / total * 100:.1f}%)")
    log(f"  VEB  (ventricular): {n_veb} ({n_veb / total * 100:.1f}%)")
    log(f"  Grouped (run)    : {n_grouped} ({n_grouped / total * 100:.1f}%)")
    log(f"  Couplets         : {couplets}")
    log(f"  Triplets         : {triplets}")
    log(f"  Runs (>3)        : {runs}")

    if len(rr_ts) > 1:
        duration_h = (rr_ts[-1] - rr_ts[0]).total_seconds() / 3600
        if duration_h > 0:
            log(f"  Rate             : {total / duration_h:.1f} /hour")

    log("─" * 50)

    timestamps = [rr_ts[i] for i in ectopic_indices if i < len(rr_ts)]
    return {
        "count": total, "sveb": n_sveb, "veb": n_veb, "grouped": n_grouped,
        "couplets": couplets, "triplets": triplets, "runs": runs,
        "indices": ectopic_indices, "types": types, "timestamps": timestamps,
    }


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

    rows = []
    first_h = rr_ts[0].replace(minute=0, second=0, microsecond=0)
    last_h  = rr_ts[-1].replace(minute=0, second=0, microsecond=0)
    h = first_h
    while h <= last_h:
        h_end = h + timedelta(hours=1)
        mask = (rr_ts >= h) & (rr_ts < h_end)
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
    Windowed HRV detail (5-min windows, 1-min stride): RMSSD, SDNN, pNN50, DFA α1.
    Aggregated by hour and plotted over the recording timeline.
    """
    log("[hrv-detail] windowed HRV analysis (5-min windows)")

    if m_start and m_stop:
        mask = (rr_ts >= m_start) & (rr_ts <= m_stop)
        rr_nk    = rr[mask]
        rr_ts_nk = rr_ts[mask]
        if len(rr_nk) == 0:
            resolve_marker_rr(rr, rr_ts, m_start, m_stop)  # logs gap diagnostics
            return
    else:
        log("[hrv-detail] no marker → using full recording")
        rr_nk    = rr
        rr_ts_nk = rr_ts

    window_td = timedelta(seconds=WINDOW_S)
    step_td   = timedelta(seconds=STEP_S)

    results = []
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

        # Build synthetic peaks array from RR intervals (ms)
        peaks_idx = np.concatenate([[0], np.round(np.cumsum(w)).astype(int)])
        try:
            # noinspection PyTypeChecker
            time_metrics = nk.hrv_time(peaks_idx, sampling_rate=SR_FAKE, show=False)
            # noinspection PyTypeChecker
            nonlinear_metrics = nk.hrv_nonlinear(peaks_idx, sampling_rate=SR_FAKE, show=False)
            rmssd = float(time_metrics.get("HRV_RMSSD",     [np.nan])[0])
            sdnn  = float(time_metrics.get("HRV_SDNN",      [np.nan])[0])
            pnn50 = float(time_metrics.get("HRV_pNN50",     [np.nan])[0])
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

    by_hour: dict = {}
    for r in results:
        by_hour.setdefault(r["t"].strftime("%H:00"), []).append(r)

    def _mean_or_nan(rows: list, key: str) -> float:
        vals = [x[key] for x in rows if not np.isnan(x[key])]
        return float(np.mean(vals)) if vals else float("nan")

    for h in sorted(by_hour):
        hrs = by_hour[h]
        log(f"  {h:>6}  {_mean_or_nan(hrs, 'rmssd'):>7.1f}  {_mean_or_nan(hrs, 'sdnn'):>7.1f}"
            f"  {_mean_or_nan(hrs, 'pnn50'):>7.1f}  {_mean_or_nan(hrs, 'dfa1'):>7.3f}")
    log("─" * 60)

    # ── Plot ─────────────────────────────────────────────────
    times_plot = [r["t"] for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("HRV detail — 5-min windows")

    axes[0].plot(times_plot, [r["rmssd"] for r in results], color="tab:blue")
    axes[0].set_ylabel("RMSSD (ms)")

    axes[1].plot(times_plot, [r["pnn50"] for r in results], color="tab:green")
    axes[1].set_ylabel("pNN50 (%)")

    axes[2].plot(times_plot, [r["dfa1"] for r in results], color="tab:orange")
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
        idx = max(0, min(int(idx), len(labels) - 1))
        stage = labels[idx]
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    parts = "  ".join(f"{s}: {stage_counts.get(s, 0)}" for s in sorted(set(labels)))
    log(f"\n[ectopic] ectopic distribution by sleep stage: {parts}")


def sleep_staging_rule_based(rr, rr_ts, times, rmssd, hr, ectopics=None):
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

    def get_stage_for_time(_t):
        idx = np.searchsorted(epochs_ts_np, _t, side="right") - 1
        idx = max(0, min(int(idx), len(labels) - 1))
        return labels[idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
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
        idx = max(0, min(int(idx), len(labels_str) - 1))
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
