# Polar H10 HRV Analyzer

Analyze heart rate variability from raw ECG recordings (Polar H10 via Polar Sensor Logger).

![Python](https://img.shields.io/badge/python-3.10+-blue.svg) ![License](https://img.shields.io/badge/license-Unlicense-green.svg)

> ⚠️ **Domain disclaimer**: This tool was built with limited formal background in cardiology and sleep physiology. Analyses are grounded in established HRV literature and open-source libraries (NeuroKit2, SleepECG), but interpretations should be treated with appropriate caution. Contributions and corrections from people with domain expertise are strongly welcome.

---

## Project Structure

```
├── main.py           # Entry point, argparse, orchestration
├── utils.py          # Logger singleton, CSV reader, timestamp parser
├── constants.py      # All shared thresholds and sampling parameters
└── hrv/
    ├── core.py       # Signal processing, HRV metrics, mode runners, plots
    └── io.py         # File discovery and data loaders
```

---

## Features

- **Raw ECG ingestion** from Polar H10 exports (130 Hz, semicolon-delimited `.txt`, Polar Sensor Logger format)
- **R-peak detection and RR extraction** via NeuroKit2 (with artifact correction)
- **Physiological RR filter** — intervals outside [200, 3000] ms are discarded before any analysis
- **Three operational modes**: Night Recovery, Morning Readiness, Exercise Performance
- **Ectopic beat detection** — SVEB/VEB classification, couplets, triplets, runs (>3); excluded from all HRV metrics
- **Sliding HRV windows** (RMSSD, HR) with configurable window size; gap-safe (dropouts are skipped, not fatal)
- **HRV by hour** — RMSSD, HR, and ectopic count aggregated per hour
- **Spectral analysis** — LF/HF via Lomb-Scargle periodogram (no interpolation, variance-normalized); Welch on 4 Hz cubic spline resampled signal available as secondary method (Readiness mode)
- **Poincaré geometry** — SD1, SD2, SD1/SD2 ratio
- **Sleep staging** — rule-based (RMSSD/HR thresholds) or GRU/WaveNet model via SleepECG (`wrn-gru-mesa`)
- **Ectopics × sleep stage** — ectopic distribution per estimated stage logged after staging
- **Windowed HRV detail** (`--hrv-detail`) — RMSSD, SDNN, pNN50, DFA α1 on 5-min windows, aggregated by hour
- **Marker support** — analyze a specific window (file-based or manual `--custom-marker`)
- **Gap detection** — flags recording interruptions and warns when a marker falls in a gap
- **Log output** (`--output`) — all console output mirrored to a timestamped `.txt` file

---

## Installation

```bash
git clone https://github.com/darthjahus/tools-fitness.git
cd tools-fitness

pip install numpy matplotlib scipy neurokit2
# Optional (GRU sleep staging):
pip install sleepecg tensorflow
```

---

## Data Source

The script expects raw exports from the **Polar H10** chest strap recorded via **Polar Sensor Logger** (Android/iOS app).

### Expected File Structure

```
2026-04-21/
├── merged_ECG.txt       # Raw ECG signal (130 Hz, semicolon-delimited)
└── merged_MARKER.txt    # Optional: marker window (MARKER_START / MARKER_STOP)
```

File naming is flexible — the script searches for filenames containing `ECG` and `MARKER`. The ECG file is required; a missing MARKER file is silently ignored.

---

## Usage

### Night Recovery (Rule-Based Staging)
```bash
python main.py --path "Y:\Health\ECG\2026-04-21\" --mode night --no-gru
```
Fast, no TensorFlow required. Uses RMSSD/HR thresholds to classify 30s epochs.

### Night Recovery (GRU Model)
```bash
python main.py --path "Y:\Health\ECG\2026-04-21\" --mode night
```
Uses the `wrn-gru-mesa` classifier from SleepECG. Slower, requires TensorFlow.

### Morning Readiness (3–10 min resting window)
```bash
python main.py --path "Y:\Health\ECG\2026-04-21\" --mode readiness --custom-marker 2026-04-21T08:00:00 2026-04-21T08:05:00
```
`--custom-marker` is mandatory in readiness mode. Outputs full time-domain, frequency-domain (LF/HF), and Poincaré metrics for the specified window.

### Exercise Performance
```bash
python main.py --path "Y:\Health\ECG\2026-04-21\" --mode exercise
```
Segments the session into 10-minute blocks. Reports per-segment RMSSD, average HR, dominant HR zone, and ectopic frequency. Includes a global session summary with HR zone breakdown and Poincaré scatter.

### Disable Marker (Full Recording)
```bash
python main.py --path "Y:\Health\ECG\2026-04-21\" --no-marker
```

### Manual Marker Window
```bash
python main.py --path "Y:\Health\ECG\2026-04-21\" --custom-marker 2026-04-21T08:00:00 2026-04-21T08:05:00
```
In Night mode, also plots a 60s ECG segment centered on the window midpoint.

### Windowed HRV Detail
```bash
python main.py --path "Y:\Health\ECG\2026-04-21\" --hrv-detail --no-gru
```
Runs RMSSD, SDNN, pNN50, and DFA α1 on 5-minute sliding windows (1-minute stride), aggregated by hour and plotted as time series. Only meaningful in Night mode.

### Custom Sliding Window
```bash
python main.py --path "Y:\Health\ECG\2026-04-21\" --window 10 --no-gru
```
Default is 5 minutes. Larger windows smooth the RMSSD curve; smaller windows are more reactive.

### Save Log to File
```bash
python main.py --path "Y:\Health\ECG\2026-04-21\" --no-gru --output "Y:\Health\ECG\logs\"
```
All console output is mirrored to a timestamped `.txt` file (e.g. `hrv_night_20260421_223015.txt`).

---

## Command Line Options

| Option                                 | Description                                               | Default |
|----------------------------------------|-----------------------------------------------------------|---------|
| `--path DIR`                           | Folder containing ECG/MARKER files                        | *(required)* |
| `--mode`                               | `night` / `readiness` / `exercise`                        | `night` |
| `--window INT`                         | Sliding HRV window in minutes                             | `5` |
| `--no-marker`                          | Ignore marker file, analyze full recording                | off |
| `--no-gru`                             | Use rule-based staging instead of GRU model               | off |
| `--no-sleep`                           | Skip sleep staging entirely                               | off |
| `--hrv-detail`                         | 5-min windowed RMSSD/SDNN/pNN50/DFA α1, aggregated by hour | off |
| `--custom-marker START STOP\|DURATION` | Manual marker window. `END` is an ISO 8601 timestamp or a duration in minutes as integer | — |
| `--output DIR`                         | Directory for timestamped log file                        | — |

---

## Output

### Console (Night mode)
```
[init] discovering files for mode: NIGHT
[ecg] Y:\Health\ECG\2026-04-21\merged_ECG.txt
[core] cleaning ECG
[core] detecting R peaks
[core] peaks: 27988
[core] rr intervals: 27987
[core] 3 RR intervals outside [200, 3000] ms removed (artefacts)

[ectopic] ── ECTOPIC BEATS ──────────────────────────
  Total            : 137
  SVEB (supravent.): 98 (71.5%)
  VEB  (ventricular): 18 (13.1%)
  Grouped (run)    : 21 (15.3%)
  Couplets         : 4
  Triplets         : 1
  Runs (>3)        : 1
  Rate             : 18.6 /hour
──────────────────────────────────────────────────

[hrv] ── HRV BY HOUR ────────────────────────────────
    Hour    RMSSD       HR  Ectopics
   23:00     52.3     61.1         8
   00:00     61.8     59.4        12
   01:00     47.2     62.8        19
────────────────────────────────────────────────────────

[sleep] rule-based sleep staging (30s epochs)
[sleep] 882 epochs × 30s = 7.3h
  WAKE :   19 =   9.5 min (2.2%)
  N1   :    2 =   1.0 min (0.2%)
  N2   :  420 = 210.0 min (47.6%)
  REM  :  370 = 185.0 min (42.0%)
  N3   :   71 =  35.5 min (8.0%)

[ectopic] ectopic distribution by sleep stage: N1: 0  N2: 89  N3: 11  REM: 31  WAKE: 6

── OVERNIGHT RECOVERY METRICS ──
  RMSSD Average: 49.0 ms (Min Window: 31.2 | Max Window: 74.5)
  SDNN Global  : 136.5 ms
  HR Average   : 63.4 bpm
  pNN50        : 22.1 %
  pNN200       : 3.4 %
  Overnight LF/HF Ratio: 1.84

[summary]
Night Summary: Mean RMSSD 49.0ms (Range: 31.2–74.5ms) | Avg HR 63.4bpm
Autonomic Tone: LF/HF 1.84 | pNN50 22.1% | Global SDNN 136.5ms
Ectopics Profile: 137 anomaly intervals isolated across sleep timeline.
```

### Plots
- **Ectopic distribution** — RR tachogram with color-coded ectopics (SVEB / VEB / run >3) + hourly density histogram
- **ECG segment** — 60s raw signal centered on marker midpoint (if marker used)
- **Sleep overview** — dual-axis RMSSD/HR time series + color-coded hypnogram (2 panels)
- **HRV detail** — RMSSD / pNN50 / DFA α1 time series (if `--hrv-detail`)

---

## Understanding the Metrics

### RMSSD
Root mean square of successive RR differences. The primary short-term HRV metric, reflecting parasympathetic (vagal) activity. Higher = better recovery state. Always computed on ectopic-filtered intervals.

| Range | Interpretation |
|-------|----------------|
| > 70 ms | Excellent recovery |
| 50–70 ms | Good |
| 30–50 ms | Average |
| < 30 ms | Poor recovery / fatigue / stress |

Values decrease with age — a 36-year-old and a 53-year-old should not be compared against the same reference range.

### SDNN
Standard deviation of all RR intervals. Reflects total autonomic variability. Inflated on full-night recordings due to inter-cycle transitions — interpret with caution outside standardized 5-min windows.

### LF/HF Ratio
Power ratio between low-frequency (0.04–0.15 Hz) and high-frequency (0.15–0.40 Hz) spectral bands, computed via Lomb-Scargle periodogram directly on non-interpolated RR intervals (variance-normalized). Welch periodogram (4 Hz cubic spline, linear detrend post-resampling) available as secondary method.

| Range | Interpretation |
|-------|----------------|
| < 1.0 | Parasympathetic dominance (HF > LF) |
| < 2.0 | Balanced autonomic regulation |
| 2.0–4.0 | Mild sympathetic dominance / elevated strain |
| > 4.0 | Marked sympathetic dominance / acute autonomic stress |

Only computed in Readiness mode (short resting windows). Unreliable on full-night data due to non-stationarity.

### DFA α1
Detrended Fluctuation Analysis short-term scaling exponent. Computed only under `--hrv-detail` on 5-min windows.

| Range | Interpretation |
|-------|----------------|
| ≈ 1.0 | Healthy long-range correlations (typical during sleep) |
| > 1.5 | Loss of complexity — possible high sympathetic load or artefact |
| < 0.75 | Uncorrelated / noisy signal — check data quality |

### Poincaré (SD1 / SD2)
- **SD1**: Short-term beat-to-beat variability (correlated with RMSSD)
- **SD2**: Long-term variability / slow regulatory trends
- **SD1/SD2 ratio**: Higher = more short-term variability relative to long-term (good recovery); lower = autonomic rigidity or fragmented sleep

### HR vs RMSSD Correlation
A strong negative correlation (≈ −0.5 to −0.8) is physiologically normal: lower HR → higher vagal tone → higher RMSSD. A weak or positive correlation may indicate arrhythmia, noise, or non-restorative sleep.

### HR Zones

| Zone | Range | Context |
|------|-------|---------|
| Rest | < 101 bpm | Sleep / recovery |
| Zone 1 | 101–117 bpm | Recovery / aerobic threshold |
| Zone 2 | 118–132 bpm | Endurance |
| Zone 3 | 133–148 bpm | Aerobic capacity |
| Zone 4 | 149–161 bpm | Anaerobic threshold |
| Zone 5 | > 161 bpm | Max capacity |

---

## Ectopic Beat Detection

Ectopic beats are detected as RR intervals shorter than 80% of the local 30-beat sliding median. This approach tracks overnight HR drift rather than using a fixed global threshold.

Classification:

- **SVEB** (supraventricular) — compensatory pause < 1.8× local median: incomplete return cycle
- **VEB** (ventricular) — compensatory pause ≥ 1.8× local median: full compensatory pause
- **Grouped** — beat is part of a consecutive ectopic sequence (couplet, triplet, run); the compensatory criterion is not applicable since `rr[i+1]` is also ectopic

Beats belonging to runs of more than 3 consecutive ectopics are plotted in purple and listed separately.

All RMSSD, SDNN, and Poincaré computations exclude ectopic-flagged intervals.

---

## Sleep Staging

### Rule-Based (`--no-gru`)
Threshold classifier on 30s epochs (constants in `constants.py`):

| Condition | Stage |
|-----------|-------|
| HR > 65 bpm and RMSSD < 25 ms | WAKE |
| RMSSD > 70 ms | N3 (deep) |
| RMSSD > 45 ms | REM |
| RMSSD > 25 ms | N2 |
| else | N1 |

**Limitations**: RMSSD alone cannot reliably distinguish REM from light NREM — both can show similar vagal activity. The N3 threshold is particularly sensitive to individual baseline.

### GRU Model (default)
Uses the `wrn-gru-mesa` classifier from [SleepECG](https://github.com/cbrnr/sleepecg), trained on the MESA dataset. Outputs 3-class staging (`WAKE / REM / NREM`). More robust than rule-based but still single-lead ECG only — not equivalent to polysomnography.

**Known issue**: The model can collapse most of the night to REM on certain recordings. If the hypnogram looks implausible, fall back to `--no-gru`.

### Typical Sleep Architecture (Adults)

| Stage | Normal Range |
|-------|-------------|
| N3 (deep sleep) | 15–20% |
| REM | 20–25% |
| N2 | 45–55% |
| N1 | < 5% |
| WAKE | < 5% |

N3 is front-loaded (first half of the night). REM is back-loaded. Fragmented sleep typically reduces N3 and inflates apparent REM via rebound.

---

## Limitations and Known Issues

- **Single-lead ECG**: Adequate for rhythm analysis and HRV; not suitable for morphological diagnosis (ST changes, axis deviation, etc.)
- **Sleep staging accuracy**: ECG-only staging is fundamentally limited without EEG, EOG, and EMG. Results are indicative, not clinical.
- **Rule-based thresholds**: Empirical values not validated against polysomnography. Individual baselines vary significantly.
- **Ectopic classification**: The SVEB/VEB compensatory-pause criterion is unreliable on couplets and longer runs — those beats are classified as `grouped` and excluded from SVEB/VEB counts.
- **Demographic normalization**: Metrics like RMSSD should be compared against age/sex-matched norms. The script does not apply normalization.
- **Gap handling**: BLE dropouts are skipped in the sliding-window loop. Large gaps still produce discontinuities in RMSSD/HR curves; full-night statistics may be misleading if dropouts are frequent.
- **LF/HF reliability**: Meaningful only on short stationary windows (Readiness mode). On full-night data the non-stationarity of the signal makes spectral band interpretation unreliable.

---

## Contributing

Contributions are strongly encouraged, especially from people with backgrounds in:
- Cardiology / autonomic nervous system physiology
- Sleep medicine
- Signal processing (ECG artifact rejection, HRV methodologies)
- Machine learning (ECG-based staging beyond 3-class)

Please open issues or submit pull requests.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical core |
| `matplotlib` | All plots |
| `scipy` | Lomb-Scargle & Welch PSD, cubic spline interpolation, linear detrend |
| `neurokit2` | ECG cleaning, R-peak detection, windowed HRV |
| `sleepecg` | GRU-based sleep staging *(optional)* |
| `tensorflow` | Required by SleepECG GRU model *(optional)* |

---

## Related Tools

- [`fit_cardio.py`](fit_cardio.md) — Heart rate zone analysis from FIT files
- [`fit_cycling.py`](fit_cycling.md) — Cycling performance analysis
- [`stress.py`](stress.md) — Garmin stress score trends
