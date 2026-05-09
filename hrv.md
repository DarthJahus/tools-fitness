# `hrv.py` (HRV Sleep Analyzer)
Analyze heart rate variability and sleep from raw ECG recordings (Polar H10 via Polar Sensor Logger).

![HRV Analysis](https://img.shields.io/badge/HRV-Analysis-blue) ![Python](https://img.shields.io/badge/python-3.7+-blue.svg) ![License](https://img.shields.io/badge/license-Unlicense-green.svg)

> ⚠️ **Domain disclaimer**: This tool was built with limited knowledge in cardiology and sleep physiology. The analyses are based on established HRV literature and open-source libraries (NeuroKit2, SleepECG), but interpretations should be taken with caution. **Contributions, corrections, and reviews from people with domain expertise are strongly welcome.**

## Features
- 📡 **Raw ECG ingestion** from Polar H10 exports (130 Hz, semicolon-delimited `.txt`, Polar Sensor Logger format)
- ♥ **R-peak detection and RR extraction** via NeuroKit2 (with artifact correction)
- 🚫 **Physiological RR filter** — intervals outside [200, 3000] ms are discarded before any analysis
- 📊 **Sliding HRV windows** (RMSSD, HR) with configurable window size; gap-safe (dropouts are skipped, not fatal)
- 🕐 **HRV by hour** — RMSSD, HR, and ectopic count aggregated per hour over the full night
- 🔷 **Poincaré plot** with SD1/SD2 interpretation (ectopics excluded)
- 💓 **Ectopic beat detection** — SVEB/VEB classification, couplets, triplets, runs; run beats visually distinct in plot
- 💤 **Sleep staging** — two modes:
  - Rule-based hypnogram (RMSSD/HR thresholds, no dependencies)
  - GRU/WaveNet model via SleepECG (`wrn-gru-mesa`)
- 🗺️ **Ectopics × sleep stage** — count of ectopics per estimated stage logged after staging
- 📈 **Windowed HRV detail** (`--hrv-detail`) — RMSSD, SDNN, pNN50, DFA α1 on 5-min windows, aggregated by hour
- 📍 **Marker support** — analyze a specific window (file-based or manual)
- 🔍 **Gap detection** — flags recording interruptions and warns when a marker falls in a gap
- 💾 **Log output** (`--output`) — all console output mirrored to a timestamped `.txt` file

## Installation
```bash
git clone https://github.com/darthjahus/tools-fitness.git
cd tools-fitness

pip install numpy matplotlib neurokit2
# Optional (GRU sleep staging):
pip install sleepecg tensorflow
```

## Data Source
The script expects raw exports from the **Polar H10** chest strap, recorded via **Polar Sensor Logger** (not the Garmin app).

### Expected File Structure
```
2026-04-21/
├── merged_ECG.txt       # Raw ECG signal (130 Hz, semicolon-delimited)
└── merged_MARKER.txt    # Optional: marker window (MARKER_START / MARKER_STOP)
```

File naming is flexible — the script searches for filenames containing `ECG` and `MARKER`. The ECG file is required; a missing MARKER file is silently ignored.

## Usage

### Full Night Analysis (Rule-Based Staging)
```bash
python hrv.py --path "Y:\Santé\ECG\2026-04-21\" --no-gru
```
Fast, no TensorFlow required. Uses RMSSD/HR thresholds to classify 30s epochs.

### Full Night Analysis (GRU Model)
```bash
python hrv.py --path "Y:\Santé\ECG\2026-04-21\"
```
Uses the `wrn-gru-mesa` classifier from SleepECG. Slower, requires TensorFlow.

### Analyze a Specific Window (Marker File)
```bash
python hrv.py --path "Y:\Santé\ECG\2026-04-21\"
```
Automatically reads `MARKER_START` / `MARKER_STOP` from the marker file. If the marker falls inside a recording gap, this is detected and reported clearly.

### Disable Marker (Full Recording)
```bash
python hrv.py --path "Y:\Santé\ECG\2026-04-21\" --no-marker
```

### Manual Marker Window
```bash
python hrv.py --path "Y:\Santé\ECG\2026-04-21\" --custom-marker 2026-04-21T08:00:00 2026-04-21T08:05:00
```
Plots a 60s ECG segment centered on the window midpoint.

### Windowed HRV Detail
```bash
python hrv.py --path "Y:\Santé\ECG\2026-04-21\" --hrv-detail --no-gru
```
Runs RMSSD, SDNN, pNN50, and DFA α1 on 5-minute sliding windows (1-minute stride). Results are aggregated by hour in a console table and plotted as time series. Replaces the old `--full` flag, which called `nk.hrv(show=True)` on the full recording and was unusably slow on night-length data.

### Custom Sliding Window
```bash
python hrv.py --path "Y:\Santé\ECG\2026-04-21\" --window 10 --no-gru
```
Default is 5 minutes. Larger windows smooth the RMSSD curve, smaller windows are more reactive.

### Save Log to File
```bash
python hrv.py --path "Y:\Santé\ECG\2026-04-21\" --no-gru --output "Y:\Santé\ECG\logs\"
```
All console output is mirrored to a timestamped `.txt` file (e.g. `hrv_20260421_223015.txt`) in the specified directory.

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--path` | Folder containing ECG/MARKER files | *(required)* |
| `--window` | Sliding HRV window in minutes | `5` |
| `--no-marker` | Ignore marker file, analyze full recording | off |
| `--no-gru` | Use rule-based staging instead of GRU model | off |
| `--hrv-detail` | Windowed HRV detail: RMSSD/SDNN/pNN50/DFA α1 by 5-min windows, aggregated by hour | off |
| `--output DIR` | Directory for timestamped log file | — |
| `--custom-marker START STOP` | Manual marker window (ISO 8601 timestamps) | — |

## Output

### Console
```
[init] discovering files
[ecg] Y:\Santé\ECG\2026-04-21\merged_ECG.txt
[core] cleaning ECG
[core] detecting R peaks
[core] peaks: 27988
[core] rr intervals: 27987
[core] 3 RR intervals outside [200, 3000] ms removed (artefacts)
[core] windows: 439
[core] HR vs RMSSD corr: -0.47
[core] full-night metrics computed on 27850 RR (137 ectopics removed)
[core] FULL NIGHT:
  RMSSD: 49.0 ms
  SDNN : 136.5 ms
  HR   : 63.4 bpm

[ectopic] ── EXTRASYSTOLES ──────────────────────
  Total        : 137
  SVEB (supraven.) : 98 (71.5%)
  VEB  (ventric.)  : 18  (13.1%)
  Grouped (run) : 21 (15.3%)
  Couplets     : 4
  Triplets     : 1
  Runs (>3)    : 1
  Fréquence    : 18.6 /heure
────────────────────────────────────────────────

[hrv] ── HRV BY HOUR ────────────────────────────────
    Hour    RMSSD       HR  Ectopics
   23:00     52.3     61.1         8
   00:00     61.8     59.4        12
   01:00     47.2     62.8        19
   ...
────────────────────────────────────────────────────────

[interpretation] Poincaré:
  SD1 (court terme): 34.6
  SD2 (long terme):  190.0
  ratio SD1/SD2:     0.18
  → variabilité faible (activation / fatigue possible)

[sleep] rule-based sleep staging (30s epochs)
[sleep] 882 epochs × 30s = 7.3h
  WAKE :   19 =   9.5 min (2.2%)
  N1   :    2 =   1.0 min (0.2%)
  N2   :  420 = 210.0 min (47.6%)
  REM  :  370 = 185.0 min (42.0%)
  N3   :   71 =  35.5 min (8.0%)

[ectopic] distribution par stade : N1: 0  N2: 89  N3: 11  REM: 31  WAKE: 6
```

### Plots
- **Ectopic distribution** — RR tachogram with color-coded ectopics (SVEB / VEB / run >3) + hourly density histogram
- **Poincaré plot** — RR(n) vs RR(n+1) scatter, ectopics excluded
- **ECG segment** — 60s raw signal centered on marker midpoint (if marker used)
- **Sleep overview** — dual-axis RMSSD/HR + color-coded hypnogram (2 panels)
- **HRV detail** — RMSSD / pNN50 / DFA α1 time series (if `--hrv-detail`)

## Understanding the Metrics

### RMSSD
Root mean square of successive RR differences. The primary short-term HRV metric, reflecting parasympathetic (vagal) activity. Higher = better recovery state. Computed on ectopic-filtered RR intervals.

| Range | Interpretation |
|-------|----------------|
| > 70 ms | Excellent recovery |
| 50–70 ms | Good |
| 30–50 ms | Average |
| < 30 ms | Poor recovery / fatigue / stress |

Values decrease with age. A 36-year-old and a 53-year-old should not be compared to the same reference.

### SDNN
Standard deviation of all RR intervals. Reflects total autonomic variability. Inflated on full-night recordings due to inter-cycle transitions — interpret with caution outside of standardized 5-min windows.

### DFA α1
Detrended Fluctuation Analysis short-term scaling exponent. Computed only under `--hrv-detail` on 5-min windows.

- **α1 ≈ 1.0** — healthy long-range correlations (typical during sleep)
- **α1 > 1.5** — loss of complexity, possible high sympathetic load or artefact window
- **α1 < 0.75** — uncorrelated / noisy signal, check data quality

### Poincaré (SD1 / SD2)
- **SD1**: Short-term beat-to-beat variability (correlated with RMSSD)
- **SD2**: Long-term variability / slow regulatory trends
- **SD1/SD2 ratio**: Higher = more short-term variability relative to long-term (good recovery); lower = autonomic rigidity or fragmented sleep

### HR vs RMSSD Correlation
Logged at startup. A strong negative correlation (≈ −0.5 to −0.8) is physiologically normal: lower HR → higher vagal tone → higher RMSSD. A weak or positive correlation may indicate an arrhythmia, noise, or non-restorative sleep.

## Ectopic Beat Detection

Ectopic beats are detected as RR intervals shorter than 80% of the local 30-beat median. Classification:

- **SVEB** (supraventricular) — compensatory pause < 1.8× local median: incomplete return cycle
- **VEB** (ventricular) — compensatory pause ≥ 1.8× local median: full compensatory pause
- **Grouped** — beat is part of a consecutive ectopic sequence (couplet, triplet, run); the compensatory criterion is not applicable since `rr[i+1]` is also ectopic

Beats belonging to runs of more than 3 consecutive ectopics are plotted in **purple** and listed separately in the legend.

Full-night RMSSD, SDNN, and the Poincaré plot all exclude ectopic-flagged intervals.

## Sleep Staging

### Rule-Based (--no-gru)
Simple threshold classifier on 30s epochs:

| Condition | Stage |
|-----------|-------|
| HR > 65 bpm and RMSSD < 25 ms | WAKE |
| RMSSD > 70 ms | N3 (deep) |
| RMSSD > 45 ms | REM |
| RMSSD > 25 ms | N2 |
| else | N1 |

**Limitations**: RMSSD alone cannot reliably distinguish REM from light NREM — both can show similar vagal activity. This classifier is a rough approximation. The N3 threshold is particularly sensitive to individual baseline.

### GRU Model (default)
Uses the `wrn-gru-mesa` classifier from the [SleepECG](https://github.com/cbrnr/sleepecg) library, trained on the MESA dataset. Outputs 3-class staging (`WAKE / REM / NREM`). More robust than rule-based but still single-lead ECG only — not equivalent to polysomnography.

**Known issue**: The model can collapse most of the night to REM on certain recordings. If the hypnogram looks implausible, fall back to `--no-gru`.

### Ectopics × Sleep Stage
After staging (both modes), the count of ectopic beats per estimated stage is logged:
```
[ectopic] distribution par stade : N1: 0  N2: 89  N3: 11  REM: 31  WAKE: 6
```
A disproportionate count in WAKE or REM may reflect sympathetic surges or artefacts rather than true ectopic burden.

## Typical Sleep Architecture (Adults)
For reference:

| Stage | Normal Range |
|-------|-------------|
| N3 (deep sleep) | 15–20% |
| REM | 20–25% |
| N2 | 45–55% |
| N1 | < 5% |
| WAKE | < 5% |

N3 is front-loaded (first half of the night). REM is back-loaded (second half). Fragmented sleep (e.g., infant wake-ups) typically reduces N3 and inflates apparent REM via rebound.

## Limitations and Known Issues
- **Single lead ECG**: Adequate for rhythm analysis and HRV, not for morphological diagnosis (ST changes, axis deviation, etc.)
- **Sleep staging accuracy**: ECG-only staging is fundamentally limited without EEG, EOG, and EMG. Results are indicative, not clinical.
- **Rule-based thresholds**: Empirical values not validated against polysomnography. Individual baselines vary significantly.
- **Ectopic classification**: The SVEB/VEB compensatory-pause criterion is unreliable on couplets and longer runs — those beats are classified as `grouped` and excluded from SVEB/VEB counts.
- **Demographic normalization**: Metrics like RMSSD should be compared against age/sex-matched norms. The script does not currently apply normalization.
- **Gap handling**: BLE dropouts during recording are skipped in the sliding-window loop. Large gaps still produce discontinuities in the RMSSD/HR curves. Full-night statistics may be misleading if dropouts are frequent.

## Contributing
Contributions are strongly encouraged, especially from people with backgrounds in:
- Cardiology / autonomic nervous system physiology
- Sleep medicine
- Signal processing (ECG artifact rejection, HRV methodologies)
- Machine learning (ECG-based staging beyond 3-class)

Please open issues or submit pull requests.

## Dependencies

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical core |
| `matplotlib` | All plots |
| `neurokit2` | ECG cleaning, R-peak detection, windowed HRV |
| `sleepecg` | GRU-based sleep staging *(optional)* |
| `tensorflow` | Required by SleepECG GRU model *(optional)* |

## Related Tools
- [`fit_cardio.py`](fit_cardio.md) — Heart rate zone analysis from FIT files
- [`fit_cycling.py`](fit_cycling.md) — Cycling performance analysis
- [`stress.py`](stress.md) — Garmin stress score trends
