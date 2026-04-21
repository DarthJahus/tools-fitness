# `hrv.py` (HRV Sleep Analyzer)
Analyze heart rate variability and sleep from raw ECG recordings (Polar H10 via Garmin export).

![HRV Analysis](https://img.shields.io/badge/HRV-Analysis-blue) ![Python](https://img.shields.io/badge/python-3.7+-blue.svg) ![License](https://img.shields.io/badge/license-Unlicense-green.svg)

> ⚠️ **Domain disclaimer**: This tool was built with limited knowledge in cardiology and sleep physiology. The analyses are based on established HRV literature and open-source libraries (NeuroKit2, SleepECG), but interpretations should be taken with caution. **Contributions, corrections, and reviews from people with domain expertise are strongly welcome.**

## Features
- 📡 **Raw ECG ingestion** from Polar H10 exports (130 Hz, semicolon-delimited `.txt`)
- ♥ **R-peak detection and RR extraction** via NeuroKit2 (with artifact correction)
- 📊 **Sliding HRV windows** (RMSSD, HR) with configurable window size
- 🔷 **Poincaré plot** with SD1/SD2 interpretation
- 💤 **Sleep staging** — two modes:
  - Rule-based hypnogram (RMSSD/HR thresholds, no dependencies)
  - GRU/WaveNet model via SleepECG (`wrn-gru-mesa`)
- 📈 **Full NeuroKit2 HRV analysis** (time-domain, frequency-domain, nonlinear)
- 📍 **Marker support** — analyze a specific window (file-based or manual)
- 🔍 **Gap detection** — flags recording interruptions and warns when a marker falls in a gap

## Installation
```bash
git clone https://github.com/darthjahus/tools-fitness.git
cd tools-fitness

pip install numpy matplotlib neurokit2
# Optional (GRU sleep staging):
pip install sleepecg tensorflow
```

## Data Source
The script expects raw exports from the **Polar H10** chest strap, recorded via the Garmin app or compatible tools.

### Expected File Structure
```
2026-04-21/
├── merged_ECG.txt       # Raw ECG signal (130 Hz)
└── merged_MARKER.txt    # Optional: marker window (MARKER_START / MARKER_STOP)
```

File naming is flexible — the script searches for filenames containing `ECG` and `MARKER`.

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
Automatically reads `MARKER_START` / `MARKER_STOP` from the marker file and runs NeuroKit2 HRV on that window.

### Disable Marker (Full Recording)
```bash
python hrv.py --path "Y:\Santé\ECG\2026-04-21\" --no-marker
```

### Manual Marker Window
```bash
python hrv.py --path "Y:\Santé\ECG\2026-04-21\" --custom-marker 2026-04-21T08:00:00 2026-04-21T08:05:00
```
Plots a 60s ECG segment centered on the window midpoint.

### Full NeuroKit2 HRV Analysis
```bash
python hrv.py --path "Y:\Santé\ECG\2026-04-21\" --full
```
Runs the complete NeuroKit2 `hrv()` function on the marker window (or full recording if no marker). Produces time-domain, frequency-domain, and nonlinear metrics.

### Custom Sliding Window
```bash
python hrv.py --path "Y:\Santé\ECG\2026-04-21\" --window 10 --no-gru
```
Default is 5 minutes. Larger windows smooth the RMSSD curve, smaller windows are more reactive.

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--path` | Folder containing ECG/MARKER files | *(required)* |
| `--window` | Sliding HRV window in minutes | `5` |
| `--no-marker` | Ignore marker file, analyze full recording | off |
| `--no-gru` | Use rule-based staging instead of GRU model | off |
| `--full` | Run full NeuroKit2 HRV analysis on marker window | off |
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
[core] windows: 439
[core] HR vs RMSSD corr: -0.47
[core] FULL NIGHT:
  RMSSD: 49.0 ms
  SDNN : 136.5 ms
  HR   : 63.4 bpm

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
```

### Plots
- **Poincaré plot** — RR(n) vs RR(n+1) scatter, full night
- **ECG segment** — 60s raw signal centered on marker midpoint (if marker used)
- **Sleep overview** — dual-axis RMSSD/HR + color-coded hypnogram (2 panels)
- **NeuroKit2 dashboard** — full HRV report (if `--full`)

## Understanding the Metrics

### RMSSD
Root mean square of successive RR differences. The primary short-term HRV metric, reflecting parasympathetic (vagal) activity. Higher = better recovery state.

| Range | Interpretation |
|-------|----------------|
| > 70 ms | Excellent recovery |
| 50–70 ms | Good |
| 30–50 ms | Average |
| < 30 ms | Poor recovery / fatigue / stress |

Values decrease with age. A 36-year-old and a 53-year-old should not be compared to the same reference.

### SDNN
Standard deviation of all RR intervals. Reflects total autonomic variability. Inflated on full-night recordings due to inter-cycle transitions — interpret with caution outside of standardized 5-min windows.

### Poincaré (SD1 / SD2)
- **SD1**: Short-term beat-to-beat variability (correlated with RMSSD)
- **SD2**: Long-term variability / slow regulatory trends
- **SD1/SD2 ratio**: Higher = more short-term variability relative to long-term (good recovery); lower = autonomic rigidity or fragmented sleep

### HR vs RMSSD Correlation
Logged at startup. A strong negative correlation (≈ −0.5 to −0.8) is physiologically normal: lower HR → higher vagal tone → higher RMSSD. A weak or positive correlation may indicate an arrhythmia, noise, or non-restorative sleep.

## Sleep Staging

### Rule-Based (--no-gru)
Simple threshold classifier on 30s epochs:

| Condition | Stage |
|-----------|-------|
| HR > 65 and RMSSD < 25 | WAKE |
| RMSSD > 70 | N3 (deep) |
| RMSSD > 45 | REM |
| RMSSD > 25 | N2 |
| else | N1 |

**Limitations**: RMSSD alone cannot reliably distinguish REM from light NREM — both can show similar vagal activity. This classifier is a rough approximation. The N3 threshold is particularly sensitive to individual baseline.

### GRU Model (default)
Uses the `wrn-gru-mesa` classifier from the [SleepECG](https://github.com/cbrnr/sleepecg) library, trained on the MESA dataset. Outputs 3-class staging (`WAKE / REM / NREM`). More robust than rule-based but still single-lead ECG only — not equivalent to polysomnography.

**Known issue**: The model can collapse most of the night to REM on certain recordings. If the hypnogram looks implausible, fall back to `--no-gru`.

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
- **Rule-based thresholds**: Hardcoded values not validated against polysomnography. Individual baselines vary significantly.
- **Demographic normalization**: Metrics like RMSSD should be compared against age/sex-matched norms. The script does not currently apply normalization.
- **Gap handling**: Recordings with large gaps (sensor dropout) may produce misleading full-night statistics.

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
| `neurokit2` | ECG cleaning, R-peak detection, full HRV |
| `sleepecg` | GRU-based sleep staging *(optional)* |
| `tensorflow` | Required by SleepECG GRU model *(optional)* |

## Related Tools
- [`fit_cardio.py`](fit_cardio.md) — Heart rate zone analysis from FIT files
- [`fit_cycling.py`](fit_cycling.md) — Cycling performance analysis
- [`stress.py`](stress.md) — Garmin stress score trends
