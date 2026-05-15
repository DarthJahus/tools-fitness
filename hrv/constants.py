# ── Thresholds (empirical, not validated against PSG) ──────
WAKE_HR_THRESHOLD = 65  # bpm: above this → likely awake
WAKE_RMSSD_THRESHOLD = 25  # ms:  below this + high HR → likely awake
N3_RMSSD_THRESHOLD = 70  # ms:  above → deep sleep (high parasympathetic)
REM_RMSSD_THRESHOLD = 45  # ms:  above → REM/light (moderate vagal, ambiguous)
N2_RMSSD_THRESHOLD = 25  # ms:  above → light sleep; else → N1
#
EPOCH_S = 30
ECG_SAMPLE_RATE = 130  # Hz — Polar H10 via Polar Sensor Logger
RR_MIN_MS, RR_MAX_MS = 200, 3000
WINDOW_S = 300  # 5 minutes
STEP_S = 60  # 1 minute stride
SR_FAKE = 1000  # ms-resolution virtual sampling rate for NeuroKit
STAGE_MAPS = {
    "wake-rem-nrem": {0: "WAKE", 1: "REM", 2: "NREM"},
    "wake-sleep": {0: "WAKE", 1: "SLEEP"},
    "wake-rem-light-deep": {0: "WAKE", 1: "REM", 2: "LIGHT", 3: "DEEP"},
}
# ── Band definitions (Hz) ─────────────────────────────────────────────────────
VLF_BAND = (0.00, 0.04)
LF_BAND  = (0.04, 0.15)
HF_BAND  = (0.15, 0.40)
FS_RESAMPLE = 4.0          # Hz — Task Force 1996 standard

DEFAULT_ZONE_BOUNDARIES = [112, 124, 136, 149, 161]
