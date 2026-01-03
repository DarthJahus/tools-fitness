# `fit_cardio.py` (FIT Cardio Analyzer)
Analyze and visualize heart rate data from FIT files (Garmin, Wahoo, etc.) with automatic zone detection, peak analysis, and recovery patterns.

![Heart Rate Analysis](https://img.shields.io/badge/Sport-Analysis-blue) ![Python](https://img.shields.io/badge/python-3.7+-blue.svg) ![License](https://img.shields.io/badge/license-Unlicense-green.svg)

## Features
- 📊 **Heart rate zone visualization** with color-coded zones
- 🔍 **Peak detection** to identify maximum effort points
- 📈 **Slope analysis** to evaluate adaptation and recovery patterns
- 🎯 **Multiple zone calculation methods**:
  - Manual zones
  - Percentage-based (from max HR)
  - Karvonen/Heart Rate Reserve (most accurate)
  - Age-based (220 - age formula)

## Installation
```bash
# Clone the repository
git clone https://github.com/darthjahus/tools-fitness.git
cd tools-fitness

# Install dependencies
pip install fitdecode pandas matplotlib scipy
```

## Usage
### Basic Usage (Manual Zones)
```bash
python fit_cardio.py activity.fit
```
Uses default zones: `112, 124, 136, 149, 161`

### Custom Manual Zones
```bash
python fit_cardio.py activity.fit --zones 110,120,135,150,165
```

Zones structure:
- **Zone 0**: 0-110 (Warmup/Recovery)
- **Zone 1**: 111-120 (Light Endurance)
- **Zone 2**: 121-135 (Endurance)
- **Zone 3**: 136-150 (Aerobic)
- **Zone 4**: 151-165 (Threshold)
- **Zone 5**: 166+ (Maximum)

### From Maximum Heart Rate
```bash
python fit_cardio.py activity.fit --max-hr 185
```
Calculates zones using standard percentages: 60%, 70%, 80%, 90%, 100%

### Karvonen Method (Recommended)
```bash
python fit_cardio.py activity.fit --karvonen 60,185
```

Most accurate method using Heart Rate Reserve formula. Requires resting and max HR.

Formula: `Target HR = ((max_hr - resting_hr) × %intensity) + resting_hr`

### Age-Based Calculation

```bash
python fit_cardio.py activity.fit --zones-age 30
```

Estimates max HR using 220 - age formula, then calculates zones.

## Zone Calculation Methods

| Method | Command | Best For | Accuracy |
|--------|---------|----------|----------|
| **Manual** | `--zones 112,124,...` | Custom/coach-defined zones | ⭐⭐⭐⭐⭐ |
| **Karvonen/HRR** | `--karvonen 60,185` | Individual fitness level | ⭐⭐⭐⭐⭐ |
| **Max HR %** | `--max-hr 185` | General training | ⭐⭐⭐⭐ |
| **Age-based** | `--zones-age 30` | Quick estimates | ⭐⭐⭐ |

## Output

The script generates a visualization showing:
- 📍 **Heart rate over time** (colored by zone)
- 🎨 **Zone backgrounds** (color-coded)
- 🔻 **Peak markers** (black triangles)
- 📈 **Slope analysis**:
  - **Green lines**: Good adaptation (rising) or recovery (descending)
  - **Red lines**: Fatigue/stress (rising too fast)
  - **Orange lines**: Poor recovery (descending too slow)
  - **Grey lines**: Neutral/insignificant

## Understanding the Analysis
### Rising Slopes (Adaptation)
- **Green**: Controlled HR increase → Good cardiovascular adaptation
- **Red**: Very rapid HR increase → High stress or fatigue
- **Grey**: Minimal change

### Descending Slopes (Recovery)
- **Green**: Rapid HR decrease → Good recovery capacity
- **Orange**: Slow HR decrease → Poor recovery or accumulated fatigue
- **Grey**: Minimal change

## Determining Your Heart Rate Zones
### Find Your Resting Heart Rate
Measure first thing in the morning, while still in bed:
1. Count your pulse for 60 seconds
2. Do this for 3-5 consecutive days
3. Take the average

### Find Your Maximum Heart Rate
**Option 1**: Supervised lab test (most accurate)

**Option 2**: Field test
1. Warm up for 15 minutes
2. Run/bike hard for 3 minutes
3. Rest 2 minutes
4. Run/bike all-out for 2 minutes
5. The highest HR recorded is your max

**Option 3**: Formula (least accurate)
- Simple: `220 - age`
- Better: `208 - (0.7 × age)`

## Troubleshooting
### "Error: No heart rate data found in FIT file"
- Your FIT file doesn't contain heart rate records
- Make sure you were wearing a heart rate monitor during the activity

### "Error: No heart rate values above 90 BPM found"
- The activity doesn't appear to be a cardio workout
- The data might be corrupted or from a non-exercise activity

### "Warning: Resting HR of X seems unusual"
- Normal resting HR: 40-80 BPM
- Athletes can have lower values (35-45 BPM)
- Check if your measurement is correct

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
