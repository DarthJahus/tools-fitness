# `stress.py` (Garmin Stress Analyzer)
Analyze and visualize stress data from Garmin Connect exports with moving averages, gap detection, and period comparison capabilities.

![Stress Analysis](https://img.shields.io/badge/Garmin-Analysis-blue) ![Python](https://img.shields.io/badge/python-3.7+-blue.svg) ![License](https://img.shields.io/badge/license-Unlicense-green.svg)

## Features
- 📊 **Comprehensive stress visualization** with separate awake/sleep/total metrics
- 📈 **Moving averages** to smooth out daily variations
- 🔍 **Automatic gap detection** to identify missing data periods
- 🔄 **Period comparison** to track stress evolution over time
- ⏮️ **Backward date ranges** using negative values (e.g., last 90 days)
- 📉 **Statistical analysis** with mean values and period differences

## Installation
```bash
# Clone the repository
git clone https://github.com/darthjahus/tools-fitness.git
cd tools-fitness

# Install dependencies
pip install pandas matplotlib
```

## Data Export from Garmin Connect

### Exporting Your Data
1. Log in to Garmin Connect
1. Go to [Garmin Account Data Management → Export Your Data](https://www.garmin.com/account/datamanagement/exportdata/)
1. Click on "REQUEST DATA EXPORT"
1. Wait for the e-mail from Garmin.
1. Download the complete data export
1. Extract the archive and locate the `DI_CONNECT/DI-Connect-Aggregator` folder (contains `UDSFile_*.json` files)

### Expected File Structure
```
DI_CONNECT/
└── DI-Connect-Aggregator/
    ├── UDSFile_2023-07-19_2023-10-27.json
    ├── UDSFile_2023-10-27_2024-02-04.json
    ├── UDSFile_2024-02-04_2024-05-14.json
    └── ...
```

## Usage

### Basic Usage (All Available Data)
```bash
python stress.py --source "./DI_CONNECT/DI-Connect-Aggregator"
```
Displays all available stress data with a 7-day moving average.

### Custom Moving Average
```bash
python stress.py --source "./path/to/data" --ma 14
```
Uses a 14-day moving average for smoother trends.

### Specific Period (Range Mode)
```bash
# Forward: 90 days starting from Jan 1, 2025
python stress.py --source "./data" --range 2025-01-01,90

# Backward: 90 days before Dec 31, 2024 (Oct 2 - Dec 30)
python stress.py --source "./data" --range 2024-12-31,-90
```

### Compare Two Periods
```bash
# Compare same dates in different years (forward)
python stress.py --source "./data" --compare 2024-01-01,2025-01-01,90

# Compare last 90 days of 2024 vs 2025 (backward)
python stress.py --source "./data" --compare 2024-12-31,2025-12-31,-90
```

### Filter Displayed Metrics
```bash
# Show only awake and sleep stress (hide average)
python stress.py --source "./data" --draw awake,sleep

# Show only average (total) stress
python stress.py --source "./data" --draw avg
```

## Command Line Options
| Option | Description | Example |
|--------|-------------|---------|
| `--source` | Path to folder containing `UDSFile_*.json` files | `--source ./data` |
| `--ma` | Moving average window in days (default: 7) | `--ma 14` |
| `--range` | Display specific period: START,LENGTH | `--range 2025-01-01,30` |
| `--compare` | Compare two periods: START1,START2,LENGTH | `--compare 2024-01-01,2025-01-01,90` |
| `--draw` | Metrics to display: all, awake, sleep, avg | `--draw awake,sleep` |

### Date Range Behavior
**Positive length**: Count forward from start date
- `--range 2025-01-01,90` → Jan 1 to Mar 31, 2025

**Negative length**: Count backward from reference date
- `--range 2025-04-01,-90` → Jan 1 to Mar 31, 2025

This is particularly useful for comparing "last X days" across different periods:
```bash
# Compare last 60 days of each year
python stress.py --source "./data" --compare 2024-12-31,2025-12-31,-60
```

## Understanding the Output
### Console Output
```
🔍 Lecture des fichiers...
📂 9 fichiers trouvés
✅ 873 jours de données chargés

📊 Extraction des données de stress...
✅ Données extraites: 2023-08-15 à 2026-01-03

⚠️  Trous détectés dans les données:
   - Du 2024-01-30 au 2024-02-08 (10 jours manquants)

📈 Application de la moyenne mobile sur 15 jours...

🔄 Mode comparaison activé

📅 Périodes comparées:
   Période 1: 2024-07-05 → 2024-12-31 (180 jours)
   Période 2: 2025-07-05 → 2025-12-31 (180 jours)

📉 Génération du graphique...

📊 Statistiques comparées:
   Stress moyen (total):
      Période 1: 38.0
      Période 2: 35.6 (-2.4)
   Stress moyen (éveillé):
      Période 1: 62.4
      Période 2: 52.2 (-10.2)
   Stress moyen (sommeil):
      Période 1: 14.1
      Période 2: 14.6 (+0.4)

✨ Graphique généré ! Affichage...
```

### Graph Interpretation
**X-axis formatting**:
- **≤90 days**: Week markers (7-day intervals)
- **>90 days**: Month markers (30-day intervals)

**Comparison mode**: When comparing periods, the X-axis shows "days since period start" (J0, J7, J14, etc.)
**Info box** (bottom right):
- Moving average window
- Period dates (in comparison mode)

### Stress Metrics
| Metric | Description | Typical Range |
|--------|-------------|---------------|
| **Stress endormi** (Sleep) | Average stress during sleep | 20-40 |
| **Stress éveillé** (Awake) | Average stress while awake | 30-70 |
| **Stress moyen** (Total) | 24-hour average stress | 25-60 |

**Color scheme**:
- 🔵 Blue: Sleep stress
- 🔴 Red: Awake stress  
- 🟢 Green: Total average stress

Lighter lines show raw daily values, bold lines show moving averages.

## Gap Detection
The script automatically detects missing data periods of **4+ days** and displays them in the console:

```
⚠️  Trous détectés dans les données:
   - Du 2024-05-14 au 2024-08-21 (100 jours manquants)
```

### Common Causes
- Missing export file from Garmin Connect
- Periods without wearing your device
- Periods where your device is in Power Saving mode
- Periods where your device is not measuring stress
- Data synchronization issues

### Resolution
If gaps appear, but you have data visible on Garmin Connect's website:
1. Try to export your data again
2. If the gap is still present, contact Garmin Support

## Use Cases
### Track Stress Evolution
Compare the same period across different years:
```bash
python stress.py --source "./data" --compare 2024-01-01,2025-01-01,365
```

### Analyze Recent Trends
View the last 30 days:
```bash
python stress.py --source "./data" --range 2025-01-03,-30
```

### Identify Patterns
Compare workdays vs vacation periods, before/after lifestyle changes, etc.

## Troubleshooting
### "❌ Aucun fichier UDSFile trouvé"
- Verify the `--source` path points to the correct folder
- Check that the folder contains files matching `UDSFile_*.json`
- Ensure you've extracted the Garmin Connect export archive

### "❌ Aucune donnée de stress trouvée"
- Your device may not record stress data
- The JSON files may be corrupted
- Try re-exporting from Garmin Connect

### "❌ Une ou plusieurs périodes n'ont pas de données"
- The specified date range is outside your available data
- Check the "Données extraites" line for your actual data range
- Verify you don't have a data gap during the requested period

### Large Data Gaps
If you find significant gaps (weeks/months), check:
1. Your source folder for missing files
2. File date ranges in filenames (e.g., `UDSFile_2024-05-14_2024-08-22.json`)
3. Contact Garmin Support if files are missing from your export

## Advanced Tips
### Optimal Moving Average Selection
- **7 days**: Good for weekly pattern detection
- **14 days**: Smooths out weekly variations
- **30 days**: Shows monthly trends, hides short-term fluctuations

### Comparison Best Practices
- Use same-length periods for fair comparison
- Consider seasonal effects when comparing different times of year
- Look for trend direction (±), not just absolute values

### Performance
- Large datasets (2+ years) may take a few seconds to load
- Consider using `--range` to focus on specific periods for faster rendering

## Contributing
Contributions welcome! Please submit pull requests or open issues for:
- Additional export format support
- New visualization types
- Advanced statistical analysis features
- Bug reports and feature requests

## Related Tools
- [`fit_cardio.py`](fit_cardio.md) - Heart rate analysis from FIT files
- [`fit_cycling.py`](fit_cycling.py) - Cycling performance analysis
