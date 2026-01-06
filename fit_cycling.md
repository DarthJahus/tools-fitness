# `fit_cycling.py` (FIT Cycling Analyzer)

Cycling data analyzer for FIT files with GPS tracking, effort calculation, performance metrics, and visualizations.

![Cycling Analysis](https://img.shields.io/badge/Sport-Cycling-blue) ![Python](https://img.shields.io/badge/python-3.7+-blue.svg) ![License](https://img.shields.io/badge/license-Unlicense-green.svg)

## Features
- 🗺️ **GPS route visualization** with slope and HR zone coloring
- 📊 **Multi-metric analysis**: speed, slope, heart rate, effort
- 🎯 **Multiple effort calculation methods**:
  - **TSS** (Training Stress Score) - requires power meter
  - **hrTSS** (Heart Rate TSS) - most accurate without power data
  - **Custom** - combines HR, slope and speed
- 🔄 **Circuit detection** with multiple algorithms
- 📈 **Local extrema detection** (peaks and valleys)
- 🔗 **Speed-HR correlation analysis** with KDE and regression
- 🌍 **3D GPS trajectory** with time and effort visualization
- 🎨 **HR zone-based coloring** throughout all visualizations

## Installation
```bash
# Install dependencies
pip install fitdecode pandas numpy matplotlib seaborn scipy

# Clone or download the scripts
# Ensure both fit_cycling.py and gps_helper.py are in the same directory
```

## Quick Start
```bash
# Basic analysis
python fit_cycling.py ride.fit

# Out-and-back route with heart rate zones
python fit_cycling.py climb.fit --circuit --karvonen 60,185

# Indoor training session
python fit_cycling.py zwift.fit --no-gps --plots metrics
```

---

## Heart Rate Zones

Configure HR zones using the same methods as `fit_cardio.py`:

### Manual Zones (Default)
```bash
python fit_cycling.py ride.fit --zones 112,124,136,149,161
```
Provide 5 boundaries separating zones 0-5.

### Percentage Method
```bash
python fit_cycling.py ride.fit --max-hr 185
```
Calculates zones as: 60%, 70%, 80%, 90%, 100% of max HR.

### Karvonen/HRR Method (Recommended)
```bash
python fit_cycling.py ride.fit --karvonen 60,185
```
Most accurate method using Heart Rate Reserve:
```
Zone HR = (HRmax - HRrest) × %intensity + HRrest
```

### Age-Based Estimation
```bash
python fit_cycling.py ride.fit --zones-age 30
```
Uses `220 - age` formula to estimate max HR.

**💡 Tip**: For best results, use Karvonen method with measured resting HR (morning) and max HR (field test).

---

## Effort Calculation Methods

### 1. TSS (Training Stress Score)

**Best for**: Cyclists with power meters

**How it works**: 
- Based on Coggan's Training Stress Score
- Uses Normalized Power (NP) and Functional Threshold Power (FTP)
- Accounts for intensity variability

**Formula**:
```
NP = (30s rolling avg power^4)^0.25
IF = NP / FTP
TSS = (duration × NP × IF) / (FTP × 36)
```

**Requirements**:
- Power data in FIT file
- FTP automatically estimated at 75th percentile if not provided

**Example**:
```bash
python fit_cycling.py race.fit --effort-method tss
```

**Note**: Falls back to hrTSS if power data unavailable.

---

### 2. hrTSS (Heart Rate TSS) - Default

**Best for**: Cyclists without power meters

**How it works**:
- Continuous interpolation between HR zones
- Non-linear gamma function (γ=1.5) emphasizes high HR
- 15-second smoothing window

**Zone multipliers** (TSS per hour):
- Zone 0 (Recovery): 20
- Zone 1 (Endurance): 30
- Zone 2 (Tempo): 50
- Zone 3 (Threshold): 70
- Zone 4 (VO2 Max): 90
- Zone 5 (Anaerobic): 120

**Advantages**:
- Works with HR monitor only
- Accounts for individual fitness via zones
- Smooth transitions between zones
- Industry-standard metric

**Example**:
```bash
python fit_cycling.py ride.fit --effort-method hrtss --karvonen 58,188
```

---

### 3. Custom Formula

**Best for**: Terrain-aware effort comparison

**How it works**:
Combines three normalized factors with cubic spline interpolation:

```python
# Weights (customizable in code)
WEIGHT_SLOPE = 0.5   # Terrain impact
WEIGHT_SPEED = 0.35  # Velocity effort
WEIGHT_HR = 0.15     # Physiological load

# Normalization
slope_norm = (slope + bound) / (2 × bound)  # [-bound, +bound] → [0,1]
speed_norm = (speed - min) / (max - min)    # [0, 60] km/h → [0,1]
hr_norm = spline(HR, zone_intensities)      # Zone-based curve

# Combined effort (rescaled 10-90%)
effort = 10 + 80 × weighted_average
```

**Zone intensity coefficients**:
- Zone 0: 0.25 (Recovery)
- Zone 1: 0.45 (Endurance)
- Zone 2: 0.70 (Tempo)
- Zone 3: 0.95 (Threshold)
- Zone 4: 1.25 (VO2 Max)
- Zone 5: 1.60 (Anaerobic)

**Characteristics**:
- Terrain-aware (slope weighted heavily)
- Speed-compensated (higher speed = more effort)
- HR-integrated (via zone intensities)
- Output range: 10-90% (never reaches extremes)

**Example**:
```bash
python fit_cycling.py alpine.fit --effort-method custom --slope-bound 25
```

---

## Circuit Detection and Turnaround Methods
Enable circuit detection for out-and-back routes to split your ride into outbound and return segments.

### Basic Usage
```bash
# Use default method (dynamic_centroid_refined)
python fit_cycling.py ride.fit --circuit

# Specify a method
python fit_cycling.py ride.fit --circuit --turnaround-method hybrid

# Adjust edge exclusion (default 5%)
python fit_cycling.py ride.fit --circuit --edge-percent 0.10
```

### Method Comparison Table

| Method | Best For | Speed | Accuracy | Requires |
|--------|----------|-------|----------|----------|
| **dynamic_centroid_refined** ⭐ | General use | Fast | ★★★★★ | GPS |
| **hybrid** | Varied terrain | Medium | ★★★★☆ | GPS |
| **furthest_point** | Simple routes | Fast | ★★★☆☆ | GPS |
| **cumulative_distance** | Distance-tracked | Fast | ★★★★☆ | Distance data |
| **3d_curvature** | Sharp turns | Medium | ★★★★☆ | GPS |
| **symmetry** | Perfect loops | Slow | ★★★★★ | GPS |
| **pca** | Straight routes | Fast | ★★★☆☆ | GPS |
| **self_overlap** | Close returns | Slow | ★★★★☆ | GPS |

---

## Visualization Control

### Plot Types

```bash
# GPS map with route and slope coloring
--plots gps

# Stacked metrics (speed, effort, slope, HR)
--plots metrics

# Speed-HR correlation with KDE
--plots correlation

# 3D trajectory (lon, lat, time)
--plots 3d

# Show all (default)
--plots all

# Combine multiple
--plots gps,metrics,correlation
```

### Examples

**GPS-only analysis**:
```bash
python fit_cycling.py ride.fit --plots gps --circuit
```

**Indoor training (no GPS)**:
```bash
python fit_cycling.py zwift.fit --no-gps --plots metrics,correlation
```

**Complete analysis with saved figures**:
```bash
python fit_cycling.py race.fit --plots all --save-figs output/race-2026/
```

---

## Advanced Parameters

### Slope Handling

```bash
# Adjust slope clipping bounds (default: ±25%)
python fit_cycling.py alpine.fit --slope-bound 30

# For extreme terrain
python fit_cycling.py mountain.fit --slope-bound 35
```

**How it works**:
1. Raw slopes calculated from altitude/distance
2. Clips extreme values (> ±bound%)
3. Smooths with 20m rolling window
4. Final clip at bound

**💡 Tip**: Increase bound for mountainous terrain, decrease for flat routes.

---

### Extrema Detection

```bash
# Adjust sensitivity (default: 30 points)
python fit_cycling.py ride.fit --extrema-window 50

# More sensitive (finds more peaks)
python fit_cycling.py ride.fit --extrema-window 20
```

**How it works**: Uses `scipy.signal.argrelextrema` with order parameter.
- **Lower value**: More peaks detected
- **Higher value**: Only major peaks

---

### Circuit Detection Tuning

```bash
# Exclude larger edges (default: 5%)
python fit_cycling.py ride.fit --circuit --edge-percent 0.10

# Tighter exclusion
python fit_cycling.py ride.fit --circuit --edge-percent 0.03
```

**Edge exclusion**: Prevents detecting turnaround at start or end of ride.

---

## Understanding the Visualizations

### 1. GPS Map

**With circuit detection** (`--circuit`):
- **Left panel**: Outbound route
- **Right panel**: Return route
- **Black X**: Turnaround point
- **Colors**: Red (uphill) to Blue (downhill)

**Without circuit**:
- Single map with full route
- Slope-colored trajectory

**Interpretation**:
- Steep climbs: Dark red segments
- Fast descents: Dark blue segments
- Flat sections: White/light colors

---

### 2. Metrics Stack

4 synchronized time-series plots:

#### Speed (km/h)
- Blue line
- ▲ = local maxima (sprint/descent)
- ▼ = local minima (climb/stop)

#### Effort
- Purple line
- Shows calculated TSS/hrTSS/custom
- No extrema markers (rescaled 10-90%)

#### Slope (%)
- Green line
- Positive = climbing
- Negative = descending
- ▲ = hilltops, ▼ = valley bottoms

#### Heart Rate (bpm)
- Colored by HR zone
- ▲ = peak efforts
- ▼ = recovery points

**💡 Tip**: Vertical dashed lines connect extrema across all plots—great for correlating events!

---

### 3. Speed-HR Correlation

**Elements**:
- **Blue contours**: KDE (density estimation)
- **Scatter points**: Colored by slope
- **Black line**: Linear regression
- **Red = uphill**, **Blue = downhill**

**Interpretation**:
- **Tight correlation**: Good pacing, consistent effort
- **Scattered points**: Variable terrain or uneven pacing
- **Slope influence**: Red points below line = climbing hard, blue above = fast descents

**Example insights**:
```
High HR at low speed + red points = steep climbing
High HR at high speed + blue points = fast effort on flat
Low HR at high speed + blue points = easy descending
```

---

### 4. 3D GPS Trajectory

**Axes**:
- **X**: Longitude
- **Y**: Latitude
- **Z**: Time (seconds from start)
- **Color**: Effort level (purple = high)

**Interpretation**:
- Visualizes entire ride in spacetime
- Helps identify when/where hard efforts occurred
- Good for comparing multiple laps

---

## Output Summary Explained

```
=== Ride Summary ===
Duration: 95.3 minutes          # Total ride time
Distance: 42.15 km              # GPS-tracked distance
Avg Speed: 26.5 km/h            # Mean speed (moving + stopped)
Max Speed: 54.2 km/h            # Peak speed (usually descent)
Avg HR: 152 bpm                 # Mean heart rate
Max HR: 178 bpm                 # Peak heart rate
Avg Effort: 67.3                # Mean effort (method-dependent)
```

**Turnaround detection output** (with `--circuit`):
```
============================================================
DÉTECTION DE DEMI-TOUR
============================================================

[MÉTHODE] Dynamic centroid + affinement local
✓ Demi-tour détecté à l'indice 1247
✓ Affiné : 1256

============================================================
Position dans le parcours : 1256/2543 (49.4%)
Distance au départ : 21.34 km
Distance à l'arrivée : 21.18 km
Distance départ → arrivée : 0.87 km
============================================================
```

**Interpretation**:
- **Position**: Should be ~50% for symmetrical out-and-back
- **Distance to start/end**: Should be similar for loop routes
- **Start→end distance**: Small value confirms circuit route

---

## Use Cases & Examples

### Weekend Training Ride
```bash
python fit_cycling.py weekend_ride.fit \
  --karvonen 58,188 \
  --circuit \
  --effort-method hrtss \
  --plots all \
  --save-figs analysis/weekend/
```

**Scenario**: Standard 2-hour training ride on familiar route.

---

### Race Analysis with Power
```bash
python fit_cycling.py criterium_race.fit \
  --max-hr 192 \
  --effort-method tss \
  --plots all \
  --save-figs races/2026-01-06/
```

**Scenario**: Criterium with power meter data.

---

### Alpine Climbing Expedition
```bash
python fit_cycling.py alpe_dhuez.fit \
  --circuit \
  --karvonen 52,182 \
  --slope-bound 35 \
  --turnaround-method dynamic_centroid_refined \
  --plots gps,metrics \
  --save-figs climbs/alpe/
```

**Scenario**: Mountain ascent with steep gradients.

---

### Indoor Smart Trainer Session
```bash
python fit_cycling.py zwift_workout.fit \
  --zones 115,130,145,160,175 \
  --no-gps \
  --effort-method custom \
  --plots metrics,correlation \
  --save-figs indoor/2026/week02/
```

**Scenario**: Zwift or TrainerRoad workout, no GPS data.

---

### Long Touring Ride
```bash
python fit_cycling.py gran_fondo.fit \
  --zones-age 35 \
  --effort-method hrtss \
  --plots gps,correlation \
  --extrema-window 50
```

**Scenario**: 6-hour gran fondo, point-to-point route (no `--circuit`).

---

### Commute Tracking
```bash
python fit_cycling.py commute.fit \
  --zones 110,125,140,155,170 \
  --plots gps \
  --save-figs commutes/
```

**Scenario**: Daily commute route analysis.

---

## Tips for Best Results

### 1. Accurate HR Zones ⭐
**Recommended**: Use Karvonen method
```bash
python fit_cycling.py ride.fit --karvonen 55,185
```

**Measuring resting HR**:
- Take first thing in morning, still in bed
- Average over 3-5 days
- Should be 40-60 bpm for trained cyclists

**Measuring max HR**:
- Field test: 5min warmup → 3min max effort → note peak
- Or use previous race data max
- Don't use 220-age (often inaccurate)

---

### 2. Choose Right Effort Method

| Situation | Method | Command |
|-----------|--------|---------|
| Have power meter | TSS | `--effort-method tss` |
| HR monitor only | hrTSS | `--effort-method hrtss` |
| Steep terrain | Custom | `--effort-method custom` |
| Quick comparison | hrTSS | Default |

---

### 3. Circuit Detection Best Practices

**✅ Use `--circuit` for**:
- Out-and-back training rides
- Loop routes returning to start
- Interval sessions on same stretch

**❌ Don't use `--circuit` for**:
- Point-to-point rides
- Figure-8 routes
- Races with complex circuits

**Validation**: Check turnaround detection output:
```
Position dans le parcours : 1256/2543 (49.4%)  ← Should be ~50%
Distance départ → arrivée : 0.87 km            ← Should be small
```

If position is far from 50% or start→end distance is large, don't use `--circuit`.

---

### 4. Slope Bounds

**Default (±25%)**: Works for most riding
**Increase (±30-35%)**: Mountains, Alpine climbs
**Decrease (±10-20%)**: Flat terrain, time trials

```bash
# Mountain stage
python fit_cycling.py alpe.fit --slope-bound 35

# Flat time trial
python fit_cycling.py tt.fit --slope-bound 15
```

---

### 5. Save Your Analysis

Create organized archives:
```bash
# By date
python fit_cycling.py ride.fit --save-figs analysis/2026-01-06/

# By ride type
python fit_cycling.py ride.fit --save-figs training/intervals/

# By location
python fit_cycling.py ride.fit --save-figs routes/canyon-loop/
```

Figures saved:
- `gps_map.png`
- `metrics_stack.png`
- `correlation.png`
- `gps_3d.png`

---

## Troubleshooting

### GPS Issues

**"No GPS data available"**
- FIT file doesn't contain position data
- **Solution**: Use `--no-gps` for indoor activities

**"Insufficient GPS data for 3D plot"**
- Not enough valid GPS points
- **Solution**: Check GPS signal quality, skip 3D plot

**Turnaround detected at wrong location**
- Method not suitable for your route type
- **Solution**: Try different method or disable `--circuit`

---

### Heart Rate Issues

**"No heart rate data found"**
- HR monitor not connected or not recording
- **Solution**: Check device pairing, analysis will skip HR-dependent features

**Effort values seem wrong**
- Incorrect HR zones
- **Solution**: Use Karvonen method with accurate resting/max HR

**HR zone colors not showing**
- Not enough valid HR data points
- **Solution**: Verify HR monitor was working throughout ride

---

### Power Issues

**"No power data available, falling back to hrTSS"**
- Expected behavior without power meter
- **Solution**: Use `--effort-method hrtss` explicitly, or get power meter data

**TSS values seem too high/low**
- FTP estimation might be off
- **Solution**: Script estimates FTP at 75th percentile—may need manual override in code

---

### Performance Issues

**Script is very slow**
- Using computationally expensive method
- **Solutions**:
  - Avoid `symmetry` or `variance` (slowest)
  - Reduce extrema window: `--extrema-window 50`

**Memory error on large files**
- FIT file has many data points (>10k)
- **Solution**: Use lighter turnaround method, skip 3D plot

---

### Data Quality

**Slope values look wrong**
- Noisy GPS altitude data
- **Solution**: Increase `--slope-bound` to clip extremes

**Speed spikes in visualization**
- GPS glitches
- **Solution**: Normal, script clips speeds >60 km/h automatically

**Extrema markers too frequent/sparse**
- Sensitivity issue
- **Solution**: Adjust `--extrema-window` (default: 30)

---

## FAQ

### General

**Q: Which effort method should I use?**
A: If you have a power meter, use `tss`. Otherwise, use `hrtss` (default). Use `custom` for terrain-aware comparisons.

**Q: How accurate is the turnaround detection?**
A: Very accurate with `dynamic_centroid_refined` for true out-and-back routes. Try different methods if default fails.

**Q: Can I analyze indoor training?**
A: Yes! Use `--no-gps --plots metrics,correlation`. GPS plots will be skipped.

**Q: Does this work with Garmin/Wahoo/etc.?**
A: Yes, any device that exports FIT files will work.

### Technical

**Q: What's the difference between slope and slope_final?**
A: `slope` is raw calculation, `slope_final` is smoothed and clipped. Visualizations use `slope_final`.

**Q: Why is effort rescaled 10-90% in custom method?**
A: Prevents extreme values, reflects that true 0% and 100% effort are never achieved in real riding.

**Q: Can I export the processed data?**
A: Not currently, but easy to add. Check the DataFrame (`df`) in code; contains all calculated metrics.

**Q: How is slope smoothed?**
A: 20-meter rolling window (forward + backward) with mean aggregation. Reduces GPS noise significantly.

### Advanced

**Q: Can I customize the effort formula weights?**
A: Yes! Edit constants in code:
```python
WEIGHT_SLOPE = 0.5   # Terrain
WEIGHT_SPEED = 0.35  # Velocity  
WEIGHT_HR = 0.15     # Physiological
```

**Q: What if I want different HR zone boundaries?**
A: Script supports 6 zones (0-5). Edit `create_zone_dict()` function for custom setup.

**Q: Can I add my own turnaround detection method?**
A: Yes! Add function to `gps_helper.py`:
```python
def method_my_custom(df, edge_pct=EDGE_PERCENT):
    # Your algorithm here
    return turnaround_idx
```
Then add to `get_turnaround_method()` dict.

**Q: How do I compare multiple rides?**
A: Currently not supported. Run script separately for each ride, save figures, compare manually.

---

## Changelog

**Current**
- Added multiple turnaround detection methods
- Implemented TSS/hrTSS with continuous zone interpolation
- Added custom effort formula with cubic spline
- Improved slope smoothing (20m window)
- Fixed GPS map colorbar positioning
- Added extensive documentation

**Past**
- Basic GPS visualization
- Simple effort calculation
- Simple turnaround detection

---

## Contributing

Contributions welcome!

**Ideas for improvements**:
- Export processed data to CSV
- Multi-ride comparison mode
- Interactive plot mode
- Web dashboard interface
- Custom FTP (Functional Threshold Power) input for TSS
- Lap analysis
- Power curve generation

**How to contribute**:
1. Fork repository
2. Create feature branch
3. Test thoroughly
4. Submit pull request

---

## License

Unlicense - Public Domain

