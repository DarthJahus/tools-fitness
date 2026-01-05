"""
FIT Cycling Analyzer
Analyzes cycling data from FIT files: GPS tracking, effort calculation, and performance metrics.

Usage:
    python fit_cycling.py <file.fit>
    python fit_cycling.py <file.fit> --zones 112,124,136,149,161
    python fit_cycling.py <file.fit> --max-hr 185 --circuit
    python fit_cycling.py <file.fit> --karvonen 60,185 --plots gps,metrics
    python fit_cycling.py <file.fit> --effort-method hrtss
"""

# ToDo: Make sure the turning point is correctly detected
# ToDo? Improve the custom formula for effort
# ToDo: Test it with multiple .FIT files

import fitdecode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # ToDo: Use or remove
from scipy.signal import argrelextrema
from scipy.spatial.distance import cdist
import argparse
import sys
from pathlib import Path

# -------------------------
# Constants
# -------------------------
SLOPE_WINDOW = 5  # Rolling window for slope smoothing
SLOPE_BOUNDS_DEFAULT = (-10, 20)  # Default slope clipping bounds (%)
EXTREMA_ORDER = 30  # Window for local min/max detection
EDGE_PERCENT = 0.05  # Edge percentage for turnaround detection
SEMICIRCLES_TO_DEGREES = 180.0 / 2 ** 31  # FIT file coordinate conversion
OVERLAP_THRESHOLD = 0.0001  # ~11 meters in degrees
MIN_OVERLAP_POINTS = 30
HEADING_WINDOW = 10

# hrTSS multipliers per zone (TSS per hour)
HRTSS_ZONE_MULTIPLIERS = {
    0: 20,  # Recovery
    1: 30,  # Endurance
    2: 50,  # Tempo
    3: 70,  # Threshold
    4: 90,  # VO2 Max
    5: 120  # Anaerobic
}

# Zone intensity coefficients for custom effort
ZONE_INTENSITY = {
    0: 0.25,  # Recovery
    1: 0.45,  # Endurance
    2: 0.70,  # Tempo
    3: 0.95,  # Threshold
    4: 1.25,  # VO2 Max
    5: 1.60   # Anaerobic
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze cycling data from FIT files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Zone Calculation Methods (mutually exclusive):
  --zones X1,X2,X3,X4,X5    Manual zone boundaries (default: 112,124,136,149,161)
  --max-hr MAX              Calculate zones from max HR (percentage method)
  --karvonen MIN,MAX        Karvonen/HRR method with resting and max HR
  --zones-age AGE           Calculate from age (220 - age formula)

Effort Calculation Methods:
  tss      - Training Stress Score (requires power data, falls back to hrTSS)
  hrtss    - Heart Rate based TSS (default)
  custom   - Custom formula with HR, slope, and speed

Examples:
  %(prog)s activity.fit
  %(prog)s activity.fit --circuit --plots gps,metrics
  %(prog)s activity.fit --karvonen 60,185 --effort-method hrtss
  %(prog)s activity.fit --no-gps --plots metrics,correlation
        """
    )

    parser.add_argument('file', help='Path to FIT file')

    # Zone calculation methods (mutually exclusive)
    zone_group = parser.add_mutually_exclusive_group()
    zone_group.add_argument(
        '--zones',
        type=str,
        help='Manual zone boundaries: X1,X2,X3,X4,X5 (default: 112,124,136,149,161)'
    )
    zone_group.add_argument(
        '--max-hr',
        type=int,
        help='Maximum heart rate for automatic zone calculation'
    )
    zone_group.add_argument(
        '--karvonen',
        type=str,
        help='Karvonen/HRR method: RESTING,MAX (e.g., 60,185)'
    )
    zone_group.add_argument(
        '--zones-age',
        type=int,
        help='Calculate max HR from age using 220-age formula'
    )

    # Circuit detection
    parser.add_argument(
        '--circuit',
        action='store_true',
        help='Enable turnaround detection for out-and-back routes'
    )
    parser.add_argument(
        '--edge-percent',
        type=float,
        default=EDGE_PERCENT,
        help=f'Edge percentage for turnaround detection (default: {EDGE_PERCENT})'
    )

    # Effort calculation
    parser.add_argument(
        '--effort-method',
        choices=['tss', 'hrtss', 'custom'],
        default='hrtss',
        help='Effort calculation method (default: hrtss)'
    )

    # Analysis parameters
    parser.add_argument(
        '--slope-bounds',
        type=str,
        help='Slope clipping bounds: MIN,MAX (default: -10,20)'
    )
    parser.add_argument(
        '--extrema-window',
        type=int,
        default=EXTREMA_ORDER,
        help=f'Window for local min/max detection (default: {EXTREMA_ORDER})'
    )

    # Visualization
    parser.add_argument(
        '--plots',
        type=str,
        default='all',
        help='Comma-separated plots: gps,metrics,correlation,3d,all (default: all)'
    )
    parser.add_argument(
        '--no-gps',
        action='store_true',
        help='Skip GPS-based analysis (for indoor activities)'
    )
    parser.add_argument(
        '--save-figs',
        type=str,
        help='Directory to save figures'
    )

    return parser.parse_args()


def parse_zones(zones_str):
    """Parse zone boundaries from comma-separated string."""
    try:
        zones = [int(x.strip()) for x in zones_str.split(',')]
        if len(zones) != 5:
            raise ValueError(f"Expected 5 zone boundaries, got {len(zones)}")
        if zones != sorted(zones):
            raise ValueError("Zone boundaries must be in ascending order")
        return zones
    except ValueError as e:
        print(f"Error: Invalid zone format: {e}")
        sys.exit(1)


def parse_karvonen(karvonen_str):
    """Parse Karvonen parameters from comma-separated string."""
    try:
        parts = [int(x.strip()) for x in karvonen_str.split(',')]
        if len(parts) != 2:
            raise ValueError(f"Expected 2 values (resting,max), got {len(parts)}")
        resting_hr, max_hr = parts
        if resting_hr >= max_hr:
            raise ValueError(f"Resting HR must be less than max HR")
        return resting_hr, max_hr
    except ValueError as e:
        print(f"Error: Invalid Karvonen format: {e}")
        sys.exit(1)


def parse_slope_bounds(bounds_str):
    """Parse slope bounds from comma-separated string."""
    try:
        parts = [float(x.strip()) for x in bounds_str.split(',')]
        if len(parts) != 2:
            raise ValueError(f"Expected 2 values (min,max), got {len(parts)}")
        if parts[0] >= parts[1]:
            raise ValueError("Min bound must be less than max bound")
        return tuple(parts)
    except ValueError as e:
        print(f"Error: Invalid slope bounds: {e}")
        sys.exit(1)


def calculate_max_hr_from_age(age):
    """Calculate maximum heart rate from age using 220-age formula."""
    return 220 - age


def calculate_zones_from_max_hr(max_hr):
    """Calculate heart rate zones from maximum heart rate (percentage method)."""
    return [
        int(max_hr * 0.60),  # Zone 1: 50-60%
        int(max_hr * 0.70),  # Zone 2: 60-70%
        int(max_hr * 0.80),  # Zone 3: 70-80%
        int(max_hr * 0.90),  # Zone 4: 80-90%
        int(max_hr * 1.00)  # Zone 5: 90-100%
    ]


def calculate_zones_hrr(max_hr, resting_hr):
    """Calculate heart rate zones using Heart Rate Reserve (Karvonen formula)."""
    hr_reserve = max_hr - resting_hr
    return [
        int((hr_reserve * 0.60) + resting_hr),
        int((hr_reserve * 0.70) + resting_hr),
        int((hr_reserve * 0.80) + resting_hr),
        int((hr_reserve * 0.90) + resting_hr),
        int((hr_reserve * 1.00) + resting_hr)
    ]


def create_zone_dict(zone_boundaries):
    """Create zone dictionary from boundaries."""
    colors = ["lightgrey", "lightblue", "green", "orange", "red", "purple"]
    zones = {}

    # Zone 0
    zones[f"Zone 0 (0-{zone_boundaries[0]})"] = (0, zone_boundaries[0], colors[0])

    # Zones 1-4
    for i in range(len(zone_boundaries) - 1):
        low = zone_boundaries[i] + 1
        high = zone_boundaries[i + 1]
        zones[f"Zone {i + 1} ({low}-{high})"] = (low, high, colors[i + 1])

    # Zone 5
    zones[f"Zone 5 ({zone_boundaries[4] + 1}+)"] = (zone_boundaries[4] + 1, 300, colors[5])

    return zones


def get_hr_zone_number(hr, zone_boundaries):
    """Get zone number (0-5) for a given heart rate."""
    if pd.isna(hr) or hr <= 0:
        return 0
    if hr <= zone_boundaries[0]:
        return 0
    for i in range(len(zone_boundaries) - 1):
        if zone_boundaries[i] < hr <= zone_boundaries[i + 1]:
            return i + 1
    return 5  # Above all zones


def get_zone_color(hr, zones):
    """Get color for a given heart rate."""
    for zone_name, (low, high, color) in zones.items():
        if low <= hr <= high:
            return color
    return "grey"


def load_fit_file(file_path):
    """Load and parse FIT file."""
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    records = []
    try:
        with fitdecode.FitReader(file_path) as fit:
            for frame in fit:
                if frame.frame_type == fitdecode.FIT_FRAME_DATA and frame.name == "record":
                    record = {field.name: field.value for field in frame.fields}
                    records.append(record)
    except Exception as e:
        print(f"Error: Failed to read FIT file: {e}")
        sys.exit(1)

    if not records:
        print("Error: No records found in FIT file")
        sys.exit(1)

    return pd.DataFrame(records)


def prepare_data(df, has_gps=True, slope_bounds=SLOPE_BOUNDS_DEFAULT):
    """Prepare and process cycling data."""
    # Time
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["time_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    # GPS coordinates
    if has_gps and "position_lat" in df.columns and "position_long" in df.columns:
        df["lat_deg"] = pd.to_numeric(df.get("position_lat"), errors="coerce") * SEMICIRCLES_TO_DEGREES
        df["lon_deg"] = pd.to_numeric(df.get("position_long"), errors="coerce") * SEMICIRCLES_TO_DEGREES
    else:
        df["lat_deg"] = np.nan
        df["lon_deg"] = np.nan

    # Speed / Distance / Altitude
    df["enhanced_speed"] = pd.to_numeric(df.get("enhanced_speed"), errors="coerce")
    df["speed"] = df.get("speed", df["enhanced_speed"])
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df["speed_kmh"] = df["speed"] * 3.6
    df["distance"] = pd.to_numeric(df.get("distance"), errors="coerce")
    df["enhanced_altitude"] = pd.to_numeric(df.get("enhanced_altitude"), errors="coerce")
    df["altitude"] = df.get("altitude", df["enhanced_altitude"])
    df["altitude"] = pd.to_numeric(df["altitude"], errors="coerce")

    # Heart rate
    df["heart_rate"] = pd.to_numeric(df.get("heart_rate"), errors="coerce")

    # Power (if available)
    df["power"] = pd.to_numeric(df.get("power"), errors="coerce")

    # Calculate slope
    df["delta_alt"] = df["altitude"].diff()
    df["delta_dist"] = df["distance"].diff()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["slope"] = (df["delta_alt"] / df["delta_dist"]) * 100.0
    df.loc[(df["delta_dist"] <= 0) | ~np.isfinite(df["slope"]), "slope"] = np.nan
    df["slope"] = df["slope"].rolling(window=SLOPE_WINDOW, center=True, min_periods=1).mean()
    df["slope_clipped"] = df["slope"].clip(slope_bounds[0], slope_bounds[1])

    return df


def calculate_effort_custom(df, zone_boundaries, resting_hr=None):
    """
    Zone-aware custom effort calculation.
    Uses HR zones as primary intensity, with slope and speed as modifiers.
    """
    # Assign HR zones
    df["hr_zone"] = df["heart_rate"].apply(
        lambda hr: get_hr_zone_number(hr, zone_boundaries)
    )

    # Base intensity from zone (exponential scaling)
    zone_intensity = df["hr_zone"].map(ZONE_INTENSITY)

    # Slope modifier: climbing is harder, descending is easier
    # ±10% slope = ±40% effort change
    slope_modifier = 1 + (df["slope_clipped"] / 10) * 0.4
    slope_modifier = slope_modifier.clip(0.5, 2.0)

    # Speed modifier: higher speed = wind resistance
    # Normalized to typical cycling speeds (25 km/h baseline)
    speed_baseline = 25.0
    speed_ratio = (df["speed_kmh"] / speed_baseline).clip(0.1, 3.0)
    speed_modifier = 0.8 + 0.2 * (speed_ratio ** 1.5)

    # Combined effort per second
    effort_per_second = zone_intensity * slope_modifier * speed_modifier * 30

    # Cumulative effort
    return effort_per_second.cumsum()


def calculate_effort_hrtss(df, zone_boundaries):
    """
    Calculate heart rate based Training Stress Score (hrTSS).
    Based on time spent in each HR zone.
    """
    # Assign zones
    df["hr_zone"] = df["heart_rate"].apply(
        lambda hr: get_hr_zone_number(hr, zone_boundaries)
    )

    # Calculate TSS per second based on zone
    df["tss_per_second"] = df["hr_zone"].map(
        lambda z: HRTSS_ZONE_MULTIPLIERS.get(z, 20) / 3600.0
    )

    # Cumulative TSS
    effort = df["tss_per_second"].cumsum()

    return effort


def calculate_effort_tss(df, ftp=None):
    """
    Calculate Training Stress Score (TSS) from power data.
    Falls back to hrTSS if power data unavailable.
    """
    if "power" not in df.columns or df["power"].isna().all():
        print("Warning: No power data available, falling back to hrTSS")
        return None

    # If FTP not provided, estimate as 95% of 20-min max power
    if ftp is None:
        # Simple estimation: use 75th percentile of power
        ftp = df["power"].dropna().quantile(0.75)
        print(f"Info: Estimated FTP = {ftp:.0f}W (adjust with --ftp if needed)")

    # Normalized Power (simplified: 30s rolling mean to 4th power)
    rolling_power = df["power"].rolling(window=30, min_periods=1).mean()
    np_power = (rolling_power ** 4).rolling(window=30, min_periods=1).mean() ** 0.25

    # Intensity Factor
    intensity_factor = np_power / ftp

    # TSS calculation
    duration_hours = df["time_s"] / 3600.0
    tss = (df["time_s"] * np_power * intensity_factor) / (ftp * 3600) * 100

    return tss.cumsum()


def calculate_heading(lat1, lon1, lat2, lon2):
    """Calculate bearing/heading between two points (degrees)."""
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    heading = np.arctan2(y, x)
    return np.degrees(heading) % 360


def calculate_moving_heading(df, window=HEADING_WINDOW):
    """Calculate moving average heading vector."""
    coords = df[["lat_deg", "lon_deg"]].copy()

    headings = []
    for i in range(len(coords)):
        if i == 0:
            headings.append(np.nan)
            continue

        # Look back up to 'window' points for stable heading
        start_idx = max(0, i - window)
        lat1, lon1 = coords.iloc[start_idx][["lat_deg", "lon_deg"]]
        lat2, lon2 = coords.iloc[i][["lat_deg", "lon_deg"]]

        if pd.notna([lat1, lon1, lat2, lon2]).all():
            heading = calculate_heading(lat1, lon1, lat2, lon2)
            headings.append(heading)
        else:
            headings.append(np.nan)

    return pd.Series(headings, index=df.index)


def detect_turnaround(df, edge_pct=EDGE_PERCENT):
    """
    Detect turnaround by finding where the path overlaps with itself.
    Verifies: path overlap, heading reversal, and return to start.
    """
    if df[["lat_deg", "lon_deg"]].isna().all().all():
        print("Warning: No GPS data available for turnaround detection")
        return None

    coords = df[["lat_deg", "lon_deg"]].dropna()
    if len(coords) < 50:
        print("Warning: Insufficient GPS points for turnaround detection")
        return None

    # 1. Find farthest point from start (candidate turnaround)
    start_point = coords.iloc[0].values
    distances_from_start = np.sqrt(
        (coords["lat_deg"] - start_point[0]) ** 2 +
        (coords["lon_deg"] - start_point[1]) ** 2
    )

    candidate_idx = distances_from_start.idxmax()

    # 2. Check if it's near edges
    total_points = len(coords)
    edge_n = max(1, int(total_points * edge_pct))
    coord_indices = coords.index.tolist()
    candidate_position = coord_indices.index(candidate_idx)

    if candidate_position < edge_n or candidate_position > total_points - edge_n:
        print("Warning: Candidate turnaround at route edge")
        return None

    # 3. Verify path overlap
    outbound = coords.iloc[:candidate_position].values
    inbound = coords.iloc[candidate_position:].values

    if len(inbound) < MIN_OVERLAP_POINTS:
        print("Warning: Not enough points after candidate to verify overlap")
        return None

    # Calculate pairwise distances between outbound and inbound points
    distances = cdist(inbound, outbound, metric='euclidean')
    min_distances = distances.min(axis=1)

    # Count overlapping points
    overlap_points = (min_distances < OVERLAP_THRESHOLD).sum()
    overlap_ratio = overlap_points / len(inbound)

    print(f"Path overlap analysis:")
    print(f"  - Candidate at index {candidate_idx} (position {candidate_position}/{total_points})")
    print(f"  - Overlapping points: {overlap_points}/{len(inbound)} ({overlap_ratio * 100:.1f}%)")
    print(f"  - Median return distance: {np.median(min_distances) * 111:.1f}m")

    if overlap_points < MIN_OVERLAP_POINTS:
        print("Warning: Insufficient path overlap - may not be out-and-back route")
        return None

    # 4. Verify heading reversal
    df_temp = df.copy()
    df_temp["heading"] = calculate_moving_heading(df, window=HEADING_WINDOW)

    if candidate_idx in df_temp.index:
        pre_idx = coord_indices[max(0, candidate_position - HEADING_WINDOW)]
        post_idx = coord_indices[min(len(coord_indices) - 1, candidate_position + HEADING_WINDOW)]

        heading_before = df_temp.loc[pre_idx, "heading"]
        heading_after = df_temp.loc[post_idx, "heading"]

        if pd.notna([heading_before, heading_after]).all():
            heading_change = abs(heading_after - heading_before)
            if heading_change > 180:
                heading_change = 360 - heading_change

            print(f"  - Heading change: {heading_change:.1f}° (expect ~180° for turnaround)")

            if heading_change < 120:
                print("Warning: Heading change suggests this may not be a turnaround")

    # 5. Check return to start
    end_point = coords.iloc[-1].values
    end_to_start = np.sqrt(
        (end_point[0] - start_point[0]) ** 2 +
        (end_point[1] - start_point[1]) ** 2
    )
    max_dist = distances_from_start.max()

    print(f"  - Distance from end to start: {end_to_start * 111:.1f}m")
    print(f"  - Maximum distance from start: {max_dist * 111:.1f}m")

    if end_to_start > max_dist * 0.3:
        print("Warning: Route doesn't return close to start")

    if overlap_ratio > 0.5:  # At least 50% overlap
        print("✓ Out-and-back route detected")
        return candidate_idx
    else:
        print("✗ Not an out-and-back route")
        return None


def detect_extrema(series, order=EXTREMA_ORDER):
    """Detect local minima and maxima in a series."""
    values = series.values
    minima = argrelextrema(values, np.less_equal, order=order)[0]
    maxima = argrelextrema(values, np.greater_equal, order=order)[0]
    return minima, maxima


def plot_gps_map(df, zones, turnaround_idx=None, save_path=None):
    """Plot GPS map with slope coloring and optional turnaround detection."""
    gps_ok = df[["lon_deg", "lat_deg"]].dropna()

    if len(gps_ok) == 0:
        print("Warning: No GPS data available, skipping GPS map")
        return

    # Determine number of subplots
    if turnaround_idx is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        df_outbound = df.loc[:turnaround_idx].copy()
        df_return = df.loc[turnaround_idx:].copy()
        turnaround_lat = df.loc[turnaround_idx, "lat_deg"]
        turnaround_lon = df.loc[turnaround_idx, "lon_deg"]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        axes = [axes]  # Make it iterable

    # Map bounds
    lon_min, lon_max = gps_ok["lon_deg"].min(), gps_ok["lon_deg"].max()
    lat_min, lat_max = gps_ok["lat_deg"].min(), gps_ok["lat_deg"].max()
    pad_lon = (lon_max - lon_min) * 0.02
    pad_lat = (lat_max - lat_min) * 0.02

    # Color by HR zone
    df["zone_color"] = df["heart_rate"].apply(lambda hr: get_zone_color(hr, zones))

    if turnaround_idx is not None:
        # Outbound
        sc1 = axes[0].scatter(
            df_outbound["lon_deg"], df_outbound["lat_deg"],
            c=df_outbound["slope"], cmap="coolwarm",
            s=10, alpha=0.8, vmin=-15, vmax=15
        )
        axes[0].scatter(turnaround_lon, turnaround_lat,
                        color="black", marker="X", s=100, label="Turnaround", zorder=5)
        axes[0].set_title("Outbound")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlim(lon_min - pad_lon, lon_max + pad_lon)
        axes[0].set_ylim(lat_min - pad_lat, lat_max + pad_lat)
        axes[0].legend()
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")

        # Return
        sc2 = axes[1].scatter(
            df_return["lon_deg"], df_return["lat_deg"],
            c=df_return["slope"], cmap="coolwarm",
            s=10, alpha=0.8, vmin=-15, vmax=15
        )
        axes[1].scatter(turnaround_lon, turnaround_lat,
                        color="black", marker="X", s=100, label="Turnaround", zorder=5)
        axes[1].set_title("Return")
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].set_xlim(lon_min - pad_lon, lon_max + pad_lon)
        axes[1].set_ylim(lat_min - pad_lat, lat_max + pad_lat)
        axes[1].legend()
        axes[1].set_xlabel("Longitude")

        fig.suptitle("GPS Route - Outbound vs Return (color = slope %)", fontsize=14)
        fig.colorbar(sc1, ax=axes, fraction=0.03, pad=0.02, label="Slope (%)")
    else:
        # Single route
        sc = axes[0].scatter(
            df["lon_deg"], df["lat_deg"],
            c=df["slope"], cmap="coolwarm",
            s=10, alpha=0.8, vmin=-15, vmax=15
        )
        axes[0].set_title("GPS Route (color = slope %)")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlim(lon_min - pad_lon, lon_max + pad_lon)
        axes[0].set_ylim(lat_min - pad_lat, lat_max + pad_lat)
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        fig.colorbar(sc, ax=axes[0], fraction=0.03, pad=0.02, label="Slope (%)")

    plt.tight_layout()
    if save_path:
        plt.savefig(Path(save_path) / "gps_map.png", dpi=150)
    plt.show()


def plot_metrics_stack(df, zones, extrema_order=EXTREMA_ORDER, save_path=None):
    """Plot stacked metrics with local extrema detection."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Assign zone colors
    df["zone_color"] = df["heart_rate"].apply(lambda hr: get_zone_color(hr, zones))

    series_info = [
        ("speed_kmh", "Speed (km/h)", "tab:blue"),
        ("slope", "Slope (%)", "tab:green"),
        ("heart_rate", "Heart Rate (bpm)", None),  # Will use zone colors
        ("effort", "Effort", "tab:purple"),
    ]

    for ax, (col, label, color) in zip(axes, series_info):
        y = df[col].dropna().values
        x = df[col].dropna().index.map(lambda i: df.loc[i, "time_s"]).values

        if col == "heart_rate":
            # Color by HR zone
            for i in range(len(x) - 1):
                ax.plot(x[i:i + 2], y[i:i + 2],
                        color=df.loc[df[col].dropna().index[i], "zone_color"],
                        linewidth=1)
            ax.set_ylabel(label, color="black")
        else:
            ax.plot(x, y, color=color, label=label, linewidth=1.5)
            ax.set_ylabel(label, color=color)

        # Detect extrema
        minima, maxima = detect_extrema(df[col].dropna(), order=extrema_order)

        # Plot extrema
        for idx in minima:
            actual_idx = df[col].dropna().index[idx]
            ax.plot(df.loc[actual_idx, "time_s"], df.loc[actual_idx, col],
                    "v", color=color if color else "red", markersize=8)
            for ax_all in axes:
                ax_all.axvline(df.loc[actual_idx, "time_s"],
                               color=color if color else "red", alpha=0.15, linestyle="--")

        for idx in maxima:
            actual_idx = df[col].dropna().index[idx]
            ax.plot(df.loc[actual_idx, "time_s"], df.loc[actual_idx, col],
                    "^", color=color if color else "red", markersize=8)
            for ax_all in axes:
                ax_all.axvline(df.loc[actual_idx, "time_s"],
                               color=color if color else "red", alpha=0.15, linestyle="--")

        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")
    axes[0].set_title("Metrics Over Time with Local Extrema")

    plt.tight_layout()
    if save_path:
        plt.savefig(Path(save_path) / "metrics_stack.png", dpi=150)
    plt.show()


def plot_correlation(df, zones, save_path=None):
    """Plot speed vs heart rate correlation with KDE and regression."""
    corr_df = df[["speed_kmh", "heart_rate", "slope"]].dropna()

    if len(corr_df) < 10:
        print("Warning: Insufficient data for correlation plot")
        return

    plt.figure(figsize=(10, 7))

    x = corr_df["speed_kmh"]
    y = corr_df["heart_rate"]
    slope_color = corr_df["slope"].clip(-15, 15)

    # KDE background
    sns.kdeplot(x=x, y=y, fill=True, cmap="Blues", alpha=0.3, levels=10, thresh=0.1)

    # Scatter colored by slope
    sc = plt.scatter(x, y, c=slope_color, cmap="coolwarm",
                     s=15, alpha=0.6, vmin=-15, vmax=15)

    # Regression line
    sns.regplot(x=x, y=y, scatter=False, color="black",
                line_kws={"linewidth": 2.5, "alpha": 0.8})

    # Adjust bounds
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
        dx = (xmax - xmin) * 0.05
        plt.xlim(xmin - dx, xmax + dx)
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        dy = (ymax - ymin) * 0.05
        plt.ylim(ymin - dy, ymax + dy)

    plt.xlabel("Speed (km/h)", fontsize=12)
    plt.ylabel("Heart Rate (bpm)", fontsize=12)
    plt.title("Speed vs Heart Rate Correlation (color = slope %)", fontsize=14)
    plt.colorbar(sc, label="Slope (%)")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(Path(save_path) / "correlation.png", dpi=150)
    plt.show()


def plot_3d_gps(df, save_path=None):
    """Plot 3D GPS trajectory with time as Z-axis and effort as color."""
    mask3d = df[["lon_deg", "lat_deg", "time_s", "effort"]].dropna().index

    if len(mask3d) < 10:
        print("Warning: Insufficient GPS data for 3D plot")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    lon = df.loc[mask3d, "lon_deg"]
    lat = df.loc[mask3d, "lat_deg"]
    time = df.loc[mask3d, "time_s"]
    effort = df.loc[mask3d, "effort"]

    p = ax.scatter(lon, lat, time, c=effort, cmap="plasma", s=8, alpha=0.7)

    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_zlabel("Time (seconds)", fontsize=11)
    ax.set_title("3D GPS Trajectory (Z = time, color = effort)", fontsize=14)

    fig.colorbar(p, label="Effort", shrink=0.6, pad=0.1)

    plt.tight_layout()
    if save_path:
        plt.savefig(Path(save_path) / "gps_3d.png", dpi=150)
    plt.show()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Parse slope bounds
    slope_bounds = SLOPE_BOUNDS_DEFAULT
    if args.slope_bounds:
        slope_bounds = parse_slope_bounds(args.slope_bounds)

    # Determine zone calculation method
    zone_boundaries = None
    user_max_hr = None
    user_resting_hr = None

    if args.karvonen:
        user_resting_hr, user_max_hr = parse_karvonen(args.karvonen)
        zone_boundaries = calculate_zones_hrr(user_max_hr, user_resting_hr)
        print(f"Using Karvonen/HRR method (Resting: {user_resting_hr}, Max: {user_max_hr})")
    elif args.zones_age:
        user_max_hr = calculate_max_hr_from_age(args.zones_age)
        zone_boundaries = calculate_zones_from_max_hr(user_max_hr)
        print(f"Using age-based zones (Age: {args.zones_age}, Estimated Max HR: {user_max_hr})")
    elif args.max_hr:
        user_max_hr = args.max_hr
        zone_boundaries = calculate_zones_from_max_hr(user_max_hr)
        print(f"Using percentage method (Max HR: {user_max_hr})")
    else:
        zones_str = args.zones if args.zones else '112,124,136,149,161'
        zone_boundaries = parse_zones(zones_str)
        print(f"Using manual zones: {zone_boundaries}")

    zones = create_zone_dict(zone_boundaries)

    # Load and prepare data
    print(f"Loading FIT file: {args.file}")
    df = load_fit_file(args.file)

    has_gps = not args.no_gps
    df = prepare_data(df, has_gps=has_gps, slope_bounds=slope_bounds)

    # Check for heart rate data
    if df["heart_rate"].isna().all():
        print("Warning: No heart rate data found in FIT file")

    # Calculate effort
    print(f"Calculating effort using method: {args.effort_method}")

    if args.effort_method == 'tss':
        effort = calculate_effort_tss(df)
        if effort is None:
            print("Falling back to hrTSS method")
            effort = calculate_effort_hrtss(df, zone_boundaries)
    elif args.effort_method == 'hrtss':
        effort = calculate_effort_hrtss(df, zone_boundaries)
    else:  # custom
        effort = calculate_effort_custom(df, zone_boundaries, resting_hr=user_resting_hr)

    df["effort"] = effort

    # Turnaround detection
    turnaround_idx = None
    if args.circuit and has_gps:
        print("Detecting turnaround point...")
        turnaround_idx = detect_turnaround(df, edge_pct=args.edge_percent)
        if turnaround_idx:
            print(f"Turnaround detected at index {turnaround_idx} "
                  f"(time: {df.loc[turnaround_idx, 'time_s']:.0f}s)")

    # Prepare save directory
    save_path = None
    if args.save_figs:
        save_path = Path(args.save_figs)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Figures will be saved to: {save_path}")

    # Determine which plots to show
    plot_list = args.plots.lower().split(',')
    show_all = 'all' in plot_list

    # Generate plots
    if (show_all or 'gps' in plot_list) and has_gps:
        print("Generating GPS map...")
        plot_gps_map(df, zones, turnaround_idx, save_path)

    if show_all or 'metrics' in plot_list:
        print("Generating metrics stack...")
        plot_metrics_stack(df, zones, args.extrema_window, save_path)

    if show_all or 'correlation' in plot_list:
        print("Generating correlation plot...")
        plot_correlation(df, zones, save_path)

    if (show_all or '3d' in plot_list) and has_gps:
        print("Generating 3D GPS plot...")
        plot_3d_gps(df, save_path)

    print("Analysis complete!")

    # Print summary statistics
    print("\n=== Ride Summary ===")
    print(f"Duration: {df['time_s'].max() / 60:.1f} minutes")
    if not df["distance"].isna().all():
        print(f"Distance: {df['distance'].max() / 1000:.2f} km")
    if not df["speed_kmh"].isna().all():
        print(f"Avg Speed: {df['speed_kmh'].mean():.1f} km/h")
        print(f"Max Speed: {df['speed_kmh'].max():.1f} km/h")
    if not df["heart_rate"].isna().all():
        print(f"Avg HR: {df['heart_rate'].mean():.0f} bpm")
        print(f"Max HR: {df['heart_rate'].max():.0f} bpm")
    if not df["effort"].isna().all():
        print(f"Total Effort: {df['effort'].iloc[-1]:.1f}")


if __name__ == "__main__":
    main()
