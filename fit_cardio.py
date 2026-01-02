"""
FIT Cardio Analyzer
Analyzes heart rate data from FIT files and visualizes HR zones and recovery patterns.

Usage:
    python fit_cardio.py <file.fit>
    python fit_cardio.py <file.fit> --zones 112,124,136,149,161
    python fit_cardio.py <file.fit> --max-hr 185
    python fit_cardio.py <file.fit> --karvonen 60,185
    python fit_cardio.py <file.fit> --zones-age 30
"""

import fitdecode
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import argparse
import sys
from pathlib import Path

__no_mini_label = True

# Hardcoded parameters
PEAK_DETECTION_DISTANCE = 60  # Minimum separation between peaks
SLOPE_WINDOW = 20  # Number of points for slope calculation
SLOPE_THRESHOLD_FATIGUE_UP = 0.6  # Rapid rise indicates fatigue
SLOPE_THRESHOLD_ADAPT_UP = 0.2  # Controlled rise indicates adaptation
SLOPE_THRESHOLD_ADAPT_DOWN = -0.5  # Rapid descent indicates good recovery
SLOPE_THRESHOLD_FATIGUE_DOWN = -0.2  # Slow descent indicates poor recovery
MIN_CARDIO_HR = 90  # Minimum heart rate to be considered cardio activity


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze heart rate from FIT files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Zone Calculation Methods (mutually exclusive):
  --zones X1,X2,X3,X4,X5    Manual zone boundaries (default: 112,124,136,149,161)
  --max-hr MAX              Calculate zones from max HR (50-60-70-80-90%% method)
  --karvonen MIN,MAX        Karvonen/HRR method with resting and max HR
  --zones-age AGE           Calculate from age (220 - age formula)

Examples:
  %(prog)s activity.fit
  %(prog)s activity.fit --zones 110,120,135,150,165
  %(prog)s activity.fit --max-hr 185
  %(prog)s activity.fit --karvonen 60,185
  %(prog)s activity.fit --zones-age 30
        """
    )

    parser.add_argument('file', help='Path to FIT file')

    # Mutually exclusive zone calculation methods
    zone_group = parser.add_mutually_exclusive_group()
    zone_group.add_argument(
        '--zones',
        type=str,
        help='Manual zone boundaries: X1,X2,X3,X4,X5 (default: 112,124,136,149,161)'
    )
    zone_group.add_argument(
        '--max-hr',
        type=int,
        help='Maximum heart rate for automatic zone calculation (percentage method)'
    )
    zone_group.add_argument(
        '--karvonen',
        type=str,
        help='Karvonen/Heart Rate Reserve method: RESTING,MAX (e.g., 60,185)'
    )
    zone_group.add_argument(
        '--zones-age',
        type=int,
        help='Calculate max HR from age using 220-age formula, then compute zones'
    )

    return parser.parse_args()


def calculate_zones_from_max_hr(max_hr):
    """
    Calculate heart rate zones from maximum heart rate.
    Uses standard percentage zones: 50-60%, 60-70%, 70-80%, 80-90%, 90-100%

    Args:
        max_hr: Maximum heart rate

    Returns:
        List of zone boundary values [z1_upper, z2_upper, z3_upper, z4_upper, z5_upper]
    """
    return [
        int(max_hr * 0.60),  # Zone 1: 50-60% (upper bound)
        int(max_hr * 0.70),  # Zone 2: 60-70%
        int(max_hr * 0.80),  # Zone 3: 70-80%
        int(max_hr * 0.90),  # Zone 4: 80-90%
        int(max_hr * 1.00)   # Zone 5: 90-100%
    ]


def calculate_zones_hrr(max_hr, resting_hr):
    """
    Calculate heart rate zones using Heart Rate Reserve (Karvonen formula).
    This is the most accurate method as it accounts for individual fitness levels.

    Formula: Target HR = ((max_hr - resting_hr) × %intensity) + resting_hr

    Args:
        max_hr: Maximum heart rate
        resting_hr: Resting heart rate

    Returns:
        List of zone boundary values [z1_upper, z2_upper, z3_upper, z4_upper, z5_upper]
    """
    hr_reserve = max_hr - resting_hr

    return [
        int((hr_reserve * 0.60) + resting_hr),  # Zone 1: 50-60% HRR
        int((hr_reserve * 0.70) + resting_hr),  # Zone 2: 60-70% HRR
        int((hr_reserve * 0.80) + resting_hr),  # Zone 3: 70-80% HRR
        int((hr_reserve * 0.90) + resting_hr),  # Zone 4: 80-90% HRR
        int((hr_reserve * 1.00) + resting_hr)   # Zone 5: 90-100% HRR
    ]


def parse_zones(zones_str):
    """
    Parse zone boundaries from comma-separated string.

    Args:
        zones_str: String like "112,124,136,149,161"

    Returns:
        List of integers representing zone upper boundaries
    """
    try:
        zones = [int(x.strip()) for x in zones_str.split(',')]
        if len(zones) != 5:
            raise ValueError(f"Expected 5 zone boundaries, got {len(zones)}")
        if zones != sorted(zones):
            raise ValueError("Zone boundaries must be in ascending order")
        return zones
    except ValueError as e:
        print(f"Error: Invalid zone format: {e}")
        print("Expected format: --zones 112,124,136,149,161")
        sys.exit(1)


def parse_karvonen(karvonen_str):
    """
    Parse Karvonen parameters from comma-separated string.

    Args:
        karvonen_str: String like "60,185" (resting_hr,max_hr)

    Returns:
        Tuple of (resting_hr, max_hr)
    """
    try:
        parts = [int(x.strip()) for x in karvonen_str.split(',')]
        if len(parts) != 2:
            raise ValueError(f"Expected 2 values (resting,max), got {len(parts)}")
        resting_hr, max_hr = parts
        if resting_hr >= max_hr:
            raise ValueError(f"Resting HR ({resting_hr}) must be less than max HR ({max_hr})")
        if resting_hr < 30 or resting_hr > 100:
            print(f"Warning: Resting HR of {resting_hr} seems unusual (typical range: 40-80)")
        if max_hr < 120 or max_hr > 220:
            print(f"Warning: Max HR of {max_hr} seems unusual (typical range: 150-200)")
        return resting_hr, max_hr
    except ValueError as e:
        print(f"Error: Invalid Karvonen format: {e}")
        print("Expected format: --karvonen 60,185")
        sys.exit(1)


def calculate_max_hr_from_age(age):
    """
    Calculate maximum heart rate from age using 220-age formula.

    Args:
        age: Age in years

    Returns:
        Estimated maximum heart rate
    """
    if age < 10 or age > 100:
        print(f"Warning: Age of {age} seems unusual (typical range: 15-80)")
    return 220 - age


def create_zone_dict(zone_boundaries, user_max_hr=None, data_max_hr=None):
    """
    Create zone dictionary from boundaries.

    Args:
        zone_boundaries: List of 5 upper boundaries [z1, z2, z3, z4, z5]
        user_max_hr: Max HR specified by user (for auto-calculated zones)
        data_max_hr: Max HR from actual dataset (for manual zones)

    Returns:
        Dictionary with zone definitions

    Zone structure (non-overlapping):
        Zone 0: 0 to boundary[0]
        Zone 1: boundary[0]+1 to boundary[1]
        Zone 2: boundary[1]+1 to boundary[2]
        Zone 3: boundary[2]+1 to boundary[3]
        Zone 4: boundary[3]+1 to boundary[4]
        Zone 5: boundary[4]+1 to upper_limit

    Upper limit for zone 5:
        - If user_max_hr provided: use that value
        - Otherwise: max(data_max_hr, zone5_start) + 10 for visual padding
    """
    colors = ["lightgrey", "lightblue", "green", "orange", "red", "purple"]
    zones = {}

    # Calculate upper limit for zone 5
    zone5_start = zone_boundaries[4] + 1
    if user_max_hr is not None:
        zone5_upper = user_max_hr
    else:
        # Use data max or zone start, whichever is higher, plus 10 for visual padding
        zone5_upper = max(data_max_hr if data_max_hr else zone5_start, zone5_start) + 10

    # Zone 0: Recovery/Warmup (0 to first boundary, inclusive)
    zones[f"Zone 0 (0-{zone_boundaries[0]})"] = (0, zone_boundaries[0], colors[0])

    # Zones 1-4: Between boundaries (non-overlapping)
    for i in range(len(zone_boundaries) - 1):
        low = zone_boundaries[i] + 1
        high = zone_boundaries[i + 1]
        zones[f"Zone {i+1} ({low}-{high})"] = (
            low,
            high,
            colors[i+1]
        )

    # Zone 5: Maximum (last boundary + 1 and above)
    if user_max_hr is not None:
        zones[f"Zone 5 ({zone5_start}-{zone5_upper})"] = (zone5_start, zone5_upper, colors[5])
    else:
        zones[f"Zone 5 ({zone5_start}+)"] = (zone5_start, zone5_upper, colors[5])

    return zones


def load_fit_file(file_path):
    """
    Load and parse FIT file.

    Args:
        file_path: Path to FIT file

    Returns:
        pandas DataFrame with heart rate records
    """
    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Load FIT file
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
        print("Error: No heart rate records found in FIT file")
        sys.exit(1)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Check if heart_rate column exists
    if "heart_rate" not in df.columns:
        print("Error: No heart rate data found in FIT file")
        sys.exit(1)

    return df


def prepare_data(df):
    """
    Prepare and validate heart rate data.

    Args:
        df: Raw DataFrame from FIT file

    Returns:
        Prepared DataFrame with time and HR columns
    """
    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["time_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    # Convert heart rate to numeric
    df["heart_rate"] = pd.to_numeric(df.get("heart_rate"), errors="coerce")

    # Validate cardio data
    hr_valid = df["heart_rate"].dropna()
    hr_above_min = hr_valid[hr_valid > MIN_CARDIO_HR]

    if len(hr_above_min) == 0:
        print(f"Error: No heart rate values above {MIN_CARDIO_HR} BPM found")
        print("This does not appear to be a cardio activity")
        sys.exit(1)

    if len(hr_above_min) < len(hr_valid) * 0.1:
        print(f"Warning: Less than 10% of heart rate values are above {MIN_CARDIO_HR} BPM")
        print("This may not be a cardio activity")

    return df


def get_zone_color(hr, zones):
    """Determine zone color for a given heart rate."""
    for zone_name, (low, high, color) in zones.items():
        if low <= hr <= high:
            return color
    return "grey"


def analyze_slopes(df, peaks, zones):
    """
    Analyze heart rate slopes around peaks.

    Returns:
        Tuple of (adapt_up, fatigue_up, neutral_up, adapt_down, fatigue_down, neutral_down)
    """
    adapt_up = fatigue_up = neutral_up = 0
    adapt_down = fatigue_down = neutral_down = 0

    for i in peaks:
        if SLOPE_WINDOW < i < len(df) - SLOPE_WINDOW:
            # Upward slope (rising)
            time_up = df["time_s"].iloc[i] - df["time_s"].iloc[i - SLOPE_WINDOW]
            delta_hr_up = df["heart_rate"].iloc[i] - df["heart_rate"].iloc[i - SLOPE_WINDOW]
            slope_up = delta_hr_up / time_up if time_up != 0 else 0

            # Downward slope (recovery)
            time_down = df["time_s"].iloc[i + SLOPE_WINDOW] - df["time_s"].iloc[i]
            delta_hr_down = df["heart_rate"].iloc[i + SLOPE_WINDOW] - df["heart_rate"].iloc[i]
            slope_down = delta_hr_down / time_down if time_down != 0 else 0

            # Classify upward slope
            if slope_up > SLOPE_THRESHOLD_FATIGUE_UP:
                color_up = "red"  # Very rapid rise: high stress/fatigue
                fatigue_up += 1
            elif SLOPE_THRESHOLD_ADAPT_UP <= slope_up <= SLOPE_THRESHOLD_FATIGUE_UP:
                color_up = "green"  # Controlled rise: good adaptation
                adapt_up += 1
            else:
                color_up = "grey"  # Neutral / not significant
                neutral_up += 1

            plt.plot(
                df["time_s"].iloc[i - SLOPE_WINDOW:i] / 60,
                df["heart_rate"].iloc[i - SLOPE_WINDOW:i],
                color=color_up,
                linewidth=1.8
            )

            # Classify downward slope
            if slope_down < SLOPE_THRESHOLD_ADAPT_DOWN:
                color_down = "green"  # Rapid descent: good recovery
                adapt_down += 1
            elif slope_down > SLOPE_THRESHOLD_FATIGUE_DOWN:
                color_down = "orange"  # Slow descent: poor recovery
                fatigue_down += 1
            else:
                color_down = "grey"  # Neutral
                neutral_down += 1

            plt.plot(
                df["time_s"].iloc[i:i + SLOPE_WINDOW] / 60,
                df["heart_rate"].iloc[i:i + SLOPE_WINDOW],
                color=color_down,
                linewidth=1.8
            )

    return adapt_up, fatigue_up, neutral_up, adapt_down, fatigue_down, neutral_down


def plot_heart_rate(df, zones, peaks):
    """Create heart rate visualization with zones and analysis."""
    # Assign zone colors
    df["zone_color"] = df["heart_rate"].apply(lambda hr: get_zone_color(hr, zones))

    # Create figure FIRST
    plt.figure(figsize=(12, 6))

    # Plot heart rate points colored by zone
    plt.scatter(df["time_s"] / 60, df["heart_rate"], c=df["zone_color"], s=8)

    # Add zone backgrounds
    for label, (low, high, color) in zones.items():
        # Don't add Zone 0 to legend (warmup/recovery zone)
        legend_label = label if not label.startswith("Zone 0") else None
        plt.axhspan(low, high, color=color, alpha=0.1, label=legend_label)

    # Analyze and plot slopes on the SAME figure
    adapt_up = fatigue_up = neutral_up = 0
    adapt_down = fatigue_down = neutral_down = 0

    for i in peaks:
        if SLOPE_WINDOW < i < len(df) - SLOPE_WINDOW:
            # Upward slope (rising)
            time_up = df["time_s"].iloc[i] - df["time_s"].iloc[i - SLOPE_WINDOW]
            delta_hr_up = df["heart_rate"].iloc[i] - df["heart_rate"].iloc[i - SLOPE_WINDOW]
            slope_up = delta_hr_up / time_up if time_up != 0 else 0

            # Downward slope (recovery)
            time_down = df["time_s"].iloc[i + SLOPE_WINDOW] - df["time_s"].iloc[i]
            delta_hr_down = df["heart_rate"].iloc[i + SLOPE_WINDOW] - df["heart_rate"].iloc[i]
            slope_down = delta_hr_down / time_down if time_down != 0 else 0

            # Classify upward slope
            if slope_up > SLOPE_THRESHOLD_FATIGUE_UP:
                color_up = "red"  # Very rapid rise: high stress/fatigue
                fatigue_up += 1
            elif SLOPE_THRESHOLD_ADAPT_UP <= slope_up <= SLOPE_THRESHOLD_FATIGUE_UP:
                color_up = "green"  # Controlled rise: good adaptation
                adapt_up += 1
            else:
                color_up = "grey"  # Neutral / not significant
                neutral_up += 1

            plt.plot(
                df["time_s"].iloc[i - SLOPE_WINDOW:i] / 60,
                df["heart_rate"].iloc[i - SLOPE_WINDOW:i],
                color=color_up,
                linewidth=1.8
            )

            # Classify downward slope
            if slope_down < SLOPE_THRESHOLD_ADAPT_DOWN:
                color_down = "green"  # Rapid descent: good recovery
                adapt_down += 1
            elif slope_down > SLOPE_THRESHOLD_FATIGUE_DOWN:
                color_down = "orange"  # Slow descent: poor recovery
                fatigue_down += 1
            else:
                color_down = "grey"  # Neutral
                neutral_down += 1

            plt.plot(
                df["time_s"].iloc[i:i + SLOPE_WINDOW] / 60,
                df["heart_rate"].iloc[i:i + SLOPE_WINDOW],
                color=color_down,
                linewidth=1.8
            )

    # Plot peaks
    plt.scatter(
        df["time_s"].iloc[peaks] / 60,
        df["heart_rate"].iloc[peaks] + 2,
        color="black",
        marker="v",
        s=20,
        label="HR Peaks",
        alpha=0.8
    )

    # Annotate peaks
    for i in peaks:
        plt.text(
            df["time_s"].iloc[i] / 60,
            df["heart_rate"].iloc[i] + 3,
            str(int(df["heart_rate"].iloc[i])),
            color="black",
            fontsize=6,
            va="bottom",
            ha="center"
        )

    # Set Y-axis limits
    hr_valid = df["heart_rate"].dropna()
    hr_max = hr_valid.max()
    hr_min = hr_valid[hr_valid > MIN_CARDIO_HR].min()
    plt.ylim(hr_min * 0.9, hr_max * 1.1)

    # Add summary text if enabled
    if not __no_mini_label:
        # Calculate text positioning
        duration_minutes = df["time_s"].iloc[-1] / 60
        hr_range = hr_max - hr_min

        # Position text box in upper right area
        text_x = duration_minutes * 0.65
        text_y_start = hr_max * 0.98
        line_height = hr_range * 0.05

        # Rising slopes
        plt.text(
            text_x, text_y_start,
            "Rising HR:",
            fontsize=8,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )
        plt.text(text_x + duration_minutes * 0.08, text_y_start,
                 f"Adapt {adapt_up}", color="green", fontsize=8)
        plt.text(text_x + duration_minutes * 0.15, text_y_start,
                 f"Fatigue {fatigue_up}", color="red", fontsize=8)
        plt.text(text_x + duration_minutes * 0.23, text_y_start,
                 f"Neutral {neutral_up}", color="grey", fontsize=8)

        # Descending slopes
        text_y_recovery = text_y_start - line_height
        plt.text(
            text_x, text_y_recovery,
            "Recovery:",
            fontsize=8,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )
        plt.text(text_x + duration_minutes * 0.08, text_y_recovery,
                 f"Good {adapt_down}", color="green", fontsize=8)
        plt.text(text_x + duration_minutes * 0.15, text_y_recovery,
                 f"Poor {fatigue_down}", color="orange", fontsize=8)
        plt.text(text_x + duration_minutes * 0.23, text_y_recovery,
                 f"Neutral {neutral_down}", color="grey", fontsize=8)

    # Labels and formatting
    plt.xlabel("Time (minutes)")
    plt.ylabel("Heart Rate (bpm)")
    plt.title("Heart Rate Analysis with Zones and Peaks")
    plt.grid(alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Load and prepare data first (we need max HR for zone calculation)
    print(f"Loading FIT file: {args.file}")
    df = load_fit_file(args.file)
    df = prepare_data(df)

    # Get max HR from dataset
    data_max_hr = df["heart_rate"].dropna().max()

    # Determine zone calculation method
    user_max_hr = None

    if args.karvonen:
        # Karvonen/HRR method
        resting_hr, max_hr = parse_karvonen(args.karvonen)
        zone_boundaries = calculate_zones_hrr(max_hr, resting_hr)
        user_max_hr = max_hr
        print(f"Using Karvonen/HRR method (Resting: {resting_hr}, Max: {max_hr})")

    elif args.zones_age:
        # Age-based calculation
        max_hr = calculate_max_hr_from_age(args.zones_age)
        zone_boundaries = calculate_zones_from_max_hr(max_hr)
        user_max_hr = max_hr
        print(f"Using age-based zones (Age: {args.zones_age}, Estimated Max HR: {max_hr})")

    elif args.max_hr:
        # Simple max HR percentage method
        zone_boundaries = calculate_zones_from_max_hr(args.max_hr)
        user_max_hr = args.max_hr
        print(f"Using percentage method (Max HR: {args.max_hr})")

    else:
        # Manual zones (default if nothing specified)
        zones_str = args.zones if args.zones else '112,124,136,149,161'
        zone_boundaries = parse_zones(zones_str)
        print(f"Using manual zones: {zone_boundaries}")

    # Create zone dictionary with appropriate upper limit
    zones = create_zone_dict(zone_boundaries, user_max_hr=user_max_hr, data_max_hr=data_max_hr)

    # Detect peaks
    peaks, _ = find_peaks(df["heart_rate"], distance=PEAK_DETECTION_DISTANCE)
    print(f"Found {len(peaks)} heart rate peaks")

    # Create visualization (slopes are analyzed within the plot function)
    plot_heart_rate(df, zones, peaks)


if __name__ == "__main__":
    main()
