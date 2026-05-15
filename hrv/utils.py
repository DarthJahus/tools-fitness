import csv
from datetime import datetime
from typing import Optional, TextIO
from constants import DEFAULT_ZONE_BOUNDARIES

_log_file: Optional[TextIO] = None


def init_log(path: str) -> None:
    """Open the log file. Call once from main before any log() call."""
    global _log_file
    _log_file = open(path, "w", encoding="utf-8")


def close_log() -> None:
    global _log_file
    if _log_file is not None:
        try:
            _log_file.close()
        except Exception:
            pass
        _log_file = None


def log(msg: str = "") -> None:
    print(msg, flush=True)
    if _log_file is not None:
        try:
            _log_file.write(msg + "\n")
            _log_file.flush()
        except Exception:
            pass


def parse_ts(s: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse timestamp: {s!r}")


def read_csv(filepath: str) -> list:
    with open(filepath, newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter=";")
        rows = [x for x in r if not (x and x[0].startswith("#"))]
    return rows[1:]


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
        raise


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
        raise


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
    Create zone dictionary from boundaries with Garmin naming conventions.
    """
    # Labels corresponding to Garmin's terminology + Rest
    labels = [
        "Rest/Recovery",
        "Zone 1 (Warm Up)",
        "Zone 2 (Easy)",
        "Zone 3 (Aerobic)",
        "Zone 4 (Threshold)",
        "Zone 5 (Maximum)"
    ]
    colors = ["lightgrey", "lightblue", "green", "orange", "red", "purple"]
    zones = {}

    # Calculate upper limit for zone 5
    zone5_start = zone_boundaries[4] + 1
    if user_max_hr is not None:
        zone5_upper = user_max_hr
    else:
        zone5_upper = max(data_max_hr if data_max_hr else zone5_start, zone5_start) + 10

    # Zone 0: Below Zone 1 upper boundary
    zones[f"{labels[0]} (0-{zone_boundaries[0]})"] = (0, zone_boundaries[0], colors[0])

    # Zones 1-4: Between boundaries
    for i in range(len(zone_boundaries) - 1):
        low = zone_boundaries[i] + 1
        high = zone_boundaries[i + 1]
        zones[f"{labels[i+1]} ({low}-{high})"] = (low, high, colors[i+1])

    # Zone 5: Maximum
    if user_max_hr is not None:
        zones[f"{labels[5]} ({zone5_start}-{zone5_upper})"] = (zone5_start, zone5_upper, colors[5])
    else:
        zones[f"{labels[5]} ({zone5_start}+)"] = (zone5_start, zone5_upper, colors[5])

    return zones


def get_zones(args, data_max_hr=None):
    if args.zones:
        boundaries = parse_zones(args.zones)
        return create_zone_dict(boundaries, data_max_hr=data_max_hr)
    elif args.max_hr:
        boundaries = calculate_zones_from_max_hr(args.max_hr)
        return create_zone_dict(boundaries, user_max_hr=args.max_hr)
    elif args.karvonen:
        resting, max_hr = parse_karvonen(args.karvonen)
        boundaries = calculate_zones_hrr(max_hr, resting)
        return create_zone_dict(boundaries, user_max_hr=max_hr)
    elif args.zones_age:
        max_hr = calculate_max_hr_from_age(args.zones_age)
        boundaries = calculate_zones_from_max_hr(max_hr)
        return create_zone_dict(boundaries, user_max_hr=max_hr)
    else:
        return create_zone_dict(DEFAULT_ZONE_BOUNDARIES, data_max_hr=data_max_hr)
