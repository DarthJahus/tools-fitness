"""
FIT Cycling Analyzer
Analyzes cycling data from FIT files: GPS tracking, effort calculation, and performance metrics.
"""

import fitdecode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import argrelextrema
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
import argparse
import sys
from pathlib import Path

# -------------------------
# Constants
# -------------------------
SLOPE_WINDOW = 10  # Rolling window for slope smoothing
SLOPE_BOUND_DEFAULT = 25  # Default slope clipping bound (±, %)
EXTREMA_ORDER = 30  # Window for local min/max detection
EDGE_PERCENT = 0.05  # Edge percentage for turnaround detection
SEMICIRCLES_TO_DEGREES = 180.0 / 2 ** 31  # FIT file coordinate conversion
OVERLAP_THRESHOLD = 0.0001  # ~11 meters in degrees
MIN_OVERLAP_POINTS = 20
HEADING_WINDOW = 10
SPEED_BOUND_MIN = 0.0
SPEED_BOUND_MAX = 60.0     # considérée comme vitesse extrême possible
WEIGHT_SLOPE = 0.5  # Pondération des facteurs
WEIGHT_SPEED = 0.35
WEIGHT_HR    = 0.15

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
        '--slope-bound',
        type=str,
        help='Slope clipping bounds: default: 50'
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
    colors = ["#cccccc", "#a6a6a6", "#3b82f6", "#10b981", "#f59e0b", "#dc2626"]

    zones = dict()

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


def prepare_data(df, has_gps=True, slope_bound=SLOPE_BOUND_DEFAULT):
    """Prepare and process cycling data with robust slope cleaning and smoothing."""
    # --- Time ---
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["time_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    # --- GPS coordinates ---
    if has_gps and "position_lat" in df.columns and "position_long" in df.columns:
        df["lat_deg"] = pd.to_numeric(df.get("position_lat"), errors="coerce") * SEMICIRCLES_TO_DEGREES
        df["lon_deg"] = pd.to_numeric(df.get("position_long"), errors="coerce") * SEMICIRCLES_TO_DEGREES
    else:
        df["lat_deg"] = np.nan
        df["lon_deg"] = np.nan

    # --- Speed / Distance / Altitude ---
    df["enhanced_speed"] = pd.to_numeric(df.get("enhanced_speed"), errors="coerce")
    df["speed"] = pd.to_numeric(df.get("speed", df["enhanced_speed"]), errors="coerce")
    df["speed_kmh"] = df["speed"] * 3.6
    df["distance"] = pd.to_numeric(df.get("distance"), errors="coerce")
    df["enhanced_altitude"] = pd.to_numeric(df.get("enhanced_altitude"), errors="coerce")
    df["altitude"] = pd.to_numeric(df.get("altitude", df["enhanced_altitude"]), errors="coerce")

    # --- Heart rate & Power ---
    df["heart_rate"] = pd.to_numeric(df.get("heart_rate"), errors="coerce")
    df["power"] = pd.to_numeric(df.get("power"), errors="coerce")

    # --- Calculate raw slope ---
    df["delta_alt"] = df["altitude"].diff()
    df["delta_dist"] = df["distance"].diff()
    df.loc[df["delta_dist"] < 5, "delta_dist"] = np.nan  # ignore trop court
    with np.errstate(divide="ignore", invalid="ignore"):
        df["slope"] = (df["delta_alt"] / df["delta_dist"]) * 100.0
    df.loc[(df["delta_dist"] <= 0) | ~np.isfinite(df["slope"]), "slope"] = np.nan

    # --- Clean extreme slopes ---
    # On supprime les pentes totalement impossibles (physiologiquement ou topographiquement)
    df["slope_clean"] = df["slope"].where(
        df["slope"].between(-slope_bound, slope_bound)
    )

    # --- Smooth slope ---
    # Fenêtre de 20 m en avant/arrière
    slope_smoothed = []
    distances = df["distance"].to_numpy()
    slopes = df["slope_clean"].to_numpy()
    window = 20  # m

    for i, d in enumerate(distances):
        mask = (distances >= d - window) & (distances <= d + window)
        valid = slopes[mask]
        valid = valid[np.isfinite(valid)]
        if len(valid) > 0:
            slope_smoothed.append(valid.mean())
        else:
            slope_smoothed.append(np.nan)

    df["slope_smoothed"] = slope_smoothed

    # --- Final clip ---
    # Sécurité pour éviter que le lissage ne crée des valeurs absurdes
    df["slope_final"] = df["slope_smoothed"].clip(-slope_bound, slope_bound)

    return df


def calculate_effort_hrtss(df, zone_boundaries, zone_values=None,
                           smooth=True, alpha_hr=1.0, default_effort=20):
    """
    Calculate instantaneous hrTSS rate with continuous HR zones.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'heart_rate' column.
    zone_boundaries : list
        HR thresholds defining zones (ascending).
    zone_values : list or None
        Effort values for each zone. If None, use HRTSS_ZONE_MULTIPLIERS defaults.
    smooth : bool
        Apply 15s rolling mean if True.
    alpha_hr : float
        Weight of HR in effort (0 = ignore HR, 1 = full HR influence)
    default_effort : float
        Fallback if HR outside defined zones.

    Returns
    -------
    pd.Series
        Instantaneous hrTSS effort.
    """
    if zone_values is None:
        zone_values = [HRTSS_ZONE_MULTIPLIERS.get(z, default_effort)
                       for z in range(len(zone_boundaries))]

    def hr_zone_continuous(hr):
        for i, bound in enumerate(zone_boundaries):
            if hr <= bound:
                prev_val = zone_values[i - 1] if i > 0 else zone_values[0]
                next_val = zone_values[i]
                prev_bound = zone_boundaries[i - 1] if i > 0 else 0
                return prev_val + (hr - prev_bound) / (bound - prev_bound) * (next_val - prev_val)
        return zone_values[-1]

    hr_effort = df["heart_rate"].apply(hr_zone_continuous)
    hr_effort = hr_effort * alpha_hr + (1 - alpha_hr) * default_effort

    if smooth:
        hr_effort = hr_effort.rolling(window=15, center=True, min_periods=1).mean()

    return hr_effort


def calculate_effort_tss(df, ftp=None):
    """
    Calculate instantaneous TSS rate from power (Coggan-style).

    Returns unitless effort comparable to TSS/h.
    """
    if "power" not in df.columns or df["power"].isna().all():
        print("Warning: No power data available, falling back to hrTSS")
        return None

    if ftp is None:
        ftp = df["power"].dropna().quantile(0.75)
        print(f"Info: Estimated FTP = {ftp:.0f}W")

    # Normalized Power (NP) standard: rolling 30s on raw power
    np_power = (df["power"].rolling(window=30, min_periods=1).mean() ** 4).rolling(window=30,
                                                                                   min_periods=1).mean() ** 0.25
    intensity_factor = np_power / ftp

    # Instantaneous TSS rate (~percent per hour)
    tss_rate = np_power * intensity_factor / (ftp * 36)

    return tss_rate


def calculate_effort_custom(df,
                            slope_weight=WEIGHT_SLOPE,
                            speed_weight=WEIGHT_SPEED,
                            hr_weight=WEIGHT_HR,
                            slope_bound=SLOPE_BOUND_DEFAULT,
                            speed_max=SPEED_BOUND_MAX,
                            user_hr_zones=None,  # liste des zones HR de l'utilisateur
                            smooth=True):
    """
    Effort custom normalisé sur slope, speed et heart_rate,
    basé sur des bornes réalistes et des poids relatifs.
    Si user_hr_zones est fourni, le HR est normalisé via ces zones avec spline.
    L'effort final est rescalé entre 10% et 90%.
    """

    # --- Normalisation pente ---
    slope = df["slope_final"].copy().interpolate(method="spline", order=3).bfill().ffill()
    slope = slope.clip(-slope_bound, slope_bound)
    slope_norm = (slope + slope_bound) / (2 * slope_bound)

    # --- Normalisation vitesse ---
    speed = df["speed_kmh"].copy().interpolate(method="spline", order=3).bfill().ffill()
    speed = speed.clip(SPEED_BOUND_MIN, speed_max)
    speed_norm = (speed - SPEED_BOUND_MIN) / (speed_max - SPEED_BOUND_MIN)

    # --- Normalisation heart rate ---
    hr = df["heart_rate"].copy().interpolate(method="spline", order=3).bfill().ffill()

    if user_hr_zones is not None and len(user_hr_zones) > 0:
        # --- Création des points pour interpolation spline ---
        hr_points = [0] + user_hr_zones
        intensity_points = [ZONE_INTENSITY.get(i, 1.0) for i in range(len(user_hr_zones))]
        intensity_points.append(intensity_points[-1])  # extrapolation pour HR > max zone

        # spline cubique pour une courbe lisse
        spline = CubicSpline(hr_points, intensity_points, extrapolate=True)

        # application spline sur la série HR
        hr_norm = hr.apply(lambda x: float(spline(x)))
    else:
        # fallback : normalisation classique min/max
        hr_min_local = hr.min()
        hr_max_local = hr.max()
        denom_hr = hr_max_local - hr_min_local if hr_max_local != hr_min_local else 1.0
        hr_norm = (hr - hr_min_local) / denom_hr

    # --- Combinaison pondérée ---
    combined = (slope_norm * slope_weight +
                speed_norm * speed_weight +
                hr_norm    * hr_weight)

    # --- Lissage exponentiel (optionnel) ---
    if smooth:
        combined = combined.ewm(span=15, adjust=False).mean()

    # --- Rescaling entre 10% et 90% ---
    combined_min = combined.min()
    combined_max = combined.max()
    denom_combined = combined_max - combined_min if combined_max != combined_min else 1.0
    effort_scaled = 10 + 80 * (combined - combined_min) / denom_combined

    return effort_scaled


def calculate_heading(lat1, lon1, lat2, lon2):
    """Calculate bearing/heading between two points (degrees)."""
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    heading = np.arctan2(y, x)
    return np.degrees(heading) % 360


# ============================================================================
# MÉTHODE 1 : Point le plus éloigné (distance euclidienne)
# ============================================================================

def method_furthest_point(df, edge_pct=EDGE_PERCENT):
    """
    Détecte le demi-tour comme le point le plus éloigné du point de départ
    (distance euclidienne / vol d'oiseau).
    """
    print("\n[MÉTHODE 1] Point le plus éloigné (euclidienne)")

    coords = df[["lat_deg", "lon_deg"]].dropna()
    if len(coords) < 50:
        print("✗ Pas assez de points GPS")
        return None

    coord_indices = coords.index.tolist()
    total_points = len(coord_indices)
    edge_n = max(10, int(total_points * edge_pct))

    # Point de départ
    start_point = coords.iloc[0].values

    # Calculer la distance euclidienne de chaque point au départ
    distances_to_start = np.sqrt(
        (coords.values[:, 0] - start_point[0]) ** 2 +
        (coords.values[:, 1] - start_point[1]) ** 2
    )

    # Exclure les bords
    distances_to_start[:edge_n] = -np.inf
    distances_to_start[-edge_n:] = -np.inf

    # Trouver le maximum
    max_pos = np.argmax(distances_to_start)
    turnaround_idx = coord_indices[max_pos]
    max_dist_km = distances_to_start[max_pos] * 111

    print(f"✓ Demi-tour détecté à l'indice {turnaround_idx}")
    print(f"  Distance au départ : {max_dist_km:.2f} km")

    return turnaround_idx


# ============================================================================
# MÉTHODE 2 : Distance GPS cumulée maximale
# ============================================================================

def method_cumulative_distance(df, edge_pct=EDGE_PERCENT):
    """
    Détecte le demi-tour comme le point où la distance GPS cumulée atteint
    son maximum (ne fonctionne que si la distance décroît au retour).
    """
    print("\n[MÉTHODE 2] Distance GPS cumulée maximale")

    if "distance" not in df.columns or df["distance"].isna().all():
        print("✗ Pas de données de distance")
        return None

    distances = df["distance"].dropna()
    if len(distances) < 50:
        print("✗ Pas assez de points avec distance")
        return None

    edge_n = max(10, int(len(distances) * edge_pct))

    # Exclure les bords
    distances_copy = distances.copy()
    distances_copy.iloc[:edge_n] = -np.inf
    distances_copy.iloc[-edge_n:] = -np.inf

    turnaround_idx = distances_copy.idxmax()
    max_dist_km = distances.loc[turnaround_idx] / 1000

    print(f"✓ Demi-tour détecté à l'indice {turnaround_idx}")
    print(f"  Distance cumulée : {max_dist_km:.2f} km")

    return turnaround_idx


# ============================================================================
# MÉTHODE 3 : Extremum 3D (courbure spatio-temporelle)
# ============================================================================

def method_3d_curvature(df, edge_pct=EDGE_PERCENT):
    """
    Détecte le demi-tour comme le point de courbure maximale dans l'espace
    3D (lat, lon, temps normalisé).
    """
    print("\n[MÉTHODE 3] Extremum 3D (courbure)")

    coords = df[["lat_deg", "lon_deg"]].dropna()
    if len(coords) < 50:
        print("✗ Pas assez de points GPS")
        return None

    coord_indices = coords.index.tolist()
    total_points = len(coord_indices)
    edge_n = max(10, int(total_points * edge_pct))

    # Construire la trajectoire 3D : (x, y, t_normalized)
    x = coords.values[:, 0]
    y = coords.values[:, 1]
    t = np.linspace(0, 1, total_points)  # Temps normalisé [0, 1]

    # Lisser les coordonnées pour réduire le bruit
    x_smooth = gaussian_filter1d(x, sigma=5)
    y_smooth = gaussian_filter1d(y, sigma=5)

    # Calculer les dérivées premières (vitesse)
    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    dt = np.gradient(t)

    # Calculer les dérivées secondes (accélération)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Courbure 3D : magnitude de l'accélération dans le plan (x, y)
    curvature = np.sqrt(ddx ** 2 + ddy ** 2)

    # Exclure les bords
    curvature[:edge_n] = -np.inf
    curvature[-edge_n:] = -np.inf

    # Trouver le maximum de courbure
    max_curv_pos = np.argmax(curvature)
    turnaround_idx = coord_indices[max_curv_pos]

    print(f"✓ Demi-tour détecté à l'indice {turnaround_idx}")
    print(f"  Courbure maximale : {curvature[max_curv_pos]:.6f}")

    return turnaround_idx


# ============================================================================
# MÉTHODE 4 : Hybride (heading + distance)
# ============================================================================

def calculate_moving_heading(df, window=10):
    """Calcule le heading avec une fenêtre mobile."""
    coords = df[["lat_deg", "lon_deg"]].dropna()
    headings = pd.Series(index=coords.index, dtype=float)

    for i in range(len(coords)):
        start = max(0, i - window)
        end = min(len(coords), i + window + 1)

        if end - start < 2:
            continue

        segment = coords.iloc[start:end]
        dlat = segment.iloc[-1, 0] - segment.iloc[0, 0]
        dlon = segment.iloc[-1, 1] - segment.iloc[0, 1]

        heading = np.degrees(np.arctan2(dlon, dlat)) % 360
        headings.iloc[i] = heading

    return headings


def method_hybrid_heading_distance(df, edge_pct=EDGE_PERCENT, heading_threshold=150):
    """
    Détecte le demi-tour en combinant :
    1. Changements de heading significatifs (>150°)
    2. Distance maximale au point de départ parmi les candidats
    3. Vérification du chevauchement de parcours
    """
    print("\n[MÉTHODE 4] Hybride (heading + distance)")

    coords = df[["lat_deg", "lon_deg"]].dropna()
    if len(coords) < 50:
        print("✗ Pas assez de points GPS")
        return None

    coord_indices = coords.index.tolist()
    total_points = len(coord_indices)
    edge_n = max(10, int(total_points * edge_pct))

    # 1. Calculer les headings
    headings = calculate_moving_heading(df, window=HEADING_WINDOW)

    # 2. Détecter les changements de heading
    candidates = []

    for i in range(edge_n, total_points - edge_n):
        idx = coord_indices[i]

        if idx not in headings.index:
            continue

        # Heading avant et après
        pre_idx = coord_indices[max(0, i - HEADING_WINDOW)]
        post_idx = coord_indices[min(total_points - 1, i + HEADING_WINDOW)]

        if pre_idx not in headings.index or post_idx not in headings.index:
            continue

        heading_before = headings.loc[pre_idx]
        heading_after = headings.loc[post_idx]

        if pd.isna([heading_before, heading_after]).any():
            continue

        heading_change = abs(heading_after - heading_before)
        if heading_change > 180:
            heading_change = 360 - heading_change

        if heading_change >= heading_threshold:
            candidates.append({
                'idx': idx,
                'position': i,
                'heading_change': heading_change
            })

    if len(candidates) == 0:
        print("✗ Aucun changement de heading significatif détecté")
        return None

    print(f"  {len(candidates)} candidats avec heading change >{heading_threshold}°")

    # 3. Parmi les candidats, choisir celui le plus éloigné du départ
    start_point = coords.iloc[0].values

    best_candidate = None
    max_distance = -1

    for cand in candidates:
        pos = cand['position']
        point = coords.iloc[pos].values
        dist = np.sqrt((point[0] - start_point[0]) ** 2 + (point[1] - start_point[1]) ** 2)

        if dist > max_distance:
            max_distance = dist
            best_candidate = cand

    turnaround_idx = best_candidate['idx']

    print(f"✓ Demi-tour détecté à l'indice {turnaround_idx}")
    print(f"  Heading change : {best_candidate['heading_change']:.1f}°")
    print(f"  Distance au départ : {max_distance * 111:.2f} km")

    return turnaround_idx


# ============================================================================
# MÉTHODE 5 : Variance de distance aux autres points
# ============================================================================

def method_distance_variance(df, edge_pct=EDGE_PERCENT):
    """
    Détecte le demi-tour comme le point où la variance des distances
    aux autres points change drastiquement.
    """
    print("\n[MÉTHODE 5] Variance de distance")

    coords = df[["lat_deg", "lon_deg"]].dropna()
    if len(coords) < 50:
        print("✗ Pas assez de points GPS")
        return None

    coord_indices = coords.index.tolist()
    total_points = len(coord_indices)
    edge_n = max(10, int(total_points * edge_pct))

    # Calculer la matrice de distances
    distances_matrix = cdist(coords.values, coords.values, metric='euclidean')

    # Pour chaque point, calculer la variance des distances aux autres points
    variances = np.var(distances_matrix, axis=1)

    # Exclure les bords
    variances[:edge_n] = -np.inf
    variances[-edge_n:] = -np.inf

    # Le demi-tour est le point de variance maximale
    max_var_pos = np.argmax(variances)
    turnaround_idx = coord_indices[max_var_pos]

    print(f"✓ Demi-tour détecté à l'indice {turnaround_idx}")
    print(f"  Variance maximale : {variances[max_var_pos]:.6f}")

    return turnaround_idx


# ============================================================================
# MÉTHODE 6 : Inversion de direction temporelle
# ============================================================================

def method_temporal_inversion(df, edge_pct=EDGE_PERCENT):
    """
    Détecte le demi-tour en identifiant le changement de direction temporelle
    des correspondances spatiales (méthode précédente).
    """
    print("\n[MÉTHODE 6] Inversion de direction temporelle")

    coords = df[["lat_deg", "lon_deg"]].dropna()
    if len(coords) < 50:
        print("✗ Pas assez de points GPS")
        return None

    coord_indices = coords.index.tolist()
    total_points = len(coord_indices)
    edge_n = max(10, int(total_points * edge_pct))

    coords_array = coords.values
    distances_matrix = cdist(coords_array, coords_array, metric='euclidean')

    temporal_directions = []

    for i in range(edge_n, total_points - edge_n):
        window = 20
        mask = np.ones(total_points, dtype=bool)
        mask[max(0, i - window):min(total_points, i + window + 1)] = False

        valid_distances = distances_matrix[i].copy()
        valid_distances[~mask] = np.inf

        if np.all(np.isinf(valid_distances)):
            continue

        nearest_idx = np.argmin(valid_distances)
        nearest_distance = valid_distances[nearest_idx]

        if nearest_distance < OVERLAP_THRESHOLD:
            temporal_direction = nearest_idx - i
            temporal_directions.append((i, temporal_direction))

    if len(temporal_directions) == 0:
        print("✗ Aucune correspondance spatiale trouvée")
        return None

    # Détecter le changement de direction
    turnaround_candidates = []

    for idx, (i, direction) in enumerate(temporal_directions[:-1]):
        next_direction = temporal_directions[idx + 1][1]

        if direction > 0 and next_direction < 0:
            turnaround_candidates.append({
                'index': i,
                'score': abs(direction) + abs(next_direction)
            })

    if len(turnaround_candidates) == 0:
        print("✗ Aucun changement de direction temporelle détecté")
        return None

    best_candidate = max(turnaround_candidates, key=lambda x: x['score'])
    turnaround_idx = coord_indices[best_candidate['index']]

    print(f"✓ Demi-tour détecté à l'indice {turnaround_idx}")

    return turnaround_idx

def refine_turn_by_local_curvature(df, start, end):
    coords = df[["lat_deg", "lon_deg"]].iloc[start:end].values
    if len(coords) < 5:
        return None

    x = gaussian_filter1d(coords[:, 0], sigma=3)
    y = gaussian_filter1d(coords[:, 1], sigma=3)

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.sqrt(ddx**2 + ddy**2)

    local_idx = np.argmax(curvature)
    return df.index[start + local_idx]


def get_candidate_window(i0, n, window_size=40):
    start = max(0, i0 - window_size)
    end = min(n, i0 + window_size + 1)
    return start, end


def refine_turn_by_spatial_curvature(df, start, end):
    coords = df[["lat_deg", "lon_deg"]].iloc[start:end].values
    if len(coords) < 5:
        return None

    # recentrage
    coords -= coords.mean(axis=0)

    x = gaussian_filter1d(coords[:, 0], sigma=3)
    y = gaussian_filter1d(coords[:, 1], sigma=3)

    # paramétrisation spatiale
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.concatenate([[0], np.cumsum(ds)])
    if s[-1] == 0:
        return None
    s /= s[-1]

    dx = np.gradient(x, s)
    dy = np.gradient(y, s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-12)**1.5

    local_idx = np.argmax(curvature)
    return df.index[start + local_idx]


def refine_turn_by_spatial_curvature_real(df, start, end):
    """
    Affine le point de demi-tour dans une fenêtre locale en utilisant
    la vraie courbure plane κ(s), paramétrée par la longueur d'arc.
    """
    coords = df[["lat_deg", "lon_deg"]].iloc[start:end].values
    n = len(coords)
    if n < 7:
        return None

    # Recentrage (invariance translation)
    coords = coords - coords.mean(axis=0)

    # Lissage léger (anti-bruit GPS)
    x = gaussian_filter1d(coords[:, 0], sigma=2)
    y = gaussian_filter1d(coords[:, 1], sigma=2)

    # Paramétrisation par longueur d’arc s
    dx_raw = np.diff(x)
    dy_raw = np.diff(y)
    ds = np.sqrt(dx_raw**2 + dy_raw**2)

    if np.all(ds == 0):
        return None

    s = np.concatenate([[0], np.cumsum(ds)])
    s /= s[-1]  # normalisation [0,1]

    # Dérivées par rapport à s
    dx = np.gradient(x, s)
    dy = np.gradient(y, s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)

    # Courbure réelle κ(s)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-12)**1.5

    # Sécurité : ignorer les bords de la fenêtre
    curvature[0] = curvature[-1] = -np.inf

    local_pos = np.argmax(curvature)
    return df.index[start + local_pos]


def method_dynamic_centroid_refined(df, edge_pct=EDGE_PERCENT, window_size=40):
    print("\n[MÉTHODE] Dynamic centroid + affinement local")

    i0 = method_dynamic_centroid(df, edge_pct)
    if i0 is None:
        return None

    coords = df[["lat_deg", "lon_deg"]].dropna()
    idx_list = coords.index.tolist()
    pos = idx_list.index(i0)

    start, end = get_candidate_window(pos, len(idx_list), window_size)

    # refined_idx = refine_turn_by_local_curvature(df, start, end)
    # refined_idx = refine_turn_by_spatial_curvature(df, start, end)
    refined_idx = refine_turn_by_spatial_curvature_real(df, start, end)

    if refined_idx is not None:
        print(f"✓ Affiné : {refined_idx}")
        return refined_idx

    print("⚠ Affinage échoué, retour du point centroid")
    return i0



def method_dynamic_centroid(df, edge_pct=EDGE_PERCENT):
    print("\n[MÉTHODE F] Centroïde dynamique")

    coords = df[["lat_deg", "lon_deg"]].dropna().values
    n = len(coords)
    edge_n = max(10, int(n * edge_pct))

    scores = np.zeros(n)

    for i in range(edge_n, n - edge_n):
        c = coords[:i].mean(axis=0)
        scores[i] = np.linalg.norm(coords[i] - c)

    idx = np.argmax(scores)
    print(f"✓ Demi-tour à {df.index[idx]}")
    return df.index[idx]

def method_pca_extremum(df, edge_pct=EDGE_PERCENT):
    print("\n[MÉTHODE E] Extremum PCA")

    coords = df[["lat_deg", "lon_deg"]].dropna().values
    coords -= coords.mean(axis=0)

    u, s, vh = np.linalg.svd(coords, full_matrices=False)
    proj = u[:, 0] * s[0]

    n = len(proj)
    edge_n = max(10, int(n * edge_pct))
    proj[:edge_n] = -np.inf
    proj[-edge_n:] = -np.inf

    idx = np.argmax(np.abs(proj))
    print(f"✓ Demi-tour à {df.index[idx]}")
    return df.index[idx]

def method_path_symmetry(df, edge_pct=EDGE_PERCENT):
    print("\n[MÉTHODE D] Symétrie de trajectoire")

    coords = df[["lat_deg", "lon_deg"]].dropna().values
    n = len(coords)
    edge_n = max(10, int(n * edge_pct))

    scores = np.full(n, np.inf)

    for i in range(edge_n, n - edge_n):
        a = coords[:i]
        b = coords[i:][::-1]
        m = min(len(a), len(b))
        scores[i] = np.mean(np.linalg.norm(a[-m:] - b[:m], axis=1))

    idx = np.argmin(scores)
    print(f"✓ Demi-tour à {df.index[idx]}")
    return df.index[idx]


def method_nearest_neighbor_inversion(df, edge_pct=EDGE_PERCENT):
    print("\n[MÉTHODE C] Inversion du voisin temporel")

    coords = df[["lat_deg", "lon_deg"]].dropna().values
    n = len(coords)
    edge_n = max(10, int(n * edge_pct))

    mat = cdist(coords, coords)
    np.fill_diagonal(mat, np.inf)

    signs = []

    for i in range(edge_n, n - edge_n):
        j = np.argmin(mat[i])
        signs.append((i, np.sign(j - i)))

    for k in range(len(signs) - 1):
        if signs[k][1] > 0 and signs[k + 1][1] < 0:
            idx = signs[k][0]
            print(f"✓ Demi-tour à {df.index[idx]}")
            return df.index[idx]

    return None


def method_self_overlap(df, edge_pct=EDGE_PERCENT):
    print("\n[MÉTHODE B] Auto-recouvrement spatial")

    coords = df[["lat_deg", "lon_deg"]].dropna().values
    n = len(coords)
    edge_n = max(10, int(n * edge_pct))

    overlap_score = np.zeros(n)

    for i in range(edge_n, n - edge_n):
        past = coords[:i]
        future = coords[i:]
        d = cdist(past, future)
        overlap_score[i] = np.mean(d.min(axis=1))

    overlap_score[:edge_n] = np.inf
    overlap_score[-edge_n:] = np.inf

    idx = np.argmin(overlap_score)
    print(f"✓ Demi-tour à {df.index[idx]}")
    return df.index[idx]


def method_distance_to_past_path(df, edge_pct=EDGE_PERCENT):
    print("\n[MÉTHODE A] Distance au chemin passé")

    coords = df[["lat_deg", "lon_deg"]].dropna().values
    n = len(coords)
    if n < 50:
        return None

    edge_n = max(10, int(n * edge_pct))
    scores = np.zeros(n)

    for i in range(edge_n, n - edge_n):
        past = coords[:i]
        dists = cdist([coords[i]], past)
        scores[i] = np.min(dists)

    scores[:edge_n] = -np.inf
    scores[-edge_n:] = -np.inf

    idx = np.argmax(scores)
    print(f"✓ Demi-tour à {df.index[idx]}")
    return df.index[idx]


def detect_turnaround(df, method="dynamic_centroid_refined", edge_pct=EDGE_PERCENT):
    """
    Détecte le point de demi-tour sur un trajet aller-retour.

    Paramètres :
    -----------
    df : DataFrame
        DataFrame contenant les données GPS (colonnes lat_deg, lon_deg)
    method : str
        Méthode de détection à utiliser :
        - "furthest_point" : Point le plus éloigné du départ (euclidienne)
        - "cumulative_distance" : Distance GPS cumulée maximale
        - "3d_curvature" : Extremum 3D (courbure spatio-temporelle)
        - "hybrid" : Hybride (heading + distance)
        - "variance" : Variance de distance aux autres points
        - "temporal_inversion" : Inversion de direction temporelle
    edge_pct : float
        Pourcentage des extrémités à exclure (défaut: 0.05 = 5%)

    Retourne :
    ---------
    int ou None : Index du point de demi-tour, ou None si non détecté
    """

    methods_map = {
        "furthest_point": method_furthest_point,
        "cumulative_distance": method_cumulative_distance,
        "3d_curvature": method_3d_curvature,
        "hybrid": method_hybrid_heading_distance,
        "variance": method_distance_variance,
        "temporal_inversion": method_temporal_inversion,
        "distance_to_path": method_distance_to_past_path,
        "self_overlap": method_self_overlap,
        "nn_inversion": method_nearest_neighbor_inversion,
        "symmetry": method_path_symmetry,
        "pca": method_pca_extremum,
        "dynamic_centroid": method_dynamic_centroid,
        "dynamic_centroid_refined": method_dynamic_centroid_refined
    }

    if method not in methods_map:
        available = ", ".join(methods_map.keys())
        raise ValueError(f"Méthode '{method}' inconnue. Disponibles : {available}")

    print(f"\n{'=' * 60}")
    print(f"DÉTECTION DE DEMI-TOUR")
    print(f"{'=' * 60}")

    # Appeler la méthode choisie
    turnaround_idx = methods_map[method](df, edge_pct)

    # Afficher des informations complémentaires si détecté
    if turnaround_idx is not None:
        coords = df[["lat_deg", "lon_deg"]].dropna()
        coord_indices = coords.index.tolist()

        if turnaround_idx in coord_indices:
            position = coord_indices.index(turnaround_idx)
            total = len(coord_indices)

            print(f"\n{'=' * 60}")
            print(f"Position dans le parcours : {position}/{total} ({position / total * 100:.1f}%)")

            # Distance au départ et à l'arrivée
            start_point = coords.iloc[0].values
            end_point = coords.iloc[-1].values
            turn_point = coords.loc[turnaround_idx].values

            dist_to_start = np.linalg.norm(turn_point - start_point) * 111
            dist_to_end = np.linalg.norm(turn_point - end_point) * 111

            print(f"Distance au départ : {dist_to_start:.2f} km")
            print(f"Distance à l'arrivée : {dist_to_end:.2f} km")

            # Distance départ-arrivée
            start_to_end = np.linalg.norm(end_point - start_point) * 111
            print(f"Distance départ → arrivée : {start_to_end:.2f} km")
            print(f"{'=' * 60}\n")

    return turnaround_idx


def detect_extrema(series, order=EXTREMA_ORDER):
    """Detect local minima and maxima in a series."""
    values = series.values
    minima = argrelextrema(values, np.less_equal, order=order)[0]
    maxima = argrelextrema(values, np.greater_equal, order=order)[0]
    return minima, maxima


def plot_gps_map(df, zones, turnaround_idx=None, save_path=None):
    """Plot GPS map with corrected layout."""
    gps_ok = df[["lon_deg", "lat_deg"]].dropna()

    if len(gps_ok) == 0:
        print("Warning: No GPS data available, skipping GPS map")
        return

    if turnaround_idx is not None:
        fig = plt.figure(figsize=(16, 6))
        # Manually position subplots to leave room for colorbar
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        axes = [ax1, ax2]

        df_outbound = df.loc[:turnaround_idx].copy()
        df_return = df.loc[turnaround_idx:].copy()
        turnaround_lat = df.loc[turnaround_idx, "lat_deg"]
        turnaround_lon = df.loc[turnaround_idx, "lon_deg"]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        axes = [axes]

    lon_min, lon_max = gps_ok["lon_deg"].min(), gps_ok["lon_deg"].max()
    lat_min, lat_max = gps_ok["lat_deg"].min(), gps_ok["lat_deg"].max()
    pad_lon = (lon_max - lon_min) * 0.05
    pad_lat = (lat_max - lat_min) * 0.05

    if turnaround_idx is not None:
        sc1 = axes[0].scatter(
            df_outbound["lon_deg"], df_outbound["lat_deg"],
            c=df_outbound["slope"], cmap="coolwarm",
            s=12, alpha=0.8, vmin=-15, vmax=15
        )
        axes[0].scatter(turnaround_lon, turnaround_lat,
                        color="black", marker="X", s=150, label="Turnaround",
                        zorder=5, edgecolors='white', linewidths=1.5)
        axes[0].set_title("Outbound", fontsize=12, pad=10)
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlim(lon_min - pad_lon, lon_max + pad_lon)
        axes[0].set_ylim(lat_min - pad_lat, lat_max + pad_lat)
        axes[0].legend(loc='best')
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        sc2 = axes[1].scatter(
            df_return["lon_deg"], df_return["lat_deg"],
            c=df_return["slope"], cmap="coolwarm",
            s=12, alpha=0.8, vmin=-15, vmax=15
        )
        axes[1].scatter(turnaround_lon, turnaround_lat,
                        color="black", marker="X", s=150, label="Turnaround",
                        zorder=5, edgecolors='white', linewidths=1.5)
        axes[1].set_title("Return", fontsize=12, pad=10)
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].set_xlim(lon_min - pad_lon, lon_max + pad_lon)
        axes[1].set_ylim(lat_min - pad_lat, lat_max + pad_lat)
        axes[1].legend(loc='best')
        axes[1].set_xlabel("Longitude")
        axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        fig.suptitle("GPS Route - Outbound vs Return (color = slope %)", fontsize=14, y=0.98)

        # Adjust subplot positions to make room for colorbar
        fig.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.08, wspace=0.15)

        # Add colorbar in remaining space
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(sc1, cax=cbar_ax, label="Slope (%)")
    else:
        sc = axes[0].scatter(
            df["lon_deg"], df["lat_deg"],
            c=df["slope"], cmap="coolwarm",
            s=12, alpha=0.8, vmin=-15, vmax=15
        )
        axes[0].set_title("GPS Route (color = slope %)")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlim(lon_min - pad_lon, lon_max + pad_lon)
        axes[0].set_ylim(lat_min - pad_lat, lat_max + pad_lat)
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        fig.colorbar(sc, ax=axes[0], fraction=0.03, pad=0.04, label="Slope (%)")
        plt.tight_layout()

    if save_path:
        plt.savefig(Path(save_path) / "gps_map.png", dpi=150, bbox_inches='tight')
    plt.show()

    # 6. Dans main(), REMPLACER LA DERNIÈRE LIGNE du summary par:
    # Au lieu de: print(f"Total Effort: {df['effort'].iloc[-1]:.1f}")
    # Utiliser:
    if not df["effort"].isna().all():
        print(f"Avg Effort: {df['effort'].mean():.1f}")
        # print(f"Max Effort: {df['effort'].max():.1f}")  # affichera toujours 90


def plot_metrics_stack(df, zones, extrema_order=EXTREMA_ORDER, save_path=None):
    """Plot stacked metrics with local extrema detection (but not on effort)."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Assign zone colors
    df["zone_color"] = df["heart_rate"].apply(lambda hr: get_zone_color(hr, zones))

    series_info = [
        ("speed_kmh", "Speed (km/h)", "tab:blue", True),
        ("effort", "Effort", "tab:purple", False),  # No extrema for effort
        ("slope_final", "Slope (%)", "tab:green", True),
        ("heart_rate", "Heart Rate (bpm)", None, True),  # Will use zone colors

    ]

    for ax, (col, label, color, detect_extrema_flag) in zip(axes, series_info):
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

        # Detect extrema only if flag is True
        if detect_extrema_flag:
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
    slope_bound = SLOPE_BOUND_DEFAULT

    if args.slope_bound:
        try:
            slope_bound = int(args.slope_bound)
        except:
            pass

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
    df = prepare_data(df, has_gps=has_gps, slope_bound=slope_bound)

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
        effort = calculate_effort_custom(df)

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
        print(f"Avg Effort: {df['effort'].mean():.1f}")
        print(f"Max Effort: {df['effort'].max():.1f}")


if __name__ == "__main__":
    main()
