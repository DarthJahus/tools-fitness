import pandas as pd
import numpy as np
EDGE_PERCENT = 0.05
DEFAULT_METHOD = "dynamic_centroid_refined"
HEADING_WINDOW = 10
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d


def calculate_heading(lat1, lon1, lat2, lon2):
    """Calculate bearing/heading between two points (degrees)."""
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    heading = np.arctan2(y, x)
    return np.degrees(heading) % 360

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

    curvature = np.sqrt(ddx ** 2 + ddy ** 2)

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
    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    s = np.concatenate([[0], np.cumsum(ds)])
    if s[-1] == 0:
        return None
    s /= s[-1]

    dx = np.gradient(x, s)
    dy = np.gradient(y, s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2 + 1e-12) ** 1.5

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
    ds = np.sqrt(dx_raw ** 2 + dy_raw ** 2)

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
    curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2 + 1e-12) ** 1.5

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


def get_turnaround_method(method=None):
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
    if method:
        if method not in methods_map:
            return False
        else:
            return methods_map[method]
    else:
        return methods_map.keys()


def detect_turnaround_method(df, edge_pct=EDGE_PERCENT, method=DEFAULT_METHOD):
    try:
        return get_turnaround_method(method)(df, edge_pct)
    except:
        raise ValueError(f"Méthode '{method}' inconnue. Disponibles : {', '.join(get_turnaround_method())}")
