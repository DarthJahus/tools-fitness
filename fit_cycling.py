import fitdecode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.signal import argrelextrema


# -------------------------
# Chargement du fichier FIT
# -------------------------
file_path = "20419063508_ACTIVITY.fit"

records = []
with fitdecode.FitReader(file_path) as fit:
    for frame in fit:
        if frame.frame_type == fitdecode.FIT_FRAME_DATA and frame.name == "record":
            record = {field.name: field.value for field in frame.fields}
            records.append(record)

df = pd.DataFrame(records)

# -------------------------
# Préparation et conversions
# -------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["time_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

# coords
df["lat_deg"] = pd.to_numeric(df.get("position_lat"), errors="coerce") * (180.0 / 2**31)
df["lon_deg"] = pd.to_numeric(df.get("position_long"), errors="coerce") * (180.0 / 2**31)

# vitesse / distance / altitude / HR
df["enhanced_speed"] = pd.to_numeric(df.get("enhanced_speed"), errors="coerce")
df["speed"] = df["enhanced_speed"]  # m/s
df["speed_kmh"] = df["speed"] * 3.6
df["distance"] = pd.to_numeric(df.get("distance"), errors="coerce")
df["enhanced_altitude"] = pd.to_numeric(df.get("enhanced_altitude"), errors="coerce")
df["heart_rate"] = pd.to_numeric(df.get("heart_rate"), errors="coerce")

# -------------------------
# Calcul pente (Δalt / Δdist en %)
# -------------------------
df["delta_alt"] = df["enhanced_altitude"].diff()
df["delta_dist"] = df["distance"].diff()
with np.errstate(divide="ignore", invalid="ignore"):
    df["slope"] = (df["delta_alt"] / df["delta_dist"]) * 100.0
df.loc[(df["delta_dist"] <= 0) | ~np.isfinite(df["slope"]), "slope"] = np.nan
df["slope"] = df["slope"].rolling(window=5, center=True, min_periods=1).mean()

# -------------------------
# Effort = HR × (1 + pente/100) × (1 + vitesse/10)
# pente limitée [-10, 20]
# -------------------------
df["effort"] = df["heart_rate"] * (1 + df["slope"].clip(-10, 20) / 100.0) * (1 + df["speed"] / 10.0)

# ---------------- Détection du demi-tour simplifiée ----------------
def cumulative_distance(positions):
    """positions = Nx3 array (lat, lon, time) normalisé"""
    diffs = np.diff(positions, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    return cum_dists

def detect_demi_tour(df, edge_pct=0.05):
    n = len(df)
    edge_n = max(1, int(n*edge_pct))
    
    # Départ = point commun dans les 2% premiers/derniers
    start_points = df[['lat_deg','lon_deg','time_s']].iloc[:edge_n].values
    end_points   = df[['lat_deg','lon_deg','time_s']].iloc[-edge_n:].values
    dists = np.linalg.norm(start_points[:, :2][:,None,:] - end_points[:, :2][None,:,:], axis=2)
    min_idx = np.unravel_index(np.argmin(dists), dists.shape)
    unified_point = (start_points[min_idx[0]] + end_points[min_idx[1]]) / 2

    # Tronquer le DataFrame
    start_idx = np.argmin(np.linalg.norm(df[['lat_deg','lon_deg']].iloc[:edge_n].values - unified_point[:2], axis=1))
    end_idx   = n - edge_n + np.argmin(np.linalg.norm(df[['lat_deg','lon_deg']].iloc[-edge_n:].values - unified_point[:2], axis=1))
    df_trunc = df.iloc[start_idx:end_idx+1].copy()

    # Normalisation axes pour distances 3D
    positions = df_trunc[['lat_deg','lon_deg','time_s']].values
    min_vals = positions.min(axis=0)
    max_vals = positions.max(axis=0)
    positions_norm = (positions - min_vals) / (max_vals - min_vals + 1e-9)

    # Distance cumulée depuis début et depuis fin
    cum_start = cumulative_distance(positions_norm)
    cum_end   = cumulative_distance(positions_norm[::-1])[::-1]

    # Point demi-tour = min de l’écart absolu
    score = -np.abs(cum_start - cum_end)
    demi_idx_trunc = np.argmax(score)

    demi_tour_idx = df_trunc.index[demi_idx_trunc]
    demi_lat = df.loc[demi_tour_idx,'lat_deg']
    demi_lon = df.loc[demi_tour_idx,'lon_deg']
    
    return demi_tour_idx, demi_lat, demi_lon
    
#--------------------------------------------------------------------------------

# -------------------------
# Application
# -------------------------
demi_tour_idx, demi_lat, demi_lon = detect_demi_tour(df)


# Découpage en Aller / Retour
df_aller  = df.loc[:demi_tour_idx].copy()
df_retour = df.loc[demi_tour_idx:].copy()



# -------------------------
# VISUALISATION GPS Aller/Retour
# -------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Bornes de la carte
gps_ok = df[["lon_deg", "lat_deg"]].dropna()
lon_min, lon_max = gps_ok["lon_deg"].min(), gps_ok["lon_deg"].max()
lat_min, lat_max = gps_ok["lat_deg"].min(), gps_ok["lat_deg"].max()
pad_lon, pad_lat = (lon_max - lon_min) * 0.02, (lat_max - lat_min) * 0.02

# Tracé Aller
sc1 = axes[0].scatter(df_aller["lon_deg"], df_aller["lat_deg"],
                      c=df_aller["slope"], cmap="coolwarm",
                      s=6, alpha=0.9, vmin=-15, vmax=15)
axes[0].scatter(demi_lon, demi_lat, color="black", marker="x", s=60, label="Demi-tour")
axes[0].set_title("Aller")
axes[0].set_aspect("equal", adjustable="box")
axes[0].set_xlim(lon_min - pad_lon, lon_max + pad_lon)
axes[0].set_ylim(lat_min - pad_lat, lat_max + pad_lat)
axes[0].legend()

# Tracé Retour
sc2 = axes[1].scatter(df_retour["lon_deg"], df_retour["lat_deg"],
                      c=df_retour["slope"], cmap="coolwarm",
                      s=6, alpha=0.9, vmin=-15, vmax=15)
axes[1].scatter(demi_lon, demi_lat, color="black", marker="x", s=60, label="Demi-tour")
axes[1].set_title("Retour")
axes[1].set_aspect("equal", adjustable="box")
axes[1].set_xlim(lon_min - pad_lon, lon_max + pad_lon)
axes[1].set_ylim(lat_min - pad_lat, lat_max + pad_lat)
axes[1].legend()

# Mise en forme
fig.suptitle("Parcours GPS — Aller vs Retour (couleur = pente %)")
fig.subplots_adjust(wspace=0.04, left=0.06, right=0.92, top=0.94, bottom=0.06)
fig.colorbar(sc1, ax=axes.ravel().tolist(), fraction=0.04, pad=0.02, label="Pente (%)")
plt.show()


# --- Graphique vitesse, pente, FC, effort (4 subplots empilés) ---
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

series_info = [
    ("speed_kmh", "Vitesse (km/h)", "tab:blue"),
    ("slope", "Pente (%)", "tab:green"),
    ("heart_rate", "FC (bpm)", "tab:red"),
    ("effort", "Effort (ind.)", "tab:purple"),
]

for ax, (col, label, color) in zip(axes, series_info):
    y = df[col].values
    x = df["time_s"].values
    
    # tracer la courbe principale
    ax.plot(x, y, color=color, label=label)
    ax.set_ylabel(label, color=color)

    # détecter minima et maxima locaux
    minima = argrelextrema(y, np.less_equal, order=30)[0]
    maxima = argrelextrema(y, np.greater_equal, order=30)[0]

    # ajouter marqueurs et lignes verticales
    for idx in minima:
        ax.plot(x[idx], y[idx], "v", color=color, markersize=8)  # triangle bas
        for ax_all in axes:
            ax_all.axvline(x[idx], color=color, alpha=0.2, linestyle="--")

    for idx in maxima:
        ax.plot(x[idx], y[idx], "^", color=color, markersize=8)  # triangle haut
        for ax_all in axes:
            ax_all.axvline(x[idx], color=color, alpha=0.2, linestyle="--")



axes[-1].set_xlabel("Temps (s)")
axes[0].set_title("Évolution des variables avec repères min/max locaux")

plt.tight_layout()
plt.show()

# ---------- FIGURE 3 : Corrélation Vitesse ↔ FC (KDE + scatter coloré par pente + régression) ----------
plt.figure(figsize=(8, 6))
corr_df = df[["speed_kmh", "heart_rate", "slope"]].dropna()
if not corr_df.empty:
    x = corr_df["speed_kmh"]
    y = corr_df["heart_rate"]
    s_c = corr_df["slope"].clip(-15, 15)

    sns.kdeplot(x=x, y=y, fill=True, cmap="Blues", alpha=0.35, levels=10, thresh=0.1)
    sc = plt.scatter(x, y, c=s_c, cmap="coolwarm", s=10, alpha=0.65, vmin=-15, vmax=15)
    sns.regplot(x=x, y=y, scatter=False, color="black", line_kws={"linewidth": 2, "alpha": 0.85})

    # ajuster bornes aux min/max (padding 5%)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
        dx = (xmax - xmin) * 0.05
        plt.xlim(xmin - dx, xmax + dx)
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        dy = (ymax - ymin) * 0.05
        plt.ylim(ymin - dy, ymax + dy)

    plt.xlabel("Vitesse (km/h)")
    plt.ylabel("Fréquence cardiaque (bpm)")
    plt.title("Corrélation Vitesse ↔ FC (couleur = pente)")
    plt.colorbar(sc, label="Pente (%)")
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---------- FIGURE 4 : GPS 3D (Z=temps, couleur=effort) ----------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
mask3d = df[["lon_deg", "lat_deg", "time_s", "effort"]].dropna().index
lon3d, lat3d, t3d, eff3d = df.loc[mask3d, "lon_deg"], df.loc[mask3d, "lat_deg"], df.loc[mask3d, "time_s"], df.loc[mask3d, "effort"]
p = ax.scatter(lon3d, lat3d, t3d, c=eff3d, cmap="plasma", s=6)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Temps (s)")
ax.set_title("Parcours GPS 3D — Z = temps, couleur = effort")
fig.colorbar(p, label="Effort (ind.)", shrink=0.6, pad=0.08)
plt.tight_layout()
plt.show()
