import fitdecode
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sys import argv

__no_mini_label = True

# -------------------------
# Chargement du fichier FIT
# -------------------------
try:
    assert len(argv) > 1
except:
    exit(1)

file_path = argv[1]  # remplace par ton nom de fichier

records = []
with fitdecode.FitReader(file_path) as fit:
    for frame in fit:
        if frame.frame_type == fitdecode.FIT_FRAME_DATA and frame.name == "record":
            record = {field.name: field.value for field in frame.fields}
            records.append(record)

df = pd.DataFrame(records)

# -------------------------
# Préparation
# -------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["time_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
df["heart_rate"] = pd.to_numeric(df.get("heart_rate"), errors="coerce")

# -------------------------
# Zones cardiaques
# -------------------------
zones = {
    "Zone 1 (112-124)": (112, 124, "lightblue"),
    "Zone 2 (125-136)": (125, 136, "green"),
    "Zone 3 (137-149)": (137, 149, "orange"),
    "Zone 4 (150-161)": (150, 161, "red"),
    "Zone 5 (>162)": (162, 300, "purple"),
}

def get_zone_color(hr):
    for z, (low, high, color) in zones.items():
        if low <= hr <= high:
            return color
    return "grey"

df["zone_color"] = df["heart_rate"].apply(get_zone_color)

# -------------------------
# Détection des pics (max locaux)
# -------------------------
peaks, _ = find_peaks(df["heart_rate"], distance=60)  # "distance" = séparation min entre pics

# -------------------------
# Graphique FC avec zones + pics
# -------------------------
plt.figure(figsize=(12, 6))
plt.scatter(df["time_s"]/60, df["heart_rate"], c=df["zone_color"], s=8)

# ajout des zones en fond
for label, (low, high, color) in zones.items():
    plt.axhspan(low, high, color=color, alpha=0.1, label=label)

# ajout des pics
plt.scatter(df["time_s"].iloc[peaks]/60, df["heart_rate"].iloc[peaks] + 2,
            color="black", marker="v", s=20, label="Pics FC", alpha=0.8)

# annotation des pics
for i in peaks:
    plt.text(df["time_s"].iloc[i]/60, df["heart_rate"].iloc[i] + 3,
             str(df["heart_rate"].iloc[i]), color="black", fontsize=6, va="bottom", ha="center")

# -------------------------
# Analyse séparée des pentes de montée et descente
# -------------------------
adapt_monte = fatigue_monte = neutre_monte = 0
adapt_desc = fatigue_desc = neutre_desc = 0

for i in peaks:
    if 10 < i < len(df) - 10:
        # Pente de montée
        t_monte = df["time_s"].iloc[i] - df["time_s"].iloc[i-10]
        delta_fc_monte = df["heart_rate"].iloc[i] - df["heart_rate"].iloc[i-10]
        pente_monte = delta_fc_monte / t_monte if t_monte != 0 else 0

        # Pente de descente
        t_descend = df["time_s"].iloc[i+10] - df["time_s"].iloc[i]
        delta_fc_descend = df["heart_rate"].iloc[i+10] - df["heart_rate"].iloc[i]
        pente_descend = delta_fc_descend / t_descend if t_descend != 0 else 0

        # --- Montée ---
        if pente_monte > 0.6:
            color_up = "red"      # montée très rapide : forte sollicitation → fatigue potentielle
            fatigue_monte += 1
        elif 0.2 <= pente_monte <= 0.6:
            color_up = "green"    # montée contrôlée → bonne adaptation
            adapt_monte += 1
        else:
            color_up = "grey"     # neutre / peu significatif
            neutre_monte += 1

        plt.plot(df["time_s"].iloc[i-10:i]/60,
                 df["heart_rate"].iloc[i-10:i],
                 color=color_up, linewidth=1.8)

        # --- Descente ---
        if pente_descend < -0.5:
            color_down = "green"  # descente rapide → bonne récupération
            adapt_desc += 1
        elif pente_descend > -0.2:
            color_down = "orange" # descente lente → fatigue ou mauvaise récupération
            fatigue_desc += 1
        else:
            color_down = "grey"
            neutre_desc += 1

        plt.plot(df["time_s"].iloc[i:i+10]/60,
                 df["heart_rate"].iloc[i:i+10],
                 color=color_down, linewidth=1.8)

# Y-Limit
fc_valid = df["heart_rate"].dropna()
fc_max = fc_valid.max()
fc_min = fc_valid[fc_valid > 90].min()
plt.ylim(fc_min * 0.9, fc_max * 1.1)  # ±10 %

if not __no_mini_label:
    #-------------------------
    # Résumé textuel clair et coloré
    #-------------------------
    y_text = fc_max * 0.98
    x_text = df["time_s"].iloc[-1] / 60 * 0.7
    dy = (fc_max - fc_min) * 0.04

    # Montées
    plt.text(x_text, y_text,
             "Montées :", fontsize=8, fontweight="bold",
             bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    plt.text(x_text + 8/2, y_text, f"Adapt {adapt_monte}", color="green", fontsize=8)
    plt.text(x_text + 18/2, y_text, f"Fatigue {fatigue_monte}", color="red", fontsize=8)
    plt.text(x_text + 30/2, y_text, f"Neutre {neutre_monte}", color="grey", fontsize=8)

    # Descentes
    plt.text(x_text, y_text - dy,
             "Descentes :", fontsize=8, fontweight="bold",
             bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    plt.text(x_text + 8/2, y_text - dy, f"Adapt {adapt_desc}", color="green", fontsize=8)
    plt.text(x_text + 18/2, y_text - dy, f"Fatigue {fatigue_desc}", color="orange", fontsize=8)
    plt.text(x_text + 30/2, y_text - dy, f"Neutre {neutre_desc}", color="grey", fontsize=8)


plt.xlabel("Temps (minutes)")
plt.ylabel("Fréquence cardiaque (bpm)")
plt.title("Fréquence cardiaque avec zones et pics")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
