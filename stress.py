#!/usr/bin/env python3
"""
Script d'analyse des données de stress Garmin
Usage:
  Mode normal (toutes les données) : python stress.py --ma 7 --draw all --source ./DI_CONNECT/DI-Connect-Aggregator
  Mode range : python stress.py --ma 7 --range 2024-12-31,-90 --source ./DI_CONNECT/DI-Connect-Aggregator
  Mode comparaison : python stress.py --ma 7 --compare 2024-01-01,2025-01-01,90 --source ./DI_CONNECT/DI-Connect-Aggregator

  Note: Les longueurs négatives comptent en arrière depuis la date de référence
"""

import argparse
import json
import glob
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyse des données de stress Garmin')
    parser.add_argument('--ma', type=int, default=7,
                        help='Nombre de jours pour la moyenne mobile (défaut: 7)')
    parser.add_argument('--draw', type=str, default='all',
                        help='Lignes à dessiner: all, sleep, awake, avg, ou combinaison séparée par des virgules (ex: sleep,awake)')
    parser.add_argument('--compare', type=str, default=None,
                        help='Comparer des périodes: START_DAY_1,START_DAY_2,LENGTH (format: YYYY-MM-DD,YYYY-MM-DD,30 ou YYYY-MM-DD,YYYY-MM-DD,-90 pour 90j en arrière)')
    parser.add_argument('--range', type=str, default=None,
                        help='Filtrer une période spécifique: START_DAY,LENGTH (format: YYYY-MM-DD,30 ou YYYY-MM-DD,-90 pour 90j en arrière)')
    parser.add_argument('--source', type=str, required=True,
                        help='Dossier contenant les fichiers UDSFile_*.json')
    return parser.parse_args()


def read_uds_files(source_folder):
    """Lit tous les fichiers UDSFile_*.json dans le dossier source"""
    pattern = os.path.join(source_folder, 'UDSFile_*.json')
    files = glob.glob(pattern)

    if not files:
        print(f"❌ Aucun fichier UDSFile trouvé dans {source_folder}")
        sys.exit(1)

    print(f"📂 {len(files)} fichiers trouvés")

    all_data = []
    for file_path in sorted(files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
        except Exception as e:
            print(f"⚠️  Erreur lors de la lecture de {os.path.basename(file_path)}: {e}")

    print(f"✅ {len(all_data)} jours de données chargés")
    return all_data


def extract_stress_data(raw_data):
    """Extrait les données de stress depuis les données brutes"""
    stress_data = []

    for day in raw_data:
        if 'calendarDate' not in day or 'allDayStress' not in day:
            continue

        date = datetime.strptime(day['calendarDate'], '%Y-%m-%d')
        all_day_stress = day['allDayStress']

        if 'aggregatorList' not in all_day_stress:
            continue

        # Initialiser les valeurs
        total_stress = None
        awake_stress = None
        asleep_stress = None

        # Extraire les données de stress
        for aggregator in all_day_stress['aggregatorList']:
            stress_type = aggregator.get('type')
            avg_level = aggregator.get('averageStressLevel')

            if avg_level is None or avg_level < 0:  # -2 signifie pas de données
                continue

            if stress_type == 'TOTAL':
                total_stress = avg_level
            elif stress_type == 'AWAKE':
                awake_stress = avg_level
            elif stress_type == 'ASLEEP':
                asleep_stress = avg_level

        stress_data.append({
            'date': date,
            'total': total_stress,
            'awake': awake_stress,
            'sleep': asleep_stress
        })

    return pd.DataFrame(stress_data).sort_values('date').reset_index(drop=True)


def detect_data_gaps(df, min_gap_days=3):
    """Détecte les trous dans les données (périodes sans données consécutives)"""
    if df.empty or len(df) < 2:
        return []

    gaps = []
    dates = df['date'].sort_values().reset_index(drop=True)

    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        gap_days = (next_date - current_date).days - 1  # -1, car on veut le nombre de jours entre les deux

        if gap_days >= min_gap_days:
            gap_start = current_date + timedelta(days=1)
            gap_end = next_date - timedelta(days=1)
            gaps.append({
                'start': gap_start,
                'end': gap_end,
                'days': gap_days
            })

    return gaps


def apply_moving_average(df, window):
    """Applique une moyenne mobile aux données de stress"""
    if window > 1:
        for col in ['total', 'awake', 'sleep']:
            if col in df.columns:
                # min_periods=1 permet de continuer la MA même avec des données manquantes
                # Elle sera calculée sur les valeurs disponibles dans la fenêtre
                df[f'{col}_ma'] = df[col].rolling(window=window, center=False, min_periods=1).mean()
    else:
        for col in ['total', 'awake', 'sleep']:
            if col in df.columns:
                df[f'{col}_ma'] = df[col]
    return df


def filter_period(df, reference_date, length_days):
    """
    Filtre le dataframe pour une période donnée
    Si length_days > 0: période de reference_date à reference_date + length_days
    Si length_days < 0: période de reference_date + length_days à reference_date
    Retourne: (filtered_df, start_date, end_date)
    """
    reference = pd.to_datetime(reference_date)

    if length_days >= 0:
        # Mode normal: on avance dans le temps
        start = reference
        end = reference + timedelta(days=length_days)
    else:
        # Mode arrière: on recule dans le temps
        start = reference + timedelta(days=length_days)  # length_days est négatif, donc on recule
        end = reference

    mask = (df['date'] >= start) & (df['date'] < end)
    filtered = df[mask].copy()
    # Créer une colonne 'day_offset' pour l'alignement des comparaisons
    filtered['day_offset'] = (filtered['date'] - start).dt.days
    return filtered, start, end


def print_statistics(df, period_name=None):
    """Affiche les statistiques pour une période et retourne les moyennes"""
    prefix = f"   [{period_name}] " if period_name else "   "

    stats = {}
    if 'total' in df.columns and df['total'].notna().any():
        stats['total'] = df['total'].mean()
        print(f"{prefix}Stress moyen (total): {stats['total']:.1f}")
    if 'awake' in df.columns and df['awake'].notna().any():
        stats['awake'] = df['awake'].mean()
        print(f"{prefix}Stress moyen (éveillé): {stats['awake']:.1f}")
    if 'sleep' in df.columns and df['sleep'].notna().any():
        stats['sleep'] = df['sleep'].mean()
        print(f"{prefix}Stress moyen (sommeil): {stats['sleep']:.1f}")

    return stats


def print_comparison_statistics(period1_df, period2_df):
    """Affiche les statistiques comparées entre deux périodes"""
    print("\n📊 Statistiques comparées:")

    # Calculer les moyennes
    stats1 = {}
    stats2 = {}

    if 'total' in period1_df.columns and period1_df['total'].notna().any():
        stats1['total'] = period1_df['total'].mean()
    if 'total' in period2_df.columns and period2_df['total'].notna().any():
        stats2['total'] = period2_df['total'].mean()

    if 'awake' in period1_df.columns and period1_df['awake'].notna().any():
        stats1['awake'] = period1_df['awake'].mean()
    if 'awake' in period2_df.columns and period2_df['awake'].notna().any():
        stats2['awake'] = period2_df['awake'].mean()

    if 'sleep' in period1_df.columns and period1_df['sleep'].notna().any():
        stats1['sleep'] = period1_df['sleep'].mean()
    if 'sleep' in period2_df.columns and period2_df['sleep'].notna().any():
        stats2['sleep'] = period2_df['sleep'].mean()

    # Afficher les comparaisons
    if 'total' in stats1 and 'total' in stats2:
        diff = stats2['total'] - stats1['total']
        sign = '+' if diff >= 0 else ''
        print(f"   Stress moyen (total):")
        print(f"      Période 1: {stats1['total']:.1f}")
        print(f"      Période 2: {stats2['total']:.1f} ({sign}{diff:.1f})")

    if 'awake' in stats1 and 'awake' in stats2:
        diff = stats2['awake'] - stats1['awake']
        sign = '+' if diff >= 0 else ''
        print(f"   Stress moyen (éveillé):")
        print(f"      Période 1: {stats1['awake']:.1f}")
        print(f"      Période 2: {stats2['awake']:.1f} ({sign}{diff:.1f})")

    if 'sleep' in stats1 and 'sleep' in stats2:
        diff = stats2['sleep'] - stats1['sleep']
        sign = '+' if diff >= 0 else ''
        print(f"   Stress moyen (sommeil):")
        print(f"      Période 1: {stats1['sleep']:.1f}")
        print(f"      Période 2: {stats2['sleep']:.1f} ({sign}{diff:.1f})")


def plot_stress_data(df, ma_window, draw_options, title="Évolution du stress"):
    """Génère le graphique de stress"""
    fig, ax = plt.subplots(figsize=(15, 7))

    # Déterminer quelles lignes dessiner
    draw_all = draw_options == 'all'
    draw_list = draw_options.split(',') if not draw_all else ['sleep', 'awake', 'avg']

    colors = {
        'sleep': '#3498db',  # Bleu
        'awake': '#e74c3c',  # Rouge
        'avg': '#2ecc71'  # Vert
    }

    labels = {
        'sleep': 'Stress endormi',
        'awake': 'Stress éveillé',
        'avg': 'Stress moyen'
    }

    # Dessiner les données brutes puis la moyenne mobile par-dessus
    if draw_all or 'sleep' in draw_list:
        if 'sleep' in df.columns:
            ax.plot(df['date'], df['sleep'],
                    color=colors['sleep'], linewidth=0.8, alpha=0.3)
        if 'sleep_ma' in df.columns:
            ax.plot(df['date'], df['sleep_ma'],
                    color=colors['sleep'], linewidth=2.5, label=labels['sleep'], alpha=0.85)

    if draw_all or 'awake' in draw_list:
        if 'awake' in df.columns:
            ax.plot(df['date'], df['awake'],
                    color=colors['awake'], linewidth=0.8, alpha=0.3)
        if 'awake_ma' in df.columns:
            ax.plot(df['date'], df['awake_ma'],
                    color=colors['awake'], linewidth=2.5, label=labels['awake'], alpha=0.85)

    if draw_all or 'avg' in draw_list:
        if 'total' in df.columns:
            ax.plot(df['date'], df['total'],
                    color=colors['avg'], linewidth=0.8, alpha=0.3)
        if 'total_ma' in df.columns:
            ax.plot(df['date'], df['total_ma'],
                    color=colors['avg'], linewidth=2.5, label=labels['avg'], alpha=0.85)

    # Formatage
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Niveau de stress', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Ajouter l'info de moyenne mobile en petit en bas à droite
    ax.text(0.99, 0.01, f'Moyenne mobile: {ma_window} jours',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Format de l'axe des X selon la durée
    date_range = (df['date'].max() - df['date'].min()).days

    if date_range > 90:  # Plus de 3 mois : graduations tous les 30 jours
        # Créer des ticks tous les 30 jours
        date_ticks = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='30D')
        ax.set_xticks(date_ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        # Colorer les labels en noir (mode normal)
        for label in ax.get_xticklabels():
            label.set_color('black')
    else:  # 3 mois ou moins : graduations tous les 7 jours
        date_ticks = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='7D')
        ax.set_xticks(date_ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    return fig


def plot_comparison(periods_data, ma_window, draw_options, period_names, period_dates):
    """Génère un graphique de comparaison entre périodes"""
    fig, ax = plt.subplots(figsize=(15, 7))

    draw_all = draw_options == 'all'
    draw_list = draw_options.split(',') if not draw_all else ['sleep', 'awake', 'avg']

    # Palette de couleurs par type de stress et par période
    # Période 1: teintes claires, Période 2: teintes foncées, etc.
    color_palettes = {
        'sleep': ['#5DADE2', '#1F618D', '#85C1E9', '#154360'],  # Bleus
        'awake': ['#EC7063', '#922B21', '#F1948A', '#641E16'],  # Rouges
        'avg': ['#52BE80', '#196F3D', '#7DCEA0', '#0E4429']  # Verts
    }

    line_styles = ['-', '--', '-.', ':']

    for idx, (period_df, period_name) in enumerate(zip(periods_data, period_names)):
        linestyle = line_styles[idx % len(line_styles)]

        # Dessiner les données brutes puis la moyenne mobile
        if draw_all or 'sleep' in draw_list:
            color_sleep = color_palettes['sleep'][idx % len(color_palettes['sleep'])]
            if 'sleep' in period_df.columns:
                ax.plot(period_df['day_offset'], period_df['sleep'],
                        color=color_sleep, linestyle=linestyle, linewidth=0.8, alpha=0.25)
            if 'sleep_ma' in period_df.columns:
                ax.plot(period_df['day_offset'], period_df['sleep_ma'],
                        color=color_sleep, linestyle=linestyle, linewidth=2.5,
                        label=f'{period_name} - Sommeil', alpha=0.8)

        if draw_all or 'awake' in draw_list:
            color_awake = color_palettes['awake'][idx % len(color_palettes['awake'])]
            if 'awake' in period_df.columns:
                ax.plot(period_df['day_offset'], period_df['awake'],
                        color=color_awake, linestyle=linestyle, linewidth=0.8, alpha=0.25)
            if 'awake_ma' in period_df.columns:
                ax.plot(period_df['day_offset'], period_df['awake_ma'],
                        color=color_awake, linestyle=linestyle, linewidth=2.5,
                        label=f'{period_name} - Éveillé', alpha=0.8)

        if draw_all or 'avg' in draw_list:
            color_avg = color_palettes['avg'][idx % len(color_palettes['avg'])]
            if 'total' in period_df.columns:
                ax.plot(period_df['day_offset'], period_df['total'],
                        color=color_avg, linestyle=linestyle, linewidth=0.8, alpha=0.25)
            if 'total_ma' in period_df.columns:
                ax.plot(period_df['day_offset'], period_df['total_ma'],
                        color=color_avg, linestyle=linestyle, linewidth=2.5,
                        label=f'{period_name} - Moyen', alpha=0.8)

    ax.set_xlabel('Jours depuis le début de la période', fontsize=12, fontweight='bold')
    ax.set_ylabel('Niveau de stress', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison des périodes', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Créer un encart avec les informations sur les périodes
    period_info_text = f"Moyenne mobile: {ma_window} jours\n\n"
    for idx, (start, end) in enumerate(period_dates, 1):
        period_info_text += f"Période {idx}: {start} → {end}\n"

    ax.text(0.99, 0.01, period_info_text.strip(),
            transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Déterminer la durée maximale pour l'axe X
    max_days = max(df['day_offset'].max() for df in periods_data if not df.empty)

    if max_days > 90:  # Plus de 3 mois : graduations tous les 30 jours
        # Créer des ticks tous les 30 jours
        x_ticks = list(range(0, int(max_days) + 1, 30))
        ax.set_xticks(x_ticks)

        # Créer des labels avec mois-année pour chaque période
        tick_labels = []
        for tick in x_ticks:
            labels_for_tick = []
            for idx, (start_date, _) in enumerate(period_dates):
                date_at_tick = pd.to_datetime(start_date) + timedelta(days=tick)
                month_year = date_at_tick.strftime('%b %Y')
                labels_for_tick.append(month_year)
            # Joindre les labels de toutes les périodes
            tick_labels.append('\n'.join(labels_for_tick))

        ax.set_xticklabels(tick_labels)

        # Mettre tous les labels en noir
        for label in ax.get_xticklabels():
            label.set_color('black')
    else:  # 3 mois ou moins : graduations tous les 7 jours
        x_ticks = list(range(0, int(max_days) + 1, 7))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'J{tick}' for tick in x_ticks])

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def main():
    args = parse_arguments()

    # Vérifier que le dossier source existe
    if not os.path.isdir(args.source):
        print(f"❌ Le dossier {args.source} n'existe pas")
        sys.exit(1)

    # Lire les données
    print("\n🔍 Lecture des fichiers...")
    raw_data = read_uds_files(args.source)

    # Extraire les données de stress
    print("\n📊 Extraction des données de stress...")
    df = extract_stress_data(raw_data)

    if df.empty:
        print("❌ Aucune donnée de stress trouvée")
        sys.exit(1)

    print(f"✅ Données extraites: {df['date'].min().strftime('%Y-%m-%d')} à {df['date'].max().strftime('%Y-%m-%d')}")

    # Détecter les trous dans les données
    gaps = detect_data_gaps(df, min_gap_days=4)
    if gaps:
        print("\n⚠️  Trous détectés dans les données:")
        for gap in gaps:
            print(
                f"   - Du {gap['start'].strftime('%Y-%m-%d')} au {gap['end'].strftime('%Y-%m-%d')} ({gap['days']} jours manquants)")
    else:
        print("\n✅ Aucun trou significatif détecté dans les données")

    # Appliquer la moyenne mobile
    print(f"\n📈 Application de la moyenne mobile sur {args.ma} jours...")
    df = apply_moving_average(df, args.ma)

    # Mode comparaison ou mode normal
    if args.compare:
        print("\n🔄 Mode comparaison activé")
        try:
            parts = args.compare.split(',')
            if len(parts) != 3:
                raise ValueError("Format attendu: START_DAY_1,START_DAY_2,LENGTH")

            ref1, ref2, length = parts[0].strip(), parts[1].strip(), int(parts[2].strip())

            # Filtrer les périodes et obtenir les dates de début/fin
            period1, start1, end1 = filter_period(df, ref1, length)
            period2, start2, end2 = filter_period(df, ref2, length)

            if period1.empty or period2.empty:
                print("❌ Une ou plusieurs périodes n'ont pas de données")
                sys.exit(1)

            # Afficher les périodes avec début et fin
            print("\n📅 Périodes comparées:")
            print(
                f"   Période 1: {start1.strftime('%Y-%m-%d')} → {(end1 - timedelta(days=1)).strftime('%Y-%m-%d')} ({abs(length)} jours)")
            print(
                f"   Période 2: {start2.strftime('%Y-%m-%d')} → {(end2 - timedelta(days=1)).strftime('%Y-%m-%d')} ({abs(length)} jours)")

            period_names = [
                f"Période 1",
                f"Période 2"
            ]

            period_dates = [
                (start1.strftime('%Y-%m-%d'), (end1 - timedelta(days=1)).strftime('%Y-%m-%d')),
                (start2.strftime('%Y-%m-%d'), (end2 - timedelta(days=1)).strftime('%Y-%m-%d'))
            ]

            print("\n📉 Génération du graphique...")
            fig = plot_comparison([period1, period2], args.ma, args.draw, period_names, period_dates)

            # Statistiques comparées
            print_comparison_statistics(period1, period2)

        except Exception as e:
            print(f"❌ Erreur lors de la comparaison: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Mode normal
        # Vérifier si on a un range spécifié
        if args.range:
            print("\n📅 Mode range activé")
            try:
                parts = args.range.split(',')
                if len(parts) != 2:
                    raise ValueError("Format attendu: START_DAY,LENGTH")

                ref_date, length = parts[0].strip(), int(parts[1].strip())
                df, start, end = filter_period(df, ref_date, length)

                if df.empty:
                    print("❌ Aucune donnée dans la période spécifiée")
                    sys.exit(1)

                print(
                    f"📊 Période sélectionnée: {start.strftime('%Y-%m-%d')} → {(end - timedelta(days=1)).strftime('%Y-%m-%d')} ({abs(length)} jours)")

            except Exception as e:
                print(f"❌ Erreur lors du filtrage de la période: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

        print("\n📉 Génération du graphique...")
        fig = plot_stress_data(df, args.ma, args.draw)

        # Statistiques
        print("\n📊 Statistiques:")
        print_statistics(df)

    print("\n✨ Graphique généré ! Affichage...")
    plt.show()


if __name__ == '__main__':
    main()
