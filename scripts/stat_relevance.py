# -*- coding: utf-8 -*-
"""
analyze_relevance.py (Angepasste Version für CSV-Ausgabe)

Führt eine statistische Relevanzanalyse der Reddit-Metriken im Zusammenhang
mit dem GME-Aktienkurs durch und speichert die vorbereiteten Daten für
manuelle Visualisierungen.

Analyseschritte:
1. Daten laden und zusammenführen.
2. Daten vorbereiten (NaNs, Preisänderungen, Log-Preis, Lags).
3. Speichern des umfassenden, vorbereiteten DataFrames als CSV.
4. Deskriptive Statistiken (Konsolenausgabe).
5. Korrelationsanalyse (Konsolenausgabe + Heatmap).
6. Granger-Kausalitätstests (Konsolenausgabe).

@author: Moritz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import warnings
import os
import sys
import traceback

# --- KONFIGURATION ---
FILE_PATHS = {
    "sentiment_daily": r"C:\Users\Moriz\OneDrive - Hochschule München\Hochschule\Bachelorarbeit\GME_Coding\GME_Coding\results\ALL_reddit_sentiment_daily.csv",
    "activity_wide": r"C:\Users\Moriz\OneDrive - Hochschule München\Hochschule\Bachelorarbeit\GME_Coding\GME_Coding\results\subreddit_activity_stats_wide.csv",
    "stock_data": r"C:\Users\Moriz\OneDrive - Hochschule München\Hochschule\Bachelorarbeit\GME_Coding\GME_Coding\results\gme_stock_data.csv",
    "output_dir": r"C:\Users\Moriz\OneDrive - Hochschule München\Hochschule\Bachelorarbeit\GME_Coding\GME_Coding\results\analysis_output", # Ordner für Plots/Ergebnisse
    "comprehensive_data": r"C:\Users\Moriz\OneDrive - Hochschule München\Hochschule\Bachelorarbeit\GME_Coding\GME_Coding\results\analysis_output\COMPREHENSIVE_ANALYSIS_DATA.csv"
}

# Parameter für die Analyse
LAG_DAYS = 5
SIGNIFICANCE_LEVEL = 0.05

# Erstelle Output-Verzeichnis, falls nicht vorhanden
os.makedirs(FILE_PATHS["output_dir"], exist_ok=True)

# --- Hilfsfunktionen ---
# Definition von plot_dual_axis bleibt bestehen, wird aber nicht mehr aufgerufen
def plot_dual_axis(df, col1, col2, label1, label2, title, filename, log_scale1=False):
    """Erstellt einen Plot mit zwei Y-Achsen."""
    fig, ax1 = plt.subplots(figsize=(15, 7))

    color1 = 'tab:red'
    ax1.set_xlabel('Datum')
    ax1.set_ylabel(label1, color=color1)
    if log_scale1:
        # Stelle sicher, dass die Daten positiv sind für Log-Skala
        plot_data1 = df[col1].clip(lower=0.001) # Clip an einem kleinen positiven Wert
        ax1.set_yscale('log')
        ax1.plot(df.index, plot_data1, color=color1, label=f"{label1} (log)")
    else:
        ax1.plot(df.index, df[col1], color=color1, label=label1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel(label2, color=color2)
    ax2.plot(df.index, df[col2], color=color2, alpha=0.7, label=label2)
    ax2.tick_params(axis='y', labelcolor=color2)
    if not df[col2].empty and df[col2].min() < 0 < df[col2].max(): # Nulllinie nur wenn sinnvoll (z.B. Sentiment)
        ax2.axhline(0, color='grey', linewidth=0.5, linestyle='--')

    # Formatierung
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2)) # Alle 2 Monate Major Tick
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.title(title)
    fig.tight_layout()
    # Verwende fig.legend statt plt.legend für bessere Platzierung mit twinx
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right') # Kombinierte Legende
    plt.grid(True, axis='x', linestyle=':')

    filepath = os.path.join(FILE_PATHS["output_dir"], filename)
    plt.savefig(filepath)
    print(f"Plot gespeichert: {filepath}")
    plt.close(fig)

# run_adf_test
def run_adf_test(series, name):
    """Führt den Augmented Dickey-Fuller Test durch und gibt das Ergebnis aus."""
    print(f"\n--- ADF Test für: {name} ---")
    series_cleaned = series.dropna()
    if series_cleaned.empty:
        print("Serie ist leer nach NaN-Entfernung. ADF Test nicht möglich.")
        return False

    try:
        result = adfuller(series_cleaned)
        print(f'ADF Statistic: {result[0]:.4f}')
        print(f'p-value: {result[1]:.4f}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.4f}')

        if result[1] <= SIGNIFICANCE_LEVEL:
            print(f"-> Ergebnis: Stationär (p <= {SIGNIFICANCE_LEVEL})")
            return True
        else:
            print(f"-> Ergebnis: Nicht stationär (p > {SIGNIFICANCE_LEVEL})")
            return False
    except Exception as e:
        print(f"FEHLER beim ADF Test für {name}: {e}")
        return False

# --- 1. Daten laden und zusammenführen ---
print("--- 1. Lade und merge Daten ---")
try:
    df_sentiment = pd.read_csv(FILE_PATHS["sentiment_daily"], sep=';', encoding='utf-8-sig', parse_dates=['Date'])
    df_sentiment.set_index('Date', inplace=True)

    df_activity = pd.read_csv(FILE_PATHS["activity_wide"], sep=';', encoding='utf-8-sig')
    first_col_name_activity = df_activity.columns[0]
    if 'Unnamed' in first_col_name_activity or first_col_name_activity.lower() == 'date':
        print(f"Identifiziere erste Spalte ('{first_col_name_activity}') als Datum in activity_wide.")
        df_activity.rename(columns={first_col_name_activity: 'Date'}, inplace=True)
        df_activity['Date'] = pd.to_datetime(df_activity['Date'], errors='coerce')
    elif 'date' in df_activity.columns:
        print("Identifiziere Spalte 'date' (lowercase) als Datum in activity_wide.")
        df_activity.rename(columns={'date': 'Date'}, inplace=True)
        df_activity['Date'] = pd.to_datetime(df_activity['Date'], errors='coerce')
    elif 'Date' not in df_activity.columns:
         raise ValueError(f"Konnte keine geeignete Datumsspalte ('Date', 'date' oder erste Spalte) in {FILE_PATHS['activity_wide']} finden.")
    df_activity.dropna(subset=['Date'], inplace=True)
    df_activity.set_index('Date', inplace=True)

    df_stock = pd.read_csv(FILE_PATHS["stock_data"], sep=';', encoding='utf-8-sig', parse_dates=['Date'])
    df_stock.set_index('Date', inplace=True)

    df_merged = df_sentiment.join(df_activity, how='outer')
    df_final = df_merged.join(df_stock, how='outer')

    df_final.sort_index(inplace=True)

    print(f"Daten geladen. Zeitraum: {df_final.index.min()} bis {df_final.index.max()}")
    print(f"Anzahl Zeilen initial: {len(df_final)}")

except FileNotFoundError as e:
    print(f"FEHLER: Datei nicht gefunden! {e}")
    sys.exit(1)
except ValueError as e:
    print(f"FEHLER: Problem mit Datumsspalte. {e}")
    sys.exit(1)
except Exception as e:
    print(f"FEHLER beim Laden oder Mergen der Daten: {e}")
    print(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)


# --- 2. Daten vorbereiten ---
print("\n--- 2. Bereite Daten vor ---")

# Spalten auswählen/umbenennen
activity_cols_mapping = { # Renaming Mapping
    'posts_per_day_wallstreetbets': 'Posts_WSB',
    'posts_per_day_gme': 'Posts_GME',
    'posts_per_day_superstonk': 'Posts_Superstonk',
    'comments_per_day_wallstreetbets': 'Comments_WSB',
    'comments_per_day_gme': 'Comments_GME',
    'comments_per_day_superstonk': 'Comments_Superstonk',
    'subscribers_wallstreetbets': 'Subs_WSB',
    'subscribers_gme': 'Subs_GME',
    'subscribers_superstonk': 'Subs_Superstonk'
}
# Wende Renaming nur auf existierende Spalten an
df_final.rename(columns={k: v for k, v in activity_cols_mapping.items() if k in df_final.columns}, inplace=True)

# Liste der umbenannten Aktivitätsspalten, die jetzt im DataFrame sind
activity_cols_renamed = [v for k, v in activity_cols_mapping.items() if v in df_final.columns]

# Gesamtaktivität berechnen (NACH dem Umbenennen)
post_cols = [col for col in ['Posts_WSB', 'Posts_GME', 'Posts_Superstonk'] if col in df_final.columns]
comment_cols = [col for col in ['Comments_WSB', 'Comments_GME', 'Comments_Superstonk'] if col in df_final.columns]
sub_cols = [col for col in ['Subs_WSB', 'Subs_GME', 'Subs_Superstonk'] if col in df_final.columns]

if post_cols:
    df_final['Total_Posts'] = df_final[post_cols].sum(axis=1, skipna=False) # skipna=False, damit ffill funktioniert, wenn eine Spalte fehlt
if comment_cols:
    df_final['Total_Comments'] = df_final[comment_cols].sum(axis=1, skipna=False)
if sub_cols:
    df_final['Total_Subscribers'] = df_final[sub_cols].sum(axis=1, skipna=False)


# --- Behandle fehlende Werte (NaNs) ---
# Schritt 1: Forward fill für Aktien und Aktivitätsdaten (Posts, Comments, Subs von subredditstats)
stock_cols_nan = ['Price', 'Volume']
activity_cols_to_ffill = activity_cols_renamed + ['Total_Posts', 'Total_Comments', 'Total_Subscribers']
# Nur ffill auf Spalten anwenden, die tatsächlich existieren
activity_cols_to_ffill_existing = [col for col in activity_cols_to_ffill if col in df_final.columns]

print(f"Wende Forward Fill (ffill) auf folgende Spalten an: {stock_cols_nan + activity_cols_to_ffill_existing}")
df_final[stock_cols_nan + activity_cols_to_ffill_existing] = df_final[stock_cols_nan + activity_cols_to_ffill_existing].ffill()

# Schritt 2: Fülle verbleibende NaNs für Sentiment/Upvote-Metriken mit 0
#            (Diese kommen aus dem local scraper, wo 0 "keine Posts gefunden" bedeutet)
sentiment_upvote_cols_nan = ['Avg_Sentiment_Final', 'Avg_Sentiment_VADER', 'Avg_Subjectivity', 'Post_Count',
                             'Median_Upvotes', 'Avg_Upvotes', 'Sum_Upvotes']
sentiment_upvote_cols_nan_existing = [col for col in sentiment_upvote_cols_nan if col in df_final.columns]

print(f"Fülle verbleibende NaNs mit 0 für: {sentiment_upvote_cols_nan_existing}")
df_final[sentiment_upvote_cols_nan_existing] = df_final[sentiment_upvote_cols_nan_existing].fillna(0)

# Schritt 3: Entferne Zeilen, bei denen der Preis *immer noch* NaN ist (ganz am Anfang, vor dem ersten Kursdatum)
df_final.dropna(subset=['Price'], inplace=True)
print(f"Anzahl Zeilen nach NaN-Behandlung: {len(df_final)}")

# --- Berechne Preisänderungen und Returns ---
df_final['Price_Change'] = df_final['Price'].diff()
df_final['Log_Return'] = np.log(df_final['Price'] / df_final['Price'].shift(1))
if 'Price' in df_final.columns:
    df_final['Price_Log'] = np.log(df_final['Price'].astype(float).replace(0, np.nan).clip(lower=0.0001))
    df_final['Price_Log'].fillna(method='ffill', inplace=True)
    df_final['Price_Log'].fillna(method='bfill', inplace=True)
    print("Logarithmierte Preisspalte 'Price_Log' hinzugefügt.")

# Ersetze initiale NaNs/Infs in Changes/Returns/Logs, die durch diff/log entstanden sind
cols_to_fill_zero = ['Price_Change', 'Log_Return']
if 'Price_Log' in df_final.columns: cols_to_fill_zero.append('Price_Log')
df_final[cols_to_fill_zero] = df_final[cols_to_fill_zero].fillna(0)
df_final.replace([np.inf, -np.inf], 0, inplace=True) # Ersetze Inf durch 0

# --- Erstelle Lagged Variables ---
metrics_to_lag = ['Avg_Sentiment_Final', 'Avg_Sentiment_VADER', 'Post_Count', 'Total_Posts',
                   'Total_Comments', 'Total_Subscribers', 'Median_Upvotes', 'Sum_Upvotes', 'Volume', 'Log_Return']
metrics_to_lag_existing = [m for m in metrics_to_lag if m in df_final.columns]
print(f"Erstelle Lagged Variables für {LAG_DAYS} Tage für: {metrics_to_lag_existing}")
for metric in metrics_to_lag_existing:
    for i in range(1, LAG_DAYS + 1):
        df_final[f'{metric}_Lag{i}'] = df_final[metric].shift(i)

# Entferne initiale Zeilen mit NaNs durch das Lagging
df_final.dropna(inplace=True)
print(f"Anzahl Zeilen nach Lag-Erstellung und NaN-Drop: {len(df_final)}")

# *** Speichere das finale, vorbereitete DataFrame ***
try:
    # Speichere MIT Datum als Index
    df_final.to_csv(FILE_PATHS["comprehensive_data"], sep=';', encoding='utf-8-sig', index=True)
    print(f"\nUmfassendes vorbereitetes DataFrame gespeichert: {FILE_PATHS['comprehensive_data']}")
    print("-> Dieses CSV enthält alle gemergten, berechneten und gelaggten Daten für eigene Visualisierungen.")
except Exception as e:
    print(f"FEHLER beim Speichern des umfassenden DataFrames: {e}")


# --- 2a. Stationaritätsprüfung ---
print("\n--- 2a. Stationaritätsprüfung (ADF-Test) ---")
vars_for_stationarity_check = ['Price_Log', 'Log_Return', 'Volume', 'Avg_Sentiment_Final', 'Post_Count', 'Total_Posts']
stationarity_results = {}
for var in vars_for_stationarity_check:
    if var in df_final.columns:
        stationarity_results[var] = run_adf_test(df_final[var], var)
    else:
        print(f"Warnung: Spalte {var} nicht gefunden für ADF-Test.")

# --- 3. Deskriptive Statistiken ---
print("\n--- 3. Deskriptive Statistiken ---")
print("\nDeskriptive Statistiken:")
desc_cols = ['Price_Log', 'Volume', 'Price_Change', 'Log_Return', 'Avg_Sentiment_Final', 'Post_Count', 'Total_Posts', 'Median_Upvotes']
desc_cols_existing = [col for col in desc_cols if col in df_final.columns]
if desc_cols_existing:
    desc_stats = df_final[desc_cols_existing].describe()
    for col in ['Volume', 'Median_Upvotes', 'Total_Posts', 'Total_Comments', 'Total_Subscribers']: # Füge Total_* hinzu
         if col in desc_stats.columns:
             desc_stats[col] = desc_stats[col].apply(lambda x: f"{x:,.0f}") # Tausendertrennzeichen
    print(desc_stats.round(4))
else:
    print("Keine Spalten für deskriptive Statistiken gefunden.")


# --- 4. Korrelationsanalyse ---
print("\n--- 4. Korrelationsanalyse ---")
corr_cols_concurrent = ['Price_Log', 'Volume', 'Price_Change', 'Log_Return', 'Avg_Sentiment_Final',
                        'Post_Count', 'Total_Posts', 'Median_Upvotes', 'Sum_Upvotes'] # Füge relevante Spalten hinzu
corr_cols_concurrent = [col for col in corr_cols_concurrent if col in df_final.columns]

if len(corr_cols_concurrent) > 1:
    correlation_matrix = df_final[corr_cols_concurrent].corr()
    print("\nKorrelationsmatrix (gleichzeitig):")
    print(correlation_matrix.round(3))

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8})
    plt.title('Korrelationsmatrix (gleichzeitige Metriken)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    heatmap_path = os.path.join(FILE_PATHS["output_dir"], 'heatmap_concurrent_correlation.png')
    plt.savefig(heatmap_path)
    print(f"Heatmap gespeichert: {heatmap_path}")
    plt.close()

    print("\nSignifikanztests (Pearson r) für ausgewählte Paare:")
    pairs_to_test = [('Avg_Sentiment_Final', 'Log_Return'), ('Post_Count', 'Log_Return'),
                     ('Post_Count', 'Volume'), ('Total_Posts', 'Volume')]
    for col1, col2 in pairs_to_test:
        if col1 in df_final.columns and col2 in df_final.columns:
            df_pair = df_final[[col1, col2]].dropna()
            if len(df_pair) > 1:
                 try:
                     res = stats.pearsonr(df_pair[col1], df_pair[col2])
                     print(f"  {col1} vs {col2}: r={res[0]:.3f}, p={res[1]:.4f}{' (SIGNIFIKANT)' if res[1] < SIGNIFICANCE_LEVEL else ''}")
                 except ValueError as pe:
                     print(f"  FEHLER bei Pearson r für {col1} vs {col2}: {pe}")
            else:
                 print(f"  {col1} vs {col2}: Zu wenige Datenpunkte für Korrelationstest.")

print("\nKorrelationsmatrix (gelaggte Reddit-Metriken vs. aktuelle Aktienmetriken):")
lagged_cols_exist = [f'{m}_Lag1' for m in metrics_to_lag_existing if f'{m}_Lag1' in df_final.columns]
stock_cols_for_lag = [col for col in ['Price_Change', 'Log_Return', 'Volume'] if col in df_final.columns]
cols_for_lagged_corr = stock_cols_for_lag + lagged_cols_exist

if len(cols_for_lagged_corr) > 1 and lagged_cols_exist and stock_cols_for_lag:
    lagged_correlation_matrix = df_final[cols_for_lagged_corr].corr()
    print(lagged_correlation_matrix.loc[stock_cols_for_lag, lagged_cols_exist].round(3))

    print("\nSignifikanztests (Pearson r) für ausgewählte gelaggte Paare:")
    lagged_pairs_to_test = [('Avg_Sentiment_Final_Lag1', 'Log_Return'), ('Post_Count_Lag1', 'Log_Return'),
                            ('Post_Count_Lag1', 'Volume'), ('Total_Posts_Lag1', 'Volume')]
    for col1_lag, col2_current in lagged_pairs_to_test:
         if col1_lag in df_final.columns and col2_current in df_final.columns:
             df_pair_lag = df_final[[col1_lag, col2_current]].dropna()
             if len(df_pair_lag) > 1:
                 try:
                     res_lag = stats.pearsonr(df_pair_lag[col1_lag], df_pair_lag[col2_current])
                     print(f"  {col1_lag} vs {col2_current}: r={res_lag[0]:.3f}, p={res_lag[1]:.4f}{' (SIGNIFIKANT)' if res_lag[1] < SIGNIFICANCE_LEVEL else ''}")
                 except ValueError as pe:
                      print(f"  FEHLER bei Pearson r für {col1_lag} vs {col2_current}: {pe}")
             else:
                 print(f"  {col1_lag} vs {col2_current}: Zu wenige Datenpunkte für Korrelationstest.")

# --- 5. Granger-Kausalitätstest ---
print("\n--- 5. Granger-Kausalitätstest ---")
warnings.filterwarnings("ignore")

granger_pairs = [
    ('Log_Return', 'Avg_Sentiment_Final'), ('Log_Return', 'Post_Count'), ('Log_Return', 'Total_Posts'),
    ('Volume', 'Avg_Sentiment_Final'), ('Volume', 'Post_Count'), ('Volume', 'Total_Posts')
]

for target, cause in granger_pairs:
    if target in df_final.columns and cause in df_final.columns:
        print(f"\nTeste: Beeinflusst '{cause}' die Variable '{target}'? (maxlag={LAG_DAYS})")
        data_granger = df_final[[target, cause]].dropna()
        if len(data_granger) < 30:
            print(f" -> Zu wenige Datenpunkte ({len(data_granger)}) nach NaN-Drop für Test.")
            continue
        # Prüfe Stationarität erneut (oder verwende gespeicherte Ergebnisse)
        target_stationary = stationarity_results.get(target, False)
        cause_stationary = stationarity_results.get(cause, False)
        if not target_stationary or not cause_stationary:
             print(f" -> WARNUNG: Mindestens eine Zeitreihe ({target}: {target_stationary}, {cause}: {cause_stationary}) nicht stationär. Ergebnisse evtl. unzuverlässig!")

        try:
             test_result = grangercausalitytests(data_granger, maxlag=LAG_DAYS, verbose=False)
             significant_lags = []
             for lag in range(1, LAG_DAYS + 1):
                 p_value_f = test_result[lag][0]['ssr_ftest'][1]
                 if p_value_f < SIGNIFICANCE_LEVEL:
                     significant_lags.append((lag, p_value_f))

             if significant_lags:
                 print(f" -> Signifikante Granger-Kausalität gefunden (p < {SIGNIFICANCE_LEVEL}) bei folgenden Lags:")
                 for lag, p_val in significant_lags:
                     print(f"    Lag {lag}: p={p_val:.4f}")
             else:
                 print(f" -> Keine signifikante Granger-Kausalität gefunden (alle p >= {SIGNIFICANCE_LEVEL}) für getestete Lags.")
        except Exception as e:
            print(f"FEHLER beim Granger-Test für {target} ~ {cause}: {e}")
    else:
        print(f"Warnung: Mindestens eine Spalte für Granger-Test ({target}, {cause}) nicht gefunden.")

warnings.filterwarnings("default")

# --- Abschluss ---
# (Abschlussbemerkungen angepasst)
print("\n\n--- Analyse abgeschlossen ---")
print(f"Ergebnisse (Heatmap, finale Daten-CSV) wurden im Ordner '{FILE_PATHS['output_dir']}' gespeichert.")
print(f"Umfassende Daten für eigene Plots: {FILE_PATHS['comprehensive_data']}")