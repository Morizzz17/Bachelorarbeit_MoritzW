# -*- coding: utf-8 -*-
"""
analyze_relevance.py (Version mit KORRIGIERTER PDF-Bericht-Generierung)

Führt eine statistische Relevanzanalyse der Reddit-Metriken im Zusammenhang
mit dem GME-Aktienkurs durch, speichert die vorbereiteten Daten für
manuelle Visualisierungen und generiert einen PDF-Bericht der Ergebnisse.

Änderung: Korrekte Tabellenerstellung mit fpdf.table() und Erklärung zur Stationarität.

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
from fpdf import FPDF # Importiere FPDF
from math import ceil # Für Tabellenzeilenhöhe

# --- KONFIGURATION ---
FILE_PATHS = {
    "sentiment_daily": r"C:\Users\Moriz\OneDrive - Hochschule München\Hochschule\Bachelorarbeit\GME_Coding\GME_Coding\results\ALL_reddit_sentiment_daily.csv",
    "activity_wide": r"C:\Users\Moriz\OneDrive - Hochschule München\Hochschule\Bachelorarbeit\GME_Coding\GME_Coding\results\subreddit_activity_stats_wide.csv", # !!! Ggf. Versionsnummer prüfen/anpassen
    "stock_data": r"C:\Users\Moriz\OneDrive - Hochschule München\Hochschule\Bachelorarbeit\GME_Coding\GME_Coding\results\gme_stock_data.csv",
    "output_dir": r"C:\Users\Moriz\OneDrive - Hochschule München\Hochschule\Bachelorarbeit\GME_Coding\GME_Coding\results\analysis_output",
    "comprehensive_data": r"C:\Users\Moriz\OneDrive - Hochschule München\Hochschule\Bachelorarbeit\GME_Coding\GME_Coding\results\analysis_output\COMPREHENSIVE_ANALYSIS_DATA.csv",
    "heatmap_image": r"C:\Users\Moriz\OneDrive - Hochschule München\Hochschule\Bachelorarbeit\GME_Coding\GME_Coding\results\analysis_output\heatmap_concurrent_correlation.png", # Pfad zur Heatmap
    "pdf_report": r"C:\Users\Moriz\OneDrive - Hochschule München\Hochschule\Bachelorarbeit\GME_Coding\GME_Coding\results\analysis_output\Statistische_Relevanzanalyse_Report.pdf" # Pfad für PDF
}

# Parameter für die Analyse
LAG_DAYS = 5
SIGNIFICANCE_LEVEL = 0.05

# Erstelle Output-Verzeichnis, falls nicht vorhanden
os.makedirs(FILE_PATHS["output_dir"], exist_ok=True)

# --- Hilfsfunktionen ---

# ADF Test modifiziert: Gibt Ergebnisse als Liste von Strings zurück
def run_adf_test(series, name):
    """Führt den ADF Test durch und gibt das Ergebnis als String-Liste zurück."""
    results_list = [f"--- ADF Test für: {name} ---"]
    series_cleaned = series.dropna()
    is_stationary = False
    if series_cleaned.empty:
        results_list.append("Serie ist leer nach NaN-Entfernung. ADF Test nicht möglich.")
        return results_list, is_stationary

    try:
        result = adfuller(series_cleaned)
        results_list.append(f'ADF Statistic: {result[0]:.4f}')
        results_list.append(f'p-value: {result[1]:.4f}')
        results_list.append('Critical Values:')
        for key, value in result[4].items():
            results_list.append(f'    {key}: {value:.4f}')

        if result[1] <= SIGNIFICANCE_LEVEL:
            results_list.append(f"-> Ergebnis: Stationär (p <= {SIGNIFICANCE_LEVEL})")
            is_stationary = True
        else:
            results_list.append(f"-> Ergebnis: Nicht stationär (p > {SIGNIFICANCE_LEVEL})")
            is_stationary = False
    except Exception as e:
        results_list.append(f"FEHLER beim ADF Test für {name}: {e}")
        is_stationary = False
    print("\n".join(results_list)) # Optional: weiterhin auf Konsole ausgeben
    return results_list, is_stationary

# *** PDF Klasse und Hilfsfunktionen ***
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Statistische Relevanzanalyse GME & Reddit', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Seite {self.page_no()}', 0, 0, 'C')

    def section_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.ln(8)
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_font('Arial', '', 10) # Reset font for content
        self.ln(4)

    def add_text_list(self, text_list, is_code=False):
        if is_code:
            self.set_font("Courier", size=9)
        else:
            self.set_font("Arial", size=10)
        for line in text_list:
            self.set_x(self.l_margin) # Stelle sicher, dass wir am linken Rand sind
            try:
                line_encoded = line
                # Verwende write für besseren Zeilenumbruch
                self.write(5, txt=line_encoded)
                self.ln() # Manueller Zeilenumbruch
            except UnicodeEncodeError:
                 line_encoded = line.encode('latin-1', 'replace').decode('latin-1')
                 self.write(5, txt=line_encoded)
                 self.ln()
        self.ln(2)
        self.set_font("Arial", size=10) # Reset font

        # *** NEUE, ROBUSTERE Funktion für Tabellen mit fpdf.table() ***
    def add_dataframe_as_table(self, df, title):
        self.section_title(title)
        self.set_font("Arial", size=8)
        self.ln(2)

        if df is None or df.empty:
             self.cell(0, 10, "Keine Daten für diese Tabelle verfügbar.", ln=True) # ln=True für Zeilenumbruch
             self.ln(4)
             return

        # --- Vorbereitung ---
        df_reset = df.reset_index()
        # Indexspalte umbenennen
        if df_reset.columns[0] in ['Date', 'index']:
            df_reset.rename(columns={df_reset.columns[0]: 'Metrik'}, inplace=True)
        elif df.index.name:
             df_reset.rename(columns={df.index.name: 'Metrik'}, inplace=True)
        else:
             df_reset.rename(columns={df.index.name if df.index.name else 'Index': 'Metrik'}, inplace=True) # Fängt auch unbenannte Indexe ab

        # Alle Daten zu Strings konvertieren
        df_string = df_reset.astype(str)

        headers = df_string.columns.tolist()
        data = df_string.values.tolist()
        table_data = [headers] + data

        # --- Spaltenbreiten berechnen ---
        effective_page_width = self.w - 2 * self.l_margin
        num_cols = len(headers)

        # Schätze eine minimale Breite pro Spalte basierend auf Header + längstem Datenpunkt
        col_widths = []
        padding = 2 # Etwas Platz links/rechts in der Zelle
        for i, header in enumerate(headers):
            header_width = self.get_string_width(header) + (2 * padding)
            max_data_width = 0
            for row_data in data:
                try:
                    data_width = self.get_string_width(row_data[i]) + (2 * padding)
                    max_data_width = max(max_data_width, data_width)
                except IndexError:
                    pass # Falls eine Zeile weniger Spalten hat
            # Nimm die größere Breite (Header oder Daten), füge etwas Puffer hinzu
            col_widths.append(max(header_width, max_data_width) + 2)

        # Skaliere Breiten, damit sie die Seitenbreite füllen (oder nicht überschreiten)
        total_calculated_width = sum(col_widths)
        if total_calculated_width > effective_page_width:
            # Skaliere proportional herunter, wenn zu breit
            scale_factor = effective_page_width / total_calculated_width
            col_widths = [w * scale_factor for w in col_widths]
        elif total_calculated_width < effective_page_width and num_cols > 0:
             # Verteile den Restplatz gleichmäßig, wenn zu schmal
             extra_space = (effective_page_width - total_calculated_width) / num_cols
             col_widths = [w + extra_space for w in col_widths]
        # Sicherstellen, dass die Summe exakt passt (kleine Korrektur für Rundungsfehler)
        final_total_width = sum(col_widths)
        width_diff = effective_page_width - final_total_width
        if num_cols > 0:
             col_widths[0] += width_diff # Füge Differenz zur ersten Spalte hinzu

        # --- Tabelle erstellen ---
        line_height = self.font_size * 1.8
        try:
            with self.table(
                col_widths=tuple(col_widths), # Übergebe Tuple der berechneten Breiten
                text_align="LEFT", # Standard links
                line_height=line_height,
                padding=padding, # Verwende definiertes Padding
                markdown=True,
                # width Parameter weglassen!
                ) as table:
                is_first_row = True
                for data_row in table_data:
                    row = table.row()
                    for i, datum in enumerate(data_row):
                        cell_text = f"**{datum}**" if is_first_row else datum
                        # Rechtsbündig für Zahlen (alle Spalten außer der ersten)
                        align = "RIGHT" if i > 0 and not is_first_row else "LEFT"
                        row.cell(cell_text, align=align)
                    is_first_row = False # Nach der ersten Zeile (Header)
        except Exception as table_err:
             print(f"FEHLER beim Erstellen der Tabelle '{title}': {table_err}")
             self.set_font("Arial", "I", 9)
             # Verwende multi_cell für die Fehlermeldung im PDF
             self.multi_cell(0, 5, f"Fehler beim Rendern der Tabelle '{title}'. Bitte Daten prüfen. Fehler: {table_err}")

        self.ln(4)
        self.set_font("Arial", size=10) # Reset Font nach Tabelle


def generate_pdf_report(adf_results_dict, desc_stats_df, corr_matrix_conc, sig_tests_conc_list,
                        corr_matrix_lag, sig_tests_lag_list, granger_results_list, file_paths):
    """Generiert den PDF-Bericht mit den Analyseergebnissen."""
    pdf = PDF(orientation='L', unit='mm', format='A4') # *** Landscape für breitere Tabellen ***
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

        # 1. Stationaritätstests
    pdf.section_title("1. Stationaritätstests (ADF)")
    for test_name, (result_text_list, _) in adf_results_dict.items():
        # Wir entpacken jetzt korrekt die Liste und ignorieren den Boolean mit '_'
        # (Oder wir könnten ihn verwenden: pdf.add_text_list(result_text_list + [f"Stationär: {_}"]))
        pdf.add_text_list(result_text_list, is_code=True)
        pdf.ln(1) # Kleiner Abstand zwischen Tests

    # *** NEU: Erklärung zur Stationarität ***
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 6, "Erläuterung zur Stationarität:", ln=1)
    pdf.set_font('Arial', '', 10)
    explanation = [
        "- Stationär: Eine Zeitreihe ist stationär, wenn ihr statistischen Eigenschaften (Mittelwert, Varianz, Autokorrelation) über die Zeit konstant bleiben. Sie schwankt um einen Mittelwert, ohne langfristigen Trend oder systematisch veränderliche Streuung. Stationarität ist oft eine Voraussetzung für Zeitreihenmodelle wie Granger-Kausalität.",
        "- Nicht stationär: Eine Zeitreihe ist nicht stationär, wenn sich Mittelwert oder Varianz über die Zeit ändern. Dies ist häufig bei Aktienkursen (Trends) oder bei Metriken mit starkem Wachstum (wie Nutzerzahlen) der Fall. Granger-Tests mit nicht-stationären Reihen können zu irreführenden Ergebnissen (Scheinkorrelationen) führen."
    ]
    pdf.add_text_list(explanation)
    pdf.ln(5)

    # 2. Deskriptive Statistiken
    # Formatieren für PDF (innerhalb der Funktion jetzt robuster)
    desc_stats_pdf = desc_stats_df.copy()
    pdf.add_dataframe_as_table(desc_stats_pdf.round(4), "2. Deskriptive Statistiken")

    # 3. Korrelationsanalyse
    pdf.add_page() # Neue Seite für die Korrelationen
    pdf.section_title("3. Korrelationsanalyse")
    # Gleichzeitig
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "3.1 Gleichzeitige Korrelationen", 0, 1, 'L')
    pdf.ln(2)
    if not corr_matrix_conc.empty:
        pdf.add_dataframe_as_table(corr_matrix_conc.round(3), "Korrelationsmatrix (gleichzeitig)")
        pdf.add_text_list([" ", "Heatmap (gleichzeitig):"], is_code=False)
        if os.path.exists(file_paths["heatmap_image"]):
            page_width = pdf.w - 2 * pdf.l_margin
            image_width = min(page_width, 180)
            x_pos = (pdf.w - image_width) / 2
            pdf.image(file_paths["heatmap_image"], x=x_pos, w=image_width)
            pdf.ln(5)
        else:
            pdf.cell(0, 10, txt="Heatmap-Bild nicht gefunden.", ln=1)
        pdf.add_text_list([" ", "Signifikanztests (Pearson r):"] + sig_tests_conc_list, is_code=True)
    else:
         pdf.add_text_list(["Keine gleichzeitigen Korrelationen berechnet."], is_code=False)

    # Gelaggt
    pdf.add_page() # Eigene Seite für gelaggte Korrelationen
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "3.2 Gelaggte Korrelationen (Lag 1 vs. Aktuell)", 0, 1, 'L')
    pdf.ln(2)
    if corr_matrix_lag is not None and not corr_matrix_lag.empty:
        pdf.add_dataframe_as_table(corr_matrix_lag.round(3), "Korrelationsmatrix (gelaggt)")
        pdf.add_text_list([" ", "Signifikanztests (Pearson r, gelaggt):"] + sig_tests_lag_list, is_code=True)
    else:
         pdf.add_text_list(["Keine gelaggten Korrelationen berechnet oder verfügbar."], is_code=False)

    # 4. Granger-Kausalität
    pdf.add_page() # Neue Seite für Granger
    pdf.section_title("4. Granger-Kausalitätstests")
    pdf.add_text_list(granger_results_list, is_code=True)

    # Speichern
    try:
        pdf.output(file_paths["pdf_report"])
        print(f"\nPDF-Bericht gespeichert unter: {file_paths['pdf_report']}")
    except Exception as e:
        print(f"FEHLER beim Speichern der PDF: {e}")
        print(f"Traceback: {traceback.format_exc()}")


# --- START HAUPTSKRIPT ---
# ... (Der Rest des Skripts von Zeile 187 bis zum Ende bleibt unverändert,
#      stellt aber sicher, dass die Ergebnisse wie oben gezeigt gesammelt werden) ...

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
activity_cols_mapping = {
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
df_final.rename(columns={k: v for k, v in activity_cols_mapping.items() if k in df_final.columns}, inplace=True)
activity_cols_renamed = [v for k, v in activity_cols_mapping.items() if v in df_final.columns]

# Gesamtaktivität berechnen
post_cols = [col for col in ['Posts_WSB', 'Posts_GME', 'Posts_Superstonk'] if col in df_final.columns]
comment_cols = [col for col in ['Comments_WSB', 'Comments_GME', 'Comments_Superstonk'] if col in df_final.columns]
sub_cols = [col for col in ['Subs_WSB', 'Subs_GME', 'Subs_Superstonk'] if col in df_final.columns]

if post_cols:
    df_final['Total_Posts'] = df_final[post_cols].sum(axis=1, skipna=False)
if comment_cols:
    df_final['Total_Comments'] = df_final[comment_cols].sum(axis=1, skipna=False)
if sub_cols:
    df_final['Total_Subscribers'] = df_final[sub_cols].sum(axis=1, skipna=False)

# NaN Behandlung
stock_cols_nan = ['Price', 'Volume']
activity_cols_to_ffill = activity_cols_renamed + ['Total_Posts', 'Total_Comments', 'Total_Subscribers']
activity_cols_to_ffill_existing = [col for col in activity_cols_to_ffill if col in df_final.columns]
print(f"Wende Forward Fill (ffill) auf folgende Spalten an: {stock_cols_nan + activity_cols_to_ffill_existing}")
df_final[stock_cols_nan + activity_cols_to_ffill_existing] = df_final[stock_cols_nan + activity_cols_to_ffill_existing].ffill()

sentiment_upvote_cols_nan = ['Avg_Sentiment_Final', 'Avg_Sentiment_VADER', 'Avg_Subjectivity', 'Post_Count',
                             'Median_Upvotes', 'Avg_Upvotes', 'Sum_Upvotes']
sentiment_upvote_cols_nan_existing = [col for col in sentiment_upvote_cols_nan if col in df_final.columns]
print(f"Fülle verbleibende NaNs mit 0 für: {sentiment_upvote_cols_nan_existing}")
df_final[sentiment_upvote_cols_nan_existing] = df_final[sentiment_upvote_cols_nan_existing].fillna(0)

df_final.dropna(subset=['Price'], inplace=True)
print(f"Anzahl Zeilen nach NaN-Behandlung: {len(df_final)}")

# Preisänderungen, Log-Return, Log-Preis berechnen
df_final['Price_Change'] = df_final['Price'].diff()
df_final['Log_Return'] = np.log(df_final['Price'] / df_final['Price'].shift(1))
if 'Price' in df_final.columns:
    df_final['Price_Log'] = np.log(df_final['Price'].astype(float).replace(0, np.nan).clip(lower=0.0001))
    df_final['Price_Log'].fillna(method='ffill', inplace=True)
    df_final['Price_Log'].fillna(method='bfill', inplace=True)
    print("Logarithmierte Preisspalte 'Price_Log' hinzugefügt.")
cols_to_fill_zero = ['Price_Change', 'Log_Return']
if 'Price_Log' in df_final.columns: cols_to_fill_zero.append('Price_Log')
df_final[cols_to_fill_zero] = df_final[cols_to_fill_zero].fillna(0)
df_final.replace([np.inf, -np.inf], 0, inplace=True)

# Lagged Variables erstellen
metrics_to_lag = ['Avg_Sentiment_Final', 'Avg_Sentiment_VADER', 'Post_Count', 'Total_Posts',
                   'Total_Comments', 'Total_Subscribers', 'Median_Upvotes', 'Sum_Upvotes', 'Volume', 'Log_Return']
metrics_to_lag_existing = [m for m in metrics_to_lag if m in df_final.columns]
print(f"Erstelle Lagged Variables für {LAG_DAYS} Tage für: {metrics_to_lag_existing}")
for metric in metrics_to_lag_existing:
    for i in range(1, LAG_DAYS + 1):
        df_final[f'{metric}_Lag{i}'] = df_final[metric].shift(i)

df_final.dropna(inplace=True)
print(f"Anzahl Zeilen nach Lag-Erstellung und NaN-Drop: {len(df_final)}")

# Speichere das finale, vorbereitete DataFrame
try:
    df_final.to_csv(FILE_PATHS["comprehensive_data"], sep=';', encoding='utf-8-sig', index=True)
    print(f"\nUmfassendes vorbereitetes DataFrame gespeichert: {FILE_PATHS['comprehensive_data']}")
    print("-> Dieses CSV enthält alle gemergten, berechneten und gelaggten Daten für eigene Visualisierungen.")
except Exception as e:
    print(f"FEHLER beim Speichern des umfassenden DataFrames: {e}")

# --- 2a. Stationaritätsprüfung ---
print("\n--- 2a. Stationaritätsprüfung (ADF-Test) ---")
vars_for_stationarity_check = ['Price_Log', 'Log_Return', 'Volume', 'Avg_Sentiment_Final', 'Post_Count', 'Total_Posts']
adf_results_text_dict = {} # Store text results and boolean for PDF
stationarity_results = {}  # Store boolean results for Granger logic
for var in vars_for_stationarity_check:
    if var in df_final.columns:
        adf_output_list, is_stationary = run_adf_test(df_final[var], var)
        # *** KORREKTUR HIER: Speichere TUPEL (Liste, Boolean) ***
        adf_results_text_dict[var] = (adf_output_list, is_stationary)
        stationarity_results[var] = is_stationary # Behalte dies für Granger Logik
    else:
        print(f"Warnung: Spalte {var} nicht gefunden für ADF-Test.")
        # *** KORREKTUR HIER: Speichere auch hier ein Tupel ***
        adf_results_text_dict[var] = ([f"--- ADF Test für: {var} ---", "Spalte nicht gefunden."], False)

# --- 3. Deskriptive Statistiken ---
print("\n--- 3. Deskriptive Statistiken ---")
desc_cols = ['Price_Log', 'Volume', 'Price_Change', 'Log_Return', 'Avg_Sentiment_Final', 'Post_Count', 'Total_Posts', 'Median_Upvotes']
desc_cols_existing = [col for col in desc_cols if col in df_final.columns]
desc_stats = pd.DataFrame()
if desc_cols_existing:
    desc_stats = df_final[desc_cols_existing].describe() # Speichere das DataFrame für PDF
    desc_stats_print = desc_stats.copy()
    for col in ['Volume', 'Median_Upvotes', 'Total_Posts', 'Total_Comments', 'Total_Subscribers']:
         if col in desc_stats_print.columns:
             # Versuche, sicher zu formatieren
             try:
                 numeric_col = pd.to_numeric(desc_stats_print[col], errors='coerce')
                 desc_stats_print[col] = numeric_col.apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "NaN")
             except Exception: pass # Ignoriere Formatierungsfehler
    print("\nDeskriptive Statistiken:")
    print(desc_stats_print.round(4))
else:
    print("Keine Spalten für deskriptive Statistiken gefunden.")

# --- 4. Korrelationsanalyse ---
print("\n--- 4. Korrelationsanalyse ---")
corr_cols_concurrent = ['Price_Log', 'Volume', 'Price_Change', 'Log_Return', 'Avg_Sentiment_Final',
                        'Post_Count', 'Total_Posts', 'Median_Upvotes', 'Sum_Upvotes']
corr_cols_concurrent = [col for col in corr_cols_concurrent if col in df_final.columns]

correlation_matrix_concurrent = pd.DataFrame()
sig_tests_concurrent_results_list = []
lagged_correlation_matrix_results = None # Initialisiere als None
sig_tests_lagged_results_list = []

if len(corr_cols_concurrent) > 1:
    correlation_matrix_concurrent = df_final[corr_cols_concurrent].corr()
    print("\nKorrelationsmatrix (gleichzeitig):")
    print(correlation_matrix_concurrent.round(3))

    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix_concurrent, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 9})
    plt.title('Korrelationsmatrix (gleichzeitige Metriken)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    heatmap_path = FILE_PATHS["heatmap_image"]
    plt.savefig(heatmap_path)
    print(f"Heatmap gespeichert: {heatmap_path}")
    plt.close()

    print("\nSignifikanztests (Pearson r) für ausgewählte Paare:")
    pairs_to_test = [('Avg_Sentiment_Final', 'Log_Return'), ('Post_Count', 'Log_Return'),
                     ('Post_Count', 'Volume'), ('Total_Posts', 'Volume')]
    for col1, col2 in pairs_to_test:
        if col1 in df_final.columns and col2 in df_final.columns:
            df_pair = df_final[[col1, col2]].dropna()
            result_str = f"  {col1} vs {col2}: "
            if len(df_pair) > 1:
                 try:
                     res = stats.pearsonr(df_pair[col1], df_pair[col2])
                     significance_tag = ' (SIGNIFIKANT)' if res[1] < SIGNIFICANCE_LEVEL else ''
                     result_str += f"r={res[0]:.3f}, p={res[1]:.4f}{significance_tag}"
                 except ValueError as pe:
                     result_str += f"FEHLER bei Pearson r: {pe}"
            else:
                 result_str += "Zu wenige Datenpunkte für Korrelationstest."
            print(result_str)
            sig_tests_concurrent_results_list.append(result_str)

print("\nKorrelationsmatrix (gelaggte Reddit-Metriken vs. aktuelle Aktienmetriken):")
lagged_cols_exist = [f'{m}_Lag1' for m in metrics_to_lag_existing if f'{m}_Lag1' in df_final.columns]
stock_cols_for_lag = [col for col in ['Price_Change', 'Log_Return', 'Volume'] if col in df_final.columns]
cols_for_lagged_corr = stock_cols_for_lag + lagged_cols_exist

if len(cols_for_lagged_corr) > 1 and lagged_cols_exist and stock_cols_for_lag:
    lagged_correlation_matrix = df_final[cols_for_lagged_corr].corr()
    lagged_correlation_matrix_results = lagged_correlation_matrix.loc[stock_cols_for_lag, lagged_cols_exist]
    print(lagged_correlation_matrix_results.round(3))

    print("\nSignifikanztests (Pearson r) für ausgewählte gelaggte Paare:")
    lagged_pairs_to_test = [('Avg_Sentiment_Final_Lag1', 'Log_Return'), ('Post_Count_Lag1', 'Log_Return'),
                            ('Post_Count_Lag1', 'Volume'), ('Total_Posts_Lag1', 'Volume')]
    for col1_lag, col2_current in lagged_pairs_to_test:
         if col1_lag in df_final.columns and col2_current in df_final.columns:
             df_pair_lag = df_final[[col1_lag, col2_current]].dropna()
             result_str = f"  {col1_lag} vs {col2_current}: "
             if len(df_pair_lag) > 1:
                 try:
                     res_lag = stats.pearsonr(df_pair_lag[col1_lag], df_pair_lag[col2_current])
                     significance_tag = ' (SIGNIFIKANT)' if res_lag[1] < SIGNIFICANCE_LEVEL else ''
                     result_str += f"r={res_lag[0]:.3f}, p={res_lag[1]:.4f}{significance_tag}"
                 except ValueError as pe:
                      result_str += f"FEHLER bei Pearson r: {pe}"
             else:
                 result_str += "Zu wenige Datenpunkte für Korrelationstest."
             print(result_str)
             sig_tests_lagged_results_list.append(result_str)
else:
     print("Nicht genügend Spalten für gelaggte Korrelationsanalyse vorhanden.")

# --- 5. Granger-Kausalitätstest ---
print("\n--- 5. Granger-Kausalitätstest ---")
warnings.filterwarnings("ignore")

granger_pairs = [
    ('Log_Return', 'Avg_Sentiment_Final'), ('Log_Return', 'Post_Count'), ('Log_Return', 'Total_Posts'),
    ('Volume', 'Avg_Sentiment_Final'), ('Volume', 'Post_Count'), ('Volume', 'Total_Posts')
]

granger_results_list = [] # Store results for PDF
for target, cause in granger_pairs:
    if target in df_final.columns and cause in df_final.columns:
        current_test_results = [f"--- Test: Beeinflusst '{cause}' die Variable '{target}'? (maxlag={LAG_DAYS}) ---"]
        print(f"\nTeste: Beeinflusst '{cause}' die Variable '{target}'? (maxlag={LAG_DAYS})")
        data_granger = df_final[[target, cause]].dropna()
        if len(data_granger) < 30:
            msg = f" -> Zu wenige Datenpunkte ({len(data_granger)}) nach NaN-Drop für Test."
            print(msg)
            current_test_results.append(msg)
            granger_results_list.extend(current_test_results)
            continue

        target_stationary = stationarity_results.get(target, False)
        cause_stationary = stationarity_results.get(cause, False)
        if not target_stationary or not cause_stationary:
             warning_msg = f" -> WARNUNG: Mind. eine Zeitreihe ({target}: {target_stationary}, {cause}: {cause_stationary}) nicht stationär. Ergebnisse evtl. unzuverlässig!"
             print(warning_msg)
             current_test_results.append(warning_msg)

        try:
             test_result = grangercausalitytests(data_granger, maxlag=LAG_DAYS, verbose=False)
             significant_lags = []
             for lag in range(1, LAG_DAYS + 1):
                 p_value_f = test_result[lag][0]['ssr_ftest'][1]
                 if p_value_f < SIGNIFICANCE_LEVEL:
                     significant_lags.append((lag, p_value_f))

             if significant_lags:
                 msg1 = f" -> Signifikante Granger-Kausalität gefunden (p < {SIGNIFICANCE_LEVEL}) bei folgenden Lags:"
                 print(msg1)
                 current_test_results.append(msg1)
                 for lag, p_val in significant_lags:
                     msg2 = f"    Lag {lag}: p={p_val:.4f}"
                     print(msg2)
                     current_test_results.append(msg2)
             else:
                 msg = f" -> Keine signifikante Granger-Kausalität gefunden (alle p >= {SIGNIFICANCE_LEVEL}) für getestete Lags."
                 print(msg)
                 current_test_results.append(msg)
        except Exception as e:
            error_msg = f"FEHLER beim Granger-Test für {target} ~ {cause}: {e}"
            print(error_msg)
            current_test_results.append(error_msg)
        granger_results_list.extend(current_test_results)
    else:
        msg = f"Warnung: Mindestens eine Spalte für Granger-Test ({target}, {cause}) nicht gefunden."
        print(msg)
        granger_results_list.append(msg)

warnings.filterwarnings("default")

# --- 6. PDF-Bericht Generieren ---
print("\n--- 6. Generiere PDF-Bericht ---")
try:
    # Stelle sicher, dass die DataFrames für den Bericht existieren
    desc_stats_for_pdf = desc_stats if not desc_stats.empty else pd.DataFrame({'Info': ['Keine deskriptiven Statistiken verfügbar.']})
    corr_matrix_conc_for_pdf = correlation_matrix_concurrent if not correlation_matrix_concurrent.empty else pd.DataFrame({'Info': ['Keine gleichz. Korrelationen verfügbar.']})
    corr_matrix_lag_for_pdf = lagged_correlation_matrix_results # Kann None sein, wird in Funktion geprüft

    generate_pdf_report(adf_results_text_dict, desc_stats_for_pdf, corr_matrix_conc_for_pdf,
                        sig_tests_concurrent_results_list, corr_matrix_lag_for_pdf,
                        sig_tests_lagged_results_list, granger_results_list, FILE_PATHS)
except Exception as pdf_err:
    print(f"FEHLER beim Generieren des PDF-Berichts: {pdf_err}")
    print(f"Traceback: {traceback.format_exc()}")


# --- Abschluss ---
# (Bleibt unverändert)
print("\n\n--- Analyse abgeschlossen ---")
print(f"Ergebnisse (Heatmap, finale Daten-CSV, PDF-Report) wurden im Ordner '{FILE_PATHS['output_dir']}' gespeichert.")
print(f"Umfassende Daten für eigene Plots: {FILE_PATHS['comprehensive_data']}")
