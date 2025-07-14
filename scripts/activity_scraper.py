# -*- coding: utf-8 -*-
"""
Activity Scraper

@author: Moritz

Automatischer HTML-Snapshot-Downloader über Wayback CDX API
Speichert tägliche Snapshots von r/wallstreetbets, r/superstonk und r/gme im gewünschten Zeitraum.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import csv
import json
import re
import traceback
from bs4 import BeautifulSoup

# --- Settings ---
SUBREDDITS = ["wallstreetbets", "superstonk", "gme"]
START_DATE = datetime(2020, 12, 1)
END_DATE = datetime(2023, 12, 31)

API_BASE_URL = "https://subredditstats.com/api/subreddit"
SUBREDDIT_PAGE_URL_TEMPLATE = "https://subredditstats.com/r/{subreddit}"

STATS_MAP = {
    "subscribers": "api_subscriberCountTimeSeries",
    "posts_per_day": "html_postsPerHourTimeSeries",
    "comments_per_day": "html_commentsPerHourTimeSeries",
}

RESULTS_DIR = r"C:\Users\Moriz\OneDrive - Hochschule München\Hochschule\Bachelorarbeit\GME_Coding\GME_Coding\results"
OUTPUT_FILENAME = "subreddit_activity_stats.csv" # Dateiname
os.makedirs(RESULTS_DIR, exist_ok=True)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/json, text/html, */*'
}
# --- Ende Settings ---


def fetch_stat_data(subreddit, stat_key, data_source_signal):
    """
    Ruft Daten ab, entweder via API oder durch HTML-Scraping des eingebetteten JSON.
    """
    # Fall 1: API-Abruf (Subscribers)
    if data_source_signal == "api_subscriberCountTimeSeries":
        api_project_name = "subscriberCountTimeSeries"
        params = {'name': subreddit, 'project': api_project_name}
        url = API_BASE_URL
        print(f"  Fetching API: {url} mit Parametern {params}")
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and api_project_name in data:
                data_list = data[api_project_name]
                if isinstance(data_list, list):
                    processed_data = []
                    epoch_start = datetime(1970, 1, 1)
                    for entry in data_list:
                        if isinstance(entry, dict) and 'utcDay' in entry and 'count' in entry:
                            try:
                                dt_object = epoch_start + timedelta(days=entry['utcDay'])
                                processed_data.append([dt_object, entry['count']])
                            except (TypeError, ValueError, OSError) as e:
                                print(f"      Fehler bei API-Datumsverarbeitung für {entry}: {e}")
                                pass
                    return processed_data
                else: return []
            elif isinstance(data, dict) and not data: return []
            else: return []

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404: print(f"    INFO: API Endpunkt für '{api_project_name}' nicht gefunden (404).")
            else: print(f"!!! HTTP FEHLER {e.response.status_code} für {url} mit {params}: {e}")
            return []
        except requests.exceptions.RequestException as e: print(f"!!! Netzwerkfehler für API {url} mit {params}: {e}")
        except json.JSONDecodeError as e: print(f"!!! JSON-Fehler für API {url} mit {params}: {e}. Antwort: {response.text[:200]}...")
        except Exception as e: print(f"!!! Unbekannter FEHLER bei API {url} mit {params}: {e}")
        return []

    # Fall 2: HTML Scraping (Posts/Comments aus eingebettetem JSON)
    elif data_source_signal.startswith("html_"):
        timeseries_key = data_source_signal.replace("html_", "")
        url = SUBREDDIT_PAGE_URL_TEMPLATE.format(subreddit=subreddit)
        print(f"  Fetching HTML: {url} für '{stat_key}' (suche nach JSON key '{timeseries_key}')")
        try:
            response = requests.get(url, headers=HEADERS, timeout=45)
            response.raise_for_status()
            html_content = response.text
            soup = BeautifulSoup(html_content, 'lxml')
            script_tag = soup.find('script', id='embeddedSubredditData')

            if not script_tag:
                print(f"    FEHLER: Konnte <script id='embeddedSubredditData'> nicht im HTML finden für {url}")
                return []
            json_string = script_tag.string
            if not json_string:
                 print(f"    FEHLER: <script id='embeddedSubredditData'> war leer für {url}")
                 return []

            try:
                embedded_data = json.loads(json_string)
            except json.JSONDecodeError as e:
                print(f"!!! JSON-Parsing-Fehler beim Extrahieren der eingebetteten Daten für {url}: {e}")
                return []

            if timeseries_key not in embedded_data:
                 print(f"    FEHLER: Schlüssel '{timeseries_key}' nicht in eingebetteten JSON-Daten gefunden für {url}")
                 return []
            raw_timeseries_data = embedded_data[timeseries_key]
            if not isinstance(raw_timeseries_data, list):
                print(f"   FEHLER: Daten für Schlüssel '{timeseries_key}' sind keine Liste.")
                return []

            processed_data = []
            epoch_start = datetime(1970, 1, 1)
            value_key = None
            if timeseries_key == "commentsPerHourTimeSeries": value_key = "commentsPerHour"
            elif timeseries_key == "postsPerHourTimeSeries": value_key = "postsPerHour"
            else: value_key = "count"

            for entry in raw_timeseries_data:
                if isinstance(entry, dict) and 'utcDay' in entry and value_key in entry:
                    try:
                        dt_object = epoch_start + timedelta(days=entry['utcDay'])
                        count_value = entry[value_key]
                        if count_value is not None:
                             daily_count = count_value * 24
                             processed_data.append([dt_object, round(daily_count)])
                    except (TypeError, ValueError, OSError) as e:
                        print(f"      Fehler bei HTML-Datenverarbeitung für {entry}: {e}")
                        pass
            return processed_data

        except requests.exceptions.HTTPError as e: print(f"!!! HTTP FEHLER {e.response.status_code} beim Laden von {url}: {e}")
        except requests.exceptions.RequestException as e: print(f"!!! Netzwerkfehler beim Laden von {url}: {e}")
        except Exception as e:
            print(f"!!! Unbekannter FEHLER beim Verarbeiten der HTML von {url}: {e}")
            print(traceback.format_exc())
        return []

    else:
        print(f"    Unbekannter Abruftyp/Signal: {data_source_signal}")
        return []


# --- Hauptlogik ---
all_subreddit_dfs_processed = []

for sub in SUBREDDITS:
    print(f"\n--- Verarbeite Subreddit: {sub} ---")
    subreddit_metric_dfs = {}
    for stat_key, data_source_signal in STATS_MAP.items():
        raw_data = fetch_stat_data(sub, stat_key, data_source_signal)
        if raw_data:
            df_stat = pd.DataFrame(raw_data, columns=['date', stat_key])
            df_stat['date'] = pd.to_datetime(df_stat['date']).dt.date
            df_stat = df_stat.set_index('date')
            df_stat = df_stat[~df_stat.index.duplicated(keep='first')]
            df_stat.index = pd.to_datetime(df_stat.index) # Sicherstellen, dass Index Datetime ist
            subreddit_metric_dfs[stat_key] = df_stat
            print(f"    Daten für '{stat_key}' ({data_source_signal}) erfolgreich erhalten ({len(df_stat)} Einträge).")
        else:
            print(f"    Keine Daten für '{stat_key}' ({data_source_signal}) erhalten.")
        time.sleep(1.0)

    if subreddit_metric_dfs:
        df_sub_merged = pd.DataFrame()
        existing_keys_in_order = [k for k in STATS_MAP.keys() if k in subreddit_metric_dfs]
        for key in existing_keys_in_order:
            if not isinstance(subreddit_metric_dfs[key].index, pd.DatetimeIndex):
                 subreddit_metric_dfs[key].index = pd.to_datetime(subreddit_metric_dfs[key].index)

        if existing_keys_in_order:
            first_key = existing_keys_in_order[0]
            df_sub_merged = subreddit_metric_dfs[first_key]
            for stat_key in existing_keys_in_order[1:]:
                 df_to_merge = subreddit_metric_dfs[stat_key]
                 df_sub_merged = pd.merge(df_sub_merged, df_to_merge, left_index=True, right_index=True, how='outer')

        if not df_sub_merged.empty:
            # Filtere nach Datum (Index ist bereits Datetime)
            df_sub_filtered = df_sub_merged[(df_sub_merged.index >= START_DATE) & (df_sub_merged.index <= END_DATE)].copy() # .copy() um SettingWithCopyWarning zu vermeiden

            if not df_sub_filtered.empty:
                rename_dict = {}
                for col in df_sub_filtered.columns:
                    rename_dict[col] = f'{col}_{sub}'
                df_sub_filtered.rename(columns=rename_dict, inplace=True)
                all_subreddit_dfs_processed.append(df_sub_filtered)
                print(f"  Daten für {sub} erfolgreich zusammengeführt und gefiltert.")
            else: print(f"  Keine Daten für {sub} im gewünschten Zeitraum.")
        else: print(f"  Keine DataFrames zum Mergen für Subreddit {sub} vorhanden.")
    else: print(f"  Konnte keine Statistik-Daten für {sub} erfolgreich abrufen.")


# --- Kombiniere alle Subreddit-DataFrames zu einem großen Wide-Format DataFrame ---
if all_subreddit_dfs_processed:
    final_wide_df = pd.concat(all_subreddit_dfs_processed, axis=1, join='outer')
    final_wide_df.sort_index(inplace=True)

    # --- NEU: Forward Fill Logik ---
    print("\n--- Fülle fehlende Werte (Forward Fill) ---")

    # Finde das erste Datum mit Daten für Superstonk (wichtig für die Ausnahme)
    superstonk_start_date = None
    superstonk_cols = [col for col in final_wide_df.columns if col.endswith('_superstonk')]
    if superstonk_cols:
        min_dates = []
        for col in superstonk_cols:
            first_valid_idx = final_wide_df[col].first_valid_index()
            if first_valid_idx is not None:
                 min_dates.append(first_valid_idx)
        if min_dates:
           superstonk_start_date = min(min_dates)
           print(f"  Erstes Datum mit Daten für Superstonk gefunden: {superstonk_start_date.strftime('%Y-%m-%d')}")
        else:
             print("  WARNUNG: Konnte kein Startdatum für Superstonk finden (keine Daten?). Fülle normal auf.")
    else:
         print("  INFO: Keine Superstonk-Spalten gefunden, fülle normal auf.")


    # Iteriere durch alle Spalten und wende ffill an
    filled_cols_count = 0
    for sub in SUBREDDITS:
        for stat in STATS_MAP.keys():
            col_name = f'{stat}_{sub}'
            if col_name in final_wide_df.columns:
                original_nan_count = final_wide_df[col_name].isna().sum()
                if original_nan_count > 0:
                     # Speichere NaN-Maske *vor* dem Fill für Superstonk-Korrektur
                     nan_mask_before_fill = final_wide_df[col_name].isna()

                     # Führe ffill durch
                     final_wide_df[col_name] = final_wide_df[col_name].ffill()
                     filled_count = original_nan_count - final_wide_df[col_name].isna().sum()

                     # Korrektur für Superstonk: Setze Werte VOR dem Startdatum zurück auf NA
                     if sub == 'superstonk' and superstonk_start_date is not None:
                          mask_to_reset = (final_wide_df.index < superstonk_start_date) & nan_mask_before_fill
                          final_wide_df.loc[mask_to_reset, col_name] = pd.NA
                          corrected_count = mask_to_reset.sum()
                          if corrected_count > 0:
                              print(f"    Korrektur für '{col_name}': {corrected_count} Werte vor {superstonk_start_date.strftime('%Y-%m-%d')} zurück auf NA gesetzt.")
                              filled_count -= corrected_count # Aktualisiere die gezählten Füllungen

                     if filled_count > 0:
                         print(f"    Spalte '{col_name}' gefüllt: {filled_count} Werte.")
                         filled_cols_count += 1
                # else: # Optional: Meldung, wenn keine NaNs vorhanden waren
                #      print(f"    Spalte '{col_name}' hatte keine fehlenden Werte.")

    if filled_cols_count == 0:
        print("  Keine fehlenden Werte zum Füllen gefunden.")
    # --- ENDE: Forward Fill Logik ---


    # Setze den Index zurück, um 'date' als Spalte zu haben
    final_wide_df.reset_index(inplace=True)
    final_wide_df.rename(columns={'index': 'date'}, inplace=True)

    # Sortiere Spalten logisch
    final_cols_order = ['date']
    for sub in SUBREDDITS:
        for stat in STATS_MAP.keys():
            col_name = f'{stat}_{sub}'
            if col_name in final_wide_df.columns:
                final_cols_order.append(col_name)

    final_cols_order = [col for col in final_cols_order if col in final_wide_df.columns]
    filtered_df_output = final_wide_df[final_cols_order].copy() # .copy() hier auch sinnvoll

    # Formatieren der Datumsspalte für die CSV
    if 'date' in filtered_df_output.columns:
       filtered_df_output['date'] = pd.to_datetime(filtered_df_output['date']).dt.strftime('%d/%m/%Y')

    # Konvertiere numerische Spalten in Integer, wo möglich (nach dem Füllen)
    # Behandle mögliche verbleibende NAs (z.B. am Anfang oder bei Superstonk)
    print("\n--- Konvertiere Spalten zu Ganzzahlen (wenn möglich) ---")
    for col in filtered_df_output.columns:
         if col != 'date':
             try:
                 # Versuche, in nullable Integer zu konvertieren
                 filtered_df_output[col] = filtered_df_output[col].astype(pd.Int64Dtype())
                 print(f"    Spalte '{col}' erfolgreich zu Int64 konvertiert.")
             except (TypeError, ValueError) as e:
                 # Wenn nicht möglich (z.B. float oder immer noch NAs), behalte den Typ bei
                 print(f"    Spalte '{col}' konnte nicht zu Int64 konvertiert werden (Typ: {filtered_df_output[col].dtype}): {e}")
                 pass # Behalte den aktuellen Typ bei (wahrscheinlich float oder object)


    output_path = os.path.join(RESULTS_DIR, OUTPUT_FILENAME)
    print(f"\nSpeichere kombinierte Breitformat-Daten nach: {output_path} ({len(filtered_df_output)} Zeilen)")
    try:
        # Verwende float_format, um '.0' bei ganzen Zahlen zu vermeiden, falls sie doch float sind
        filtered_df_output.to_csv(
            output_path,
            index=False,
            sep=";",
            encoding="utf-8-sig",
            quoting=csv.QUOTE_MINIMAL, # Minimales Quoting ist oft sauberer
            decimal='.', # Standard Dezimaltrennzeichen
            # float_format='%.0f' # Zeige Floats ohne Nachkommastellen an, wenn sie ganzzahlig sind
        )
        print("✅ CSV-Datei erfolgreich gespeichert.")
    except Exception as e:
        print(f"!!! FEHLER beim Speichern der CSV-Datei: {e}")
else:
    print("\n⚠️ Keine Daten von den Subreddits zum Kombinieren vorhanden, keine CSV erstellt.")