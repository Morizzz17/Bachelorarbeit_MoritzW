# -*- coding: utf-8 -*-
"""
Gme Stock Data

@author: Moritz 

Dieses Skript lädt die Aktienkursdaten von GME für den Zeitraum 2020‑01‑01 bis 2025‑02‑10 und speichert sie in gme_stock_data.csv.
"""

import os
import pandas as pd
from bs4 import BeautifulSoup
import re
import csv

# === EINSTELLUNGEN ===
# Pfad zur lokal gespeicherten HTML-Datei von Yahoo Finance
HTML_FILE_PATH = r"C:\Users\Moriz\OneDrive - Hochschule München\Hochschule\Bachelorarbeit\GME_Coding\GME_Coding\data\GameStop Corp. (GME) Stock Historical Prices & Data_full.html"
# Zielpfad für die resultierende CSV-Datei
OUTPUT_CSV_PATH = r"C:\Users\Moriz\OneDrive - Hochschule München\Hochschule\Bachelorarbeit\GME_Coding\GME_Coding\results\gme_stock_data_full.csv"
# Sicherstellen, dass der Zielordner existiert
RESULTS_DIR = os.path.dirname(OUTPUT_CSV_PATH)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Hilfsfunktion zum Bereinigen von Zahlenstrings ---
def clean_numeric_string(num_str):
    """Entfernt Kommas und konvertiert in Float, gibt None bei Fehler."""
    if not isinstance(num_str, str):
        return None
    # Ignoriere spezielle Einträge wie 'Dividend' oder '-'
    if not re.match(r'^-?[\d,.]+$', num_str):
        return None
    try:
        # Entferne Kommas
        cleaned_str = num_str.replace(',', '')
        return float(cleaned_str)
    except ValueError:
        return None

# --- Hauptlogik ---
print(f"Lese HTML-Datei: {HTML_FILE_PATH}")

# Überprüfen, ob die HTML-Datei existiert
if not os.path.exists(HTML_FILE_PATH):
    print(f"FEHLER: HTML-Datei nicht gefunden unter: {HTML_FILE_PATH}")
    exit() # Beendet das Skript, wenn die Datei nicht existiert

stock_data = []

try:
    # HTML-Datei öffnen und parsen
    with open(HTML_FILE_PATH, 'r', encoding='utf-8', errors='replace') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Finde die Tabelle mit den historischen Daten
    data_table = soup.find('table', attrs={'data-test': 'historical-prices'})

    if not data_table:
        print("FEHLER: Konnte die Datentabelle in der HTML-Datei nicht finden.")
        print("Mögliche Ursachen: Struktur der Yahoo Finance Seite hat sich geändert oder die HTML-Datei ist beschädigt.")
        print("Versuche, *irgendeine* Tabelle zu finden (weniger zuverlässig)...")
        # Fallback: Nimm die erste gefundene Tabelle (kann falsch sein!)
        data_table = soup.find('table')
        if not data_table:
             print("FEHLER: Konnte überhaupt keine Tabelle in der HTML-Datei finden.")
             exit()

    # Finde alle Zeilen (<tr>) im Tabellenkörper (<tbody>)
    rows = data_table.find('tbody').find_all('tr')
    print(f"Gefundene Datenzeilen (geschätzt): {len(rows)}")

    # Iteriere durch jede Zeile
    for row in rows:
        cols = row.find_all('td') # Finde alle Datenzellen (<td>) in der Zeile

        # Überprüfe, ob die Zeile genügend Spalten hat (typisch sind 7: Date, Open, High, Low, Close, Adj Close, Volume)
        if len(cols) >= 7:
            date_str = cols[0].get_text(strip=True)
            # 'Adj Close' als relevanten 'Price', da er Dividenden/Splits berücksichtigt
            adj_close_str = cols[5].get_text(strip=True)
            volume_str = cols[6].get_text(strip=True)

            # Bereinige die numerischen Werte
            price = clean_numeric_string(adj_close_str)
            volume = clean_numeric_string(volume_str)

            # Überspringe Zeilen, die keine gültigen Daten enthalten (z.B. Dividend-Zeilen)
            if price is not None and volume is not None:
                 try:
                     pd.to_datetime(date_str)
                     stock_data.append({
                         'Date': date_str,
                         'Price': price,
                         'Volume': int(volume) # Volumen ist normalerweise eine ganze Zahl
                     })
                 except (ValueError, TypeError):
                     print(f"WARNUNG: Ungültiges Datumsformat übersprungen: {date_str}")

except FileNotFoundError:
    print(f"FEHLER: Die Datei {HTML_FILE_PATH} wurde nicht gefunden.")
    exit()
except Exception as e:
    print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
    exit()

# --- Daten in DataFrame umwandeln und speichern ---
if not stock_data:
    print("FEHLER: Keine gültigen Aktiendaten aus der HTML-Datei extrahiert.")
    print("Bitte überprüfe die HTML-Datei und die Tabellenstruktur.")
else:
    # Erstelle einen Pandas DataFrame
    df = pd.DataFrame(stock_data)

    # Konvertiere die 'Date'-Spalte in Datetime-Objekte
    df['Date'] = pd.to_datetime(df['Date'])

    # Sortiere nach Datum (aufsteigend)
    df.sort_values(by='Date', inplace=True)

    # Speichere den DataFrame als CSV-Datei
    try:
        df.to_csv(
            OUTPUT_CSV_PATH,
            index=False,          
            sep=';',              
            encoding='utf-8-sig',
            quoting=csv.QUOTE_MINIMAL
        )
        print("\n✅ Aktienkursdaten erfolgreich extrahiert und gespeichert unter:")
        print(OUTPUT_CSV_PATH)
        print(f"Anzahl der gespeicherten Datensätze: {len(df)}")
    except Exception as e:
        print(f"FEHLER beim Speichern der CSV-Datei: {e}")