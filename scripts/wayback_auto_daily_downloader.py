# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 17:54:35 2025

WSB, SUPERSTONK, GME
@author: Moriz

Automatischer HTML-Snapshot-Downloader über Wayback CDX API
Speichert tägliche Snapshots von r/wallstreetbets/ im gewünschten Zeitraum.
"""

import os
import requests
import time
from datetime import datetime, timedelta
from tqdm import tqdm

# === Einstellungen ===
START_DATE = datetime(2021, 1, 1) # Superstonk wurde am 15.03.2021 erstellt, als Flucht aus r/wallstreetbets, nachdem Diskussionen über Gamestop verboten wurden
END_DATE = datetime(2025, 3, 24)
SAVE_DIR = r"C:\Users\Moriz\OneDrive - Finance Network - FNI® e.V\Bachelorarbeit\HTML-Snapshots\html_snapshots_gme"
FAILED_LOG_PATH = os.path.join(SAVE_DIR, "wayback_failed_log.txt")
os.makedirs(SAVE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

RATE_LIMIT_DELAY = 4  # Sekunden
CDX_RETRY_DELAY = 7   # für zusätzliche Pausen in Retries

# === Funktionen ===

def get_snapshot_timestamp_cdx(date: datetime, retries: int = 3) -> str | None:
    """CDX Snapshot Timestamp holen mit Retry"""
    url = (
        "https://web.archive.org/cdx/search/cdx"
        "?url=https://www.reddit.com/r/gme"
        f"&from={date.strftime('%Y%m%d')}&to={date.strftime('%Y%m%d')}"
        "&output=json&fl=timestamp&filter=statuscode:200&collapse=timestamp:8"
    )
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            data = response.json()
            if len(data) > 1:
                return data[1][0]
            return None
        except Exception as e:
            print(f"{date.strftime('%Y-%m-%d')}: ❌ Versuch {attempt}/{retries} fehlgeschlagen – {e}")
            if attempt < retries:
                sleep_time = CDX_RETRY_DELAY + attempt * 2
                print(f"⏳ Warte {sleep_time}s bis nächster Versuch...")
                time.sleep(sleep_time)
    return None

def download_snapshot(date: datetime, timestamp: str) -> bool:
    url = f"https://web.archive.org/web/{timestamp}/https://www.reddit.com/r/gme"
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        filepath = os.path.join(SAVE_DIR, f"{date.strftime('%Y-%m-%d')}.html")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(response.text)
        return True
    except Exception as e:
        print(f"{date.strftime('%Y-%m-%d')}: ❌ Fehler beim Speichern – {e}")
        return False

# === Hauptlauf ===

print(f"🔍 Lade Wayback Snapshots von {START_DATE.date()} bis {END_DATE.date()}...\n")
current_date = START_DATE
failed_dates = []

for _ in tqdm(range((END_DATE - START_DATE).days + 1)):
    date_str = current_date.strftime('%Y-%m-%d')
    timestamp = get_snapshot_timestamp_cdx(current_date)
    time.sleep(RATE_LIMIT_DELAY)

    if timestamp:
        if download_snapshot(current_date, timestamp):
            print(f"{date_str}: ✅ gespeichert")
        else:
            failed_dates.append(date_str)
            print(f"{date_str}: ❌ Fehler beim Download")
    else:
        failed_dates.append(date_str)
        print(f"{date_str}: ❌ Kein Snapshot gefunden")

    current_date += timedelta(days=1)

# === Fehlerlog schreiben ===
if failed_dates:
    with open(FAILED_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("Fehlgeschlagene Tage beim Wayback Download:\n")
        for date in failed_dates:
            f.write(f"{date}\n")
    print(f"\n⚠️ {len(failed_dates)} fehlgeschlagene Tage – siehe {FAILED_LOG_PATH}")

    # === Wiederholungs-Schleife ===
    print("\n🔁 Starte erneuten Versuch für fehlgeschlagene Tage...\n")
    still_failed = []
    for date_str in failed_dates:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        timestamp = get_snapshot_timestamp_cdx(date_obj)
        time.sleep(RATE_LIMIT_DELAY)

        if timestamp:
            if download_snapshot(date_obj, timestamp):
                print(f"{date_str}: ✅ zweiter Versuch erfolgreich")
            else:
                print(f"{date_str}: ❌ erneut fehlgeschlagen beim Download")
                still_failed.append(date_str)
        else:
            print(f"{date_str}: ❌ erneut kein Snapshot gefunden")
            still_failed.append(date_str)

    # Neues Fehlerlog
    if still_failed:
        with open(FAILED_LOG_PATH, "w", encoding="utf-8") as f:
            f.write("Auch im 2. Versuch fehlgeschlagene Tage:\n")
            for date in still_failed:
                f.write(f"{date}\n")
        print(f"\n⚠️ {len(still_failed)} Tage sind auch im 2. Versuch fehlgeschlagen.")
    else:
        os.remove(FAILED_LOG_PATH)
        print("\n✅ Alle Snapshots im 2. Versuch erfolgreich gespeichert.")
else:
    print("\n✅ Alle Snapshots im ersten Durchlauf erfolgreich gespeichert.")
    