# Bachelor Thesis: GameStop - A Symptom of Systemic Weaknesses

Dieses Repository enthält den gesamten Quellcode, die Daten und die Analyse für die Bachelorarbeit mit dem Titel: **"GameStop – Ein Symptom systematischer Schwächen im Finanzmarkt: Eine empirische Analyse von Short Selling, Retail-Investoren und Social Media"**.

## 📝 Projektübersicht

Dieses Projekt führt eine empirische Analyse durch, um den Zusammenhang zwischen den Aktivitäten in Reddit-Communities (insbesondere r/wallstreetbets, r/gme, r/Superstonk) und den Marktbewegungen der GameStop-Aktie (GME) zu untersuchen. Der Fokus liegt darauf, quantitative Belege für den Einfluss von Social Media und koordinierten Retail-Investoren auf den Aktienkurs, das Handelsvolumen und die Volatilität zu finden.

Der Workflow umfasst die Datenerhebung aus verschiedenen Quellen, die Bereinigung und Aufbereitung der Daten, die Sentiment-Analyse von Textdaten sowie eine umfassende statistische Auswertung mittels Korrelations- und Kausalitätsanalysen.

### Repository-Inhalte

*   `/Skripte:` Enthält alle Python-Skripte (`.py`), die für den gesamten Prozess von der Datenerhebung bis zur finalen Analyse verwendet wurden.
*   `/Daten:` Enthält die Rohdaten sowie die aufbereiteten CSV-Dateien, die von den Skripten generiert und verwendet werden. Das zentrale Ergebnis ist die `COMPREHENSIVE_ANALYSIS_DATA.csv`.
*   `/Ergebnisse:` Enthält die finalen Ergebnisse der Analyse wie Grafiken (z.B. `heatmap_concurrent_correlation.png`) und den exportierten PDF-Bericht.
*   `Bachelorarbeit.pdf:` (Optional) Die finale schriftliche Ausarbeitung der Bachelorarbeit.

## 🎓 Informationen zur Abschlussarbeit

*   **Autor:** Moritz Waldmann
*   **Hochschule:** Hochschule München
*   **Studiengang:** Betriebswirtschaft
*   **Thema:** GameStop – Ein Symptom systematischer Schwächen im Finanzmarkt: Eine empirische Analyse von Short Selling, Retail-Investoren und Social Media
*   **Betreuer:** [Name des Betreuers]
*   **Datum:** [Datum der Abgabe]

## 🚀 Erste Schritte

Um dieses Projekt lokal nachzubauen und die Analysen auszuführen, sind folgende Schritte erforderlich.

### Voraussetzungen

*   Python 3.8 oder höher
*   Pip (Python Package Installer)
*   Git für das Klonen des Repositories

### Installation & Einrichtung

1.  **Repository klonen:**
    ```bash
    git clone https://github.com/Morizzz17/Bachelorarbeit_MoritzW.git
    cd Bachelorarbeit_MoritzW
    ```

2.  **Virtuelle Umgebung erstellen und aktivieren (empfohlen):**
    *Dadurch werden die Projekt-Abhängigkeiten von anderen Python-Projekten isoliert.*
    ```bash
    # Für Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Für macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Abhängigkeiten installieren:**
    Es wird empfohlen, eine `requirements.txt`-Datei mit allen notwendigen Paketen zu erstellen. Die wichtigsten sind:
    *   `pandas`
    *   `numpy`
    *   `matplotlib`
    *   `seaborn`
    *   `statsmodels`
    *   `yfinance` (für Aktiendaten)
    *   `praw` (für Reddit-Daten)

    Installieren Sie diese mit:
    ```bash
    pip install pandas numpy matplotlib seaborn statsmodels yfinance praw
    ```

### Ausführung

Die Skripte sind so konzipiert, dass sie in einer bestimmten Reihenfolge ausgeführt werden, um die Datenpipeline abzubilden.

1.  **Datenerhebung:**
    Führen Sie die Scraper-Skripte aus, um die Rohdaten zu sammeln. *Hinweis: Dies kann aufgrund von API-Limits und der Datenmenge sehr lange dauern. Alternativ können die bereits erhobenen Rohdaten verwendet werden.*
    *   `wayback_auto_daily_downloader.py`
    *   `activity_scraper.py`
    *   `gme_stock_data.py`

2.  **Datenaufbereitung und -analyse:**
    Nach der Datenerhebung führen Sie die Verarbeitungs- und Analyse-Skripte aus.
    *   `data_cleaning.py` & `local_scraper_unified.py`: Bereinigen und konsolidieren die gesammelten Rohdaten.
    *   `sentiment_analysis.py`: Führt die Sentiment-Analyse durch und erstellt tägliche aggregierte Sentiment-Werte.

3.  **Statistische Auswertung:**
    Das Hauptskript `stat_relevance.py` führt alle aufbereiteten Daten zusammen und führt die finale statistische Analyse durch.
    ```bash
    python stat_relevance.py
    ```
    Dieses Skript generiert die `COMPREHENSIVE_ANALYSIS_DATA.csv` und gibt die statistischen Ergebnisse auf der Konsole aus.

4.  **Export der Ergebnisse:**
    Das Skript `stat_relevance_pdf_export.py` kann verwendet werden, um die wichtigsten Ergebnisse und Grafiken in einem PDF-Bericht zusammenzufassen.

## 🛠️ Code-Beschreibung

Die Skripte sind modular aufgebaut, um die verschiedenen Phasen des Projekts abzubilden.

### `activity_scraper.py`
*Dieses Skript sammelt Metriken zur Nutzeraktivität wie die Anzahl der Abonnenten und täglichen Posts aus den relevanten Subreddits.*

**Details:** Nutzt wahrscheinlich die `PRAW`-Bibliothek, um auf die Reddit-API zuzugreifen. Es iteriert durch eine Liste von Ziel-Subreddits (r/wallstreetbets, r/gme, r/Superstonk), ruft Metadaten wie die aktuelle Abonnentenzahl und die neuesten Posts ab und aggregiert diese Daten auf einer täglichen Basis in einer CSV-Datei.

### `data_cleaning.py`
*Dieses Skript bereinigt die Rohdaten, indem es beispielsweise Duplikate entfernt, Formate vereinheitlicht und fehlende Werte behandelt.*

**Details:** Lädt die rohen Post- und Kommentardaten, entfernt doppelte Einträge, konvertiert Zeitstempel in ein einheitliches Datumsformat und bereinigt Textdaten (z.B. Entfernung von URLs) als Vorbereitung für die Sentiment-Analyse.

### `gme_stock_data.py`
*Dieses Skript ruft die historischen Aktienkurs- und Volumendaten für GameStop (GME) von einer Finanzdatenquelle ab.*

**Details:** Verwendet die `yfinance`-Bibliothek, um für den Ticker "GME" und einen definierten Zeitraum die täglichen Daten (Open, High, Low, Close, Volume) herunterzuladen und in einer sauberen CSV-Datei (`gme_stock_data.csv`) zu speichern.

### `local_scraper_unified.py`
*Dies ist eine konsolidierte Version, die mehrere lokale Datenverarbeitungsschritte zu einem einheitlichen Skript zusammenfasst.*

**Details:** Dient als Wrapper oder kombiniertes Skript, das verschiedene lokal gespeicherte Datenquellen (z.B. heruntergeladene JSON- oder CSV-Dumps) durchsucht. Es filtert nach relevanten Keywords oder Zeiträumen und harmonisiert die Daten aus verschiedenen Quellen zu einem einheitlichen Format.

### `sentiment_analysis.py`
*Dieses Skript analysiert die gesammelten Reddit-Texte, um daraus tägliche Sentiment-Werte zu berechnen.*

**Details:** Lädt die bereinigten Textdaten (Posts, Kommentare), initialisiert einen Sentiment-Analyzer (VADER) und iteriert durch jeden Text, um einen Sentiment-Score zu berechnen. Die Scores werden auf täglicher Basis aggregiert und als Durchschnittswerte (`Avg_Sentiment_VADER`, `Avg_Sentiment_Final`) in einer CSV-Datei gespeichert.

### `stat_relevance.py`
*Dieses Skript führt die statistische Hauptanalyse durch, indem es die Reddit- und Aktiendaten zusammenführt und Korrelationen sowie Kausalitätstests berechnet.*

**Details:** Dies ist das zentrale Analyse-Skript. Es lädt die Ergebnisse der vorherigen Skripte, führt sie in einem umfassenden DataFrame zusammen und führt ein Feature Engineering durch (Berechnung von `Price_Change`, `Log_Return` und verzögerten Variablen). Es führt ADF-Tests, Korrelationsanalysen und Granger-Kausalitätstests durch und gibt deren Ergebnisse aus. Der finale, aufbereitete DataFrame wird als `COMPREHENSIVE_ANALYSIS_DATA.csv` gespeichert.

### `stat_relevance_pdf_export.py`
*Dieses Skript exportiert die Ergebnisse und Visualisierungen der statistischen Analyse in ein formatiertes PDF-Dokument.*

**Details:** Lädt die Ergebnisse und Grafiken aus dem Analyseprozess und verwendet Bibliotheken wie `matplotlib` und `FPDF`, um einen strukturierten Bericht mit den wichtigsten Tabellen und Grafiken zu erstellen.

### `wayback_auto_daily_downloader.py`
*Dieses Skript ist darauf ausgelegt, automatisiert tägliche Daten-Snapshots von einer Internet-Archivquelle wie der Wayback Machine herunterzuladen.*

**Details:** Dient dazu, historische Daten von Reddit zu sammeln, insbesondere für Zeiträume, in denen die offizielle API keine vollständigen Daten mehr liefert.

## 📈 Methodik

*   **Forschungsdesign:** Mixed-Methods-Ansatz, mit einem Schwerpunkt auf quantitativer Zeitreihenanalyse, ergänzt durch qualitative Experteninterviews.
*   **Datenerhebungsmethoden:** Primärdatenerhebung von Reddit mittels API-Zugriff (`PRAW`, `Pushshift`) und Sekundärdatenerhebung von Finanzdaten (`yfinance`).
*   **Stichprobe / Auswahlverfahren:** Vollerhebung aller Posts/Kommentare in den Subreddits r/wallstreetbets, r/gme und r/Superstonk innerhalb des definierten Zeitraums sowie der täglichen GME-Kursdaten.
*   **Operationalisierung der Variablen:** Die Variablen wurden in Aktienmarkt- (Preis, Volumen, Log-Return), Reddit-Sentiment-, Reddit-Aktivitäts- und Reddit-Engagement-Metriken unterteilt und auf täglicher Basis aggregiert. Lagged-Variablen wurden zur Analyse zeitversetzter Effekte erstellt.
*   **Statistische Verfahren / Auswertungsmethoden:**
    *   Deskriptive Statistik
    *   Zeitreihen-Visualisierung
    *   Stationaritätstests (Augmented Dickey-Fuller)
    *   Korrelationsanalyse (Pearson r) für gleichzeitige und gelaggte Variablen
    *   Granger-Kausalitätstests zur Untersuchung der prädiktiven Kraft der Reddit-Metriken
