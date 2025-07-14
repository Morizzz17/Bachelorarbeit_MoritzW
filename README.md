# Bachelor Thesis: GameStop & Social Media

Dieses Repository enthält den gesamten Quellcode, die Daten und die Analyse für die Bachelorarbeit mit dem Titel: **"GameStop – Ein Symptom systematischer Schwächen im Finanzmarkt: Eine empirische Analyse von Short Selling, Retail-Investoren und Social Media"**.

## 📝 Projektübersicht

Dieses Projekt führt eine empirische Analyse durch, um den Zusammenhang zwischen den Aktivitäten in Reddit-Communities (insbesondere r/wallstreetbets, r/gme, r/Superstonk) und den Marktbewegungen der GameStop-Aktie (GME) zu untersuchen. Der Fokus liegt darauf, quantitative Belege für den Einfluss von Social Media und koordinierten Retail-Investoren auf den Aktienkurs, das Handelsvolumen und die Volatilität zu finden.

Der Workflow ist in mehrere Phasen unterteilt:
1.  **Datenerhebung:** Automatisches Herunterladen von historischen Reddit-Seiten und Abrufen von Aktivitäts-Metriken sowie Aktienkursdaten.
2.  **Datenverarbeitung:** Parsen der HTML-Dateien, Extraktion von Posts, Bereinigung der Textdaten und Durchführung einer Sentiment-Analyse.
3.  **Analyse:** Zusammenführen aller Datenquellen und Durchführung statistischer Tests (Korrelation, Granger-Kausalität) zur Überprüfung der aufgestellten Hypothesen.

### Repository-Inhalte

*   `/Skripte:` Enthält alle Python-Skripte (`.py`), die für den gesamten Prozess von der Datenerhebung bis zur finalen Analyse verwendet wurden.
*   `/Daten:` Enthält die Rohdaten (z.B. heruntergeladene HTML-Dateien) sowie die aufbereiteten CSV-Dateien, die von den Skripten generiert und verwendet werden.
*   `/Ergebnisse:` Enthält die finalen Ergebnisse der Analyse wie die umfassende CSV-Datei (`COMPREHENSIVE_ANALYSIS_DATA.csv`), Grafiken (z.B. `heatmap_concurrent_correlation.png`) und den generierten PDF-Bericht (`Statistische_Relevanzanalyse_Report.pdf`).

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
    Die Skripte erfordern eine Reihe von Python-Paketen. Installieren Sie diese mit:
    ```bash
    pip install pandas numpy matplotlib seaborn statsmodels scikit-learn spacy beautifulsoup4 textblob requests tqdm fpdf2 vaderSentiment
    ```
    Laden Sie zusätzlich das spaCy-Sprachmodell herunter:
    ```bash
    python -m spacy download en_core_web_sm
    ```

### Ausführung

Die Skripte sind so konzipiert, dass sie in einer bestimmten Reihenfolge ausgeführt werden, um die Datenpipeline abzubilden. Darüberhinaus ist der Input und Output Ordner innerhalb der Skripte festzulegen. Hierfür müssen *OUTPUT_DIR* und gegebenenfalls *Input-Pfade* angepasst werden.

1.  **Datenerhebung (mehrere Quellen):**
    *   Führen Sie `wayback_auto_daily_downloader.py` aus, um historische HTML-Snapshots der Reddit-Seiten zu sammeln.
    *   Führen Sie `activity_scraper.py` aus, um tägliche Metriken zu Abonnenten, Posts und Kommentaren zu sammeln.
    *   Führen Sie `gme_stock_data.py` aus, um die Aktienkursdaten aus der lokal gespeicherten HTML-Datei zu extrahieren.

2.  **Datenaufbereitung und -analyse:**
    *   Führen Sie `local_scraper_unified.py` aus. Dieses Skript ist der zentrale Verarbeitungsschritt: Es parst die HTML-Dateien, bereinigt die Daten (mithilfe von `data_cleaning.py`), führt die Sentiment-Analyse durch (mithilfe von `sentiment_analysis.py`) und erstellt die aggregierten CSV-Dateien `ALL_reddit_posts_extended.csv` und `ALL_reddit_sentiment_daily.csv`.

3.  **Statistische Auswertung:**
    *   Führen Sie `stat_relevance.py` aus. Dieses Skript führt alle aufbereiteten Daten zusammen, berechnet die finalen Kennzahlen und Lag-Variablen und führt die statistischen Tests durch.
    *   Das Ergebnis ist die Konnektivitätsdatei `COMPREHENSIVE_ANALYSIS_DATA.csv` und eine Heatmap der Korrelationen.

4.  **Export der Ergebnisse:**
    *   Führen Sie abschließend `stat_relevance_pdf_export.py` aus, um die wichtigsten Ergebnisse in einem PDF-Bericht zusammenzufassen.

## 🛠️ Code-Beschreibung

Die Skripte sind modular aufgebaut, um die verschiedenen Phasen des Projekts abzubilden.

### `wayback_auto_daily_downloader.py`
*Dieses Skript ist darauf ausgelegt, automatisiert tägliche Daten-Snapshots von einer Internet-Archivquelle wie der Wayback Machine herunterzuladen.*

**Details:** Das Skript verwendet die Wayback CDX API, um für einen definierten Zeitraum tägliche HTML-Snapshots der angegebenen Subreddits zu finden und herunterzuladen. Es beinhaltet eine robuste Fehlerbehandlung und eine Retry-Logik, um fehlgeschlagene Downloads zu wiederholen und zu protokollieren.

### `activity_scraper.py`
*Dieses Skript sammelt Metriken zur Nutzeraktivität wie die Anzahl der Abonnenten und täglichen Posts aus den relevanten Subreddits.*

**Details:** Anstatt Reddit direkt anzufragen, nutzt dieses Skript die Webseite `subredditstats.com`. Es kombiniert zwei Methoden: Für Abonnentenzahlen verwendet es eine API-Anfrage, während für tägliche Post- und Kommentarzahlen das HTML der Seite geparst und ein eingebettetes JSON-Objekt extrahiert wird. Die Daten werden aggregiert, verarbeitet (inkl. Forward-Fill-Logik) und in der Datei `subreddit_activity_stats.csv` gespeichert.

### `gme_stock_data.py`
*Dieses Skript ruft die historischen Aktienkurs- und Volumendaten für GameStop (GME) von einer Finanzdatenquelle ab.*

**Details:** Dieses Skript parst eine lokal gespeicherte HTML-Datei von Yahoo Finance. Es extrahiert die Datentabelle, bereinigt die numerischen Werte (z.B. Entfernung von Kommas) und speichert die relevanten Spalten (Datum, Preis, Volumen) in einer sauberen CSV-Datei. Es ist für eine einmalige Extraktion aus einer bestehenden Datei konzipiert.

### `local_scraper_unified.py`
*Dies ist eine konsolidierte Version, die mehrere lokale Datenverarbeitungsschritte zu einem einheitlichen Skript zusammenfasst.*

**Details:** Dies ist das Kernstück der Text- und Post-Verarbeitung. Das Skript lädt die von `wayback_auto_daily_downloader.py` gesammelten HTML-Dateien, parst sie mit BeautifulSoup und extrahiert einzelne Posts. Es führt eine Keyword-Filterung durch, extrahiert Metadaten wie Upvotes (mit komplexen Fallbacks für verschiedene HTML-Strukturen), führt eine Sentiment-Analyse durch (mithilfe von `sentiment_analysis.py`), identifiziert Emojis, berechnet das Verhältnis von Großbuchstaben, taggt Themen und extrahiert Entitäten mit spaCy. Das Skript führt eine globale Deduplizierung über alle Subreddits hinweg durch und speichert mehrere Ergebnisdateien: eine detaillierte Liste aller einzigartigen Posts (`ALL_reddit_posts_extended.csv`), tägliche aggregierte Metriken (`ALL_reddit_sentiment_daily.csv`) und eine Frequenzliste der am häufigsten verwendeten Wörter.

### `sentiment_analysis.py`
*Dieses Skript analysiert die gesammelten Reddit-Texte, um daraus tägliche Sentiment-Werte zu berechnen.*

**Details:** Dieses Hilfsskript initialisiert den VADER Sentiment-Analyzer und erweitert dessen Standard-Lexikon um ein benutzerdefiniertes Wörterbuch (`wsb_lexicon`). Dieses Lexikon enthält spezifische Begriffe und Emojis aus der Finanz- und Reddit-Kultur (z.B. "tendies", "diamond hands", "🚀") mit angepassten Sentiment-Werten, um eine präzisere Analyse im Kontext der GME-Diskussionen zu ermöglichen. Die Hauptfunktion `analyze_sentiment` wird von anderen Skripten importiert.

### `data_cleaning.py`
*Dieses Skript bereinigt die Rohdaten, indem es beispielsweise Duplikate entfernt, Formate vereinheitlicht und fehlende Werte behandelt.*

**Details:** Ein einfaches Hilfsskript, das eine `clean_text`-Funktion bereitstellt. Diese Funktion entfernt URLs und Sonderzeichen aus einem Text und konvertiert ihn in Kleinbuchstaben, um die Daten für die NLP-Verarbeitung vorzubereiten.

### `stat_relevance.py`
*Dieses Skript führt die statistische Hauptanalyse durch, indem es die Reddit- und Aktiendaten zusammenführt und Korrelationen sowie Kausalitätstests berechnet.*

**Details:** Dieses Skript ist der finale Schritt der quantitativen Analyse. Es lädt die aggregierten Tagesdaten aus den vorherigen Schritten (`ALL_reddit_sentiment_daily.csv`, `subreddit_activity_stats_wide.csv`, `gme_stock_data.csv`), führt sie zu einem umfassenden Zeitreihen-DataFrame zusammen und bereitet diesen auf: Es füllt fehlende Werte, berechnet finanzielle Kennzahlen wie logarithmische Renditen und erstellt verzögerte (Lagged) Variablen. Anschließend führt es die statistischen Tests durch: Augmented Dickey-Fuller (ADF) zur Prüfung der Stationarität, Pearson-Korrelationsanalysen (gleichzeitig und zeitversetzt) und Granger-Kausalitätstests. Die Ergebnisse werden auf der Konsole ausgegeben und die finale Datentabelle wird als `COMPREHENSIVE_ANALYSIS_DATA.csv` für weitere Analysen und Visualisierungen gespeichert.

### `stat_relevance_pdf_export.py`
*Dieses Skript exportiert die Ergebnisse und Visualisierungen der statistischen Analyse in ein formatiertes PDF-Dokument.*

**Details:** Dieses Skript nimmt die numerischen Ergebnisse und die generierte Heatmap-Grafik aus `stat_relevance.py` und verwendet die `fpdf2`-Bibliothek, um einen zusammenfassenden, mehrseitigen Bericht im PDF-Format zu erstellen. Es beinhaltet formatierte Tabellen für deskriptive Statistiken und Korrelationsmatrizen sowie die eingebettete Heatmap-Grafik, um eine übersichtliche Präsentation der Ergebnisse zu gewährleisten.

## 📈 Methodik

*   **Forschungsdesign:** Mixed-Methods-Ansatz, mit einem Schwerpunkt auf quantitativer Zeitreihenanalyse, ergänzt durch qualitative Experteninterviews.
*   **Datenerhebungsmethoden:** Primärdatenerhebung von Reddit mittels Scraping von `subredditstats.com` und der Wayback Machine; Sekundärdatenerhebung von Finanzdaten durch Parsen einer lokalen HTML-Datei von Yahoo Finance.
*   **Stichprobe / Auswahlverfahren:** Vollerhebung aller Posts/Kommentare in den Subreddits r/wallstreetbets, r/gme und r/Superstonk innerhalb des definierten Zeitraums (gefiltert nach Keywords) sowie der täglichen GME-Kursdaten.
*   **Operationalisierung der Variablen:** Die Variablen wurden in Aktienmarkt- (Preis, Volumen, Log-Return), Reddit-Sentiment-, Reddit-Aktivitäts- und Reddit-Engagement-Metriken unterteilt und auf täglicher Basis aggregiert. Lagged-Variablen wurden zur Analyse zeitversetzter Effekte erstellt.
*   **Statistische Verfahren / Auswertungsmethoden:**
    *   Deskriptive Statistik
    *   Zeitreihen-Visualisierung
    *   Stationaritätstests (Augmented Dickey-Fuller)
    *   Korrelationsanalyse (Pearson r) für gleichzeitige und gelaggte Variablen
    *   Granger-Kausalitätstests zur Untersuchung der prädiktiven Kraft der Reddit-Metriken
