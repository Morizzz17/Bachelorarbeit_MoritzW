# Bachelorarbeit_MoritzW
Diese Bachelorarbeit analysiert den Zusammenhang zwischen Reddit-Aktivitäten (Sentiment, Posts) und dem GameStop-Aktienkurs. Mittels Python wurden Daten gesammelt und aufbereitet. Eine statistische Auswertung durch Korrelations- und Granger-Kausalitätsanalysen untersucht den empirischen Einfluss von Social Media auf den Finanzmarkt.

## Skripte:

**activity_scraper.py:** Dieses Skript sammelt Metriken zur Nutzeraktivität wie die Anzahl der Abonnenten und täglichen Posts aus den relevanten Subreddits.

**data_cleaning.py:** Dieses Skript bereinigt die Rohdaten, indem es beispielsweise Duplikate entfernt, Formate vereinheitlicht und fehlende Werte behandelt.

**gme_stock_data.py:** Dieses Skript ruft die historischen Aktienkurs- und Volumendaten für GameStop (GME) von einer Finanzdatenquelle ab.

**local_scraper_unified.py:** Dies ist eine konsolidierte Version, die mehrere lokale Datenverarbeitungsschritte zu einem einheitlichen Skript zusammenfasst.

**sentiment_analysis.py:** Dieses Skript analysiert die gesammelten Reddit-Texte, um daraus tägliche Sentiment-Werte zu berechnen.

**stat_relevance.py:** Dieses Skript führt die statistische Hauptanalyse durch, indem es die Reddit- und Aktiendaten zusammenführt und Korrelationen sowie Kausalitätstests berechnet.

**stat_relevance_pdf_export.py:** Dieses Skript exportiert die Ergebnisse und Visualisierungen der statistischen Analyse in ein formatiertes PDF-Dokument.

**wayback_auto_daily_downloader.py:** Dieses Skript ist darauf ausgelegt, automatisiert tägliche Daten-Snapshots von einer Internet-Archivquelle wie der Wayback Machine herunterzuladen.
