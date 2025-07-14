# -*- coding: utf-8 -*-
"""
Unified Local HTML Scraper für mehrere Subreddits (Angepasst für neuere HTML-Struktur von Reddit seit 2023)
- Verarbeitet HTML-Snapshots von r/GME, r/Wallstreetbets, r/Superstonk in einem Durchlauf.
- Speichert individuelle Post-Listen pro Subreddit (intra-dedupliziert).
- Erstellt eine global deduplizierte Gesamtliste aller Posts.
- Berechnet aggregierte Tagesmetriken und Wortfrequenzen basierend auf den kombinierten, deduplizierten Daten.

@author: Moritz
"""
import spacy
import os
import pandas as pd
import re
import emoji
import csv
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from bs4 import BeautifulSoup
from textblob import TextBlob
import traceback
import sys

try:
    from sentiment_analysis import analyze_sentiment # Stelle sicher, dass diese Datei existiert und korrekt ist
except ImportError:
    print("FEHLER: sentiment_analysis.py nicht gefunden oder fehlerhaft.")
    # Fallback definieren, falls Sentiment-Analyse nicht verfügbar ist
    def analyze_sentiment(text): return {'compound': 0.0} # Gibt neutralen Score zurück
try:
    from data_cleaning import clean_text # Stelle sicher, dass diese Datei existiert und korrekt ist
except ImportError:
    print("FEHLER: data_cleaning.py nicht gefunden oder fehlerhaft.")
    # Fallback definieren
    def clean_text(text): return re.sub(r'\s+', ' ', text).lower() # Einfache Bereinigung

# === KONFIGURATION ===
# Mapping: Subreddit-Name -> Pfad zum HTML-Verzeichnis
SUBREDDIT_SOURCES = {
    "gme": r"C:\Users\Moriz\OneDrive - Finance Network - FNI® e.V\Bachelorarbeit\HTML-Snapshots\html_snapshots_gme",
    "wallstreetbets": r"C:\Users\Moriz\OneDrive - Finance Network - FNI® e.V\Bachelorarbeit\HTML-Snapshots\html_snapshots_wsb",
    "superstonk": r"C:\Users\Moriz\OneDrive - Finance Network - FNI® e.V\Bachelorarbeit\HTML-Snapshots\html_snapshots_superstonk"
}

KEYWORDS = ["gme", "amc", "moass", "short squeeze", "robinhood", "diamond hands", "hedge fund", "citadel", "melvin", "dfv"] # Ggf. anpassen
# Zielverzeichnis für ALLE Ausgabedateien
FINAL_RESULTS_DIR = r"C:\Users\Moriz\OneDrive - Hochschule München\Hochschule\Bachelorarbeit\GME_Coding\GME_Coding\results"
os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)

all_posts_combined = [] # Sammelt Posts von ALLEN Subreddits

# --- Kodierungsfix ---
def fix_encoding(text):
    # (Funktion bleibt unverändert)
    try:
        # Versuche häufige Fehlkodierungen zu korrigieren
        return text.encode('latin1').decode('utf-8')
    except UnicodeEncodeError:
        try:
            return text.encode('cp1252').decode('utf-8')
        except:
            return text # Im schlimmsten Fall Original zurückgeben
    except Exception:
        return text

# --- spaCy NLP-Model laden ---
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy Modell 'en_core_web_sm' geladen.")
except OSError:
    print("WARNUNG: spaCy Modell 'en_core_web_sm' nicht gefunden.")
    print("Bitte herunterladen: python -m spacy download en_core_web_sm")
    print("NER-Features werden nicht verfügbar sein.")
    nlp = None

# --- Themen-Tagging ---
TOPIC_PATTERNS = {
    "short squeeze": ["short squeeze", "short cover", "covering", "gamma squeeze", "squoze"],
    "robinhood": ["robinhood", "robin hood", "rh"],
    "regulation": ["sec", "regulation", "oversight", "investigation", "ban", "hearing", "congress"],
    "meme stocks": ["meme stock", "hype", "trending", "yolo"],
    "hedge funds": ["hedge fund", "citadel", "melvin", "ken griffin", "kenny g", "fund", "hf", "shf"],
    "diamond hands": ["diamond hands", "💎🤲", "💎🙌", "paper hands", "hold", "hodl"],
    "apes/community": ["ape", "apes together strong", "retard", "autist", "wsb", "community"],
    "dd/analysis": ["dd", "due diligence", "analysis", "research", "theory"],
    "drs": ["drs", "direct registration", "computershare", "cs"]
}

# === Hilfsfunktionen ===
def parse_numeric_string(text):
    if not text: return None
    text_lower = str(text).strip().lower()
    if text_lower == 'vote': return 0 # 'Vote' als 0 Upvotes interpretieren
    if text_lower == '•': return None # Spezielles Zeichen ignorieren
    # Entferne alles außer Zahlen, Punkt, Komma, k, m
    cleaned_text = re.sub(r'[^\d.,km]', '', text_lower)
    if not cleaned_text: return None
    # Ersetze Komma durch Punkt für Dezimaltrennung (falls verwendet)
    cleaned_text = cleaned_text.replace(',', '.')
    # Überprüfen, ob es mit einer Ziffer beginnt (verhindert 'k', 'm' alleine)
    if not cleaned_text or not cleaned_text[0].isdigit(): return None

    multiplier = 1
    if cleaned_text.endswith('k'):
        multiplier = 1000
        cleaned_text = cleaned_text[:-1].strip()
    elif cleaned_text.endswith('m'):
        multiplier = 1_000_000
        cleaned_text = cleaned_text[:-1].strip()

    try:
        # Prüfe, ob nach Entfernen von k/m noch etwas Sinnvolles übrig ist
        if not cleaned_text: return None
        number = float(cleaned_text)
        return int(number * multiplier)
    except ValueError:
        #print(f"Debug: ValueError beim Parsen von '{text}' -> '{cleaned_text}'") # Debugging
        return None

# (find_upvotes_near bleibt unverändert - dient als Fallback)
def find_upvotes_near(tag):
    # Diese Funktion wird als Fallback verwendet, wenn der 'score'-Attribut fehlt
    if not tag: # Funktion braucht ein Start-Tag
        return None

    # Versuche den Container zu finden, der das Tag umschließt
    # Da 'tag' jetzt auch ein <a> sein kann, müssen wir flexibler suchen
    post_container = tag.find_parent(['shreddit-post', 'div', 'article']) # Suche nach verschiedenen Eltern
    if not post_container:
         post_container = tag.find_parent(class_=re.compile(r"\bPost\b|scrollerItem\b"))

    if not post_container: return None

    # --- Logik aus deiner alten Funktion, angepasst ---
    # 1. Suche nach spezifischen Klassen für Score-Divs (Beispiele, können variieren)
    score_divs = post_container.find_all('div', class_=re.compile(r'_1rZYMD_4xY3gRcSS3p8ODO|faceplate-number')) # faceplate-number ist neu
    for score_div in score_divs:
        score_text = score_div.get_text(strip=True)
        parsed_score = parse_numeric_string(score_text)
        if parsed_score is not None: return parsed_score # Nimm den ersten Treffer

    # 2. Suche nach Screenreader-Spans (Beispielklassen)
    score_span_screenreader = post_container.find('span', class_=re.compile(r'D6SuXeSnAAagG8dKAb4O4|_2IpBiNIh3UquYptXX4iDbJ'))
    if score_span_screenreader:
        parsed_score = parse_numeric_string(score_span_screenreader.get_text(strip=True))
        if parsed_score is not None: return parsed_score

    # 3. Suche im Voting-Bereich (Beispielklassen)
    vote_area_div = post_container.find('div', class_=re.compile(r'_23h0-EcaBUorIHC-JZyh6J|_3a2ZHWaih05DgAOtvu6cIo|top-level')) # top-level für neuere Struktur
    if vote_area_div:
        # Finde speziell das Element mit faceplate-number innerhalb des Vote-Bereichs
        fp_number = vote_area_div.find('faceplate-number')
        if fp_number and fp_number.has_attr('number'):
            parsed_score = parse_numeric_string(fp_number['number'])
            if parsed_score is not None: return parsed_score

        # Fallback: Allgemeine Suche nach Divs im Vote-Bereich
        possible_score_divs = vote_area_div.find_all(['div', 'span'], limit=5)
        for div in possible_score_divs:
             div_text = div.get_text(strip=True)
             if re.match(r'^[\d.,]+[km]?$|^vote$', div_text.lower()):
                 parsed_score = parse_numeric_string(div_text)
                 if parsed_score is not None: return parsed_score

    # Wenn immer noch nichts gefunden wurde
    return None


# (extract_emojis bleibt unverändert)
def extract_emojis(text):
    cleaned_text_for_emojis = fix_encoding(text)
    return "".join(c for c in cleaned_text_for_emojis if c in emoji.EMOJI_DATA)

# (amplify_sentiment_by_caps bleibt unverändert)
def amplify_sentiment_by_caps(score, capslock_ratio, threshold=0.4, factor=1.1):
    if score is None: return None # Sicherstellen, dass Score nicht None ist
    if capslock_ratio > threshold:
        amplified = score * factor
        return max(-1.0, min(1.0, amplified)) # Stelle sicher, dass Score im Bereich [-1, 1] bleibt
    return score

# === HAUPTVERARBEITUNG ===
print("=== Starte unified Scraper für alle Subreddits ===")

for subreddit_name, html_dir in SUBREDDIT_SOURCES.items():
    print(f"\n--- Verarbeite Subreddit: {subreddit_name} ---")
    print(f"HTML-Quelle: {html_dir}")
    if not os.path.isdir(html_dir):
        print(f"WARNUNG: Verzeichnis nicht gefunden: {html_dir}. Überspringe.")
        continue

    files_processed = 0
    posts_found_in_subreddit = 0
    # Iteriere durch die Dateien in chronologischer Reihenfolge (wichtig für spätere Deduplizierung)
    try:
        filenames = sorted(os.listdir(html_dir))
    except FileNotFoundError:
        print(f"FEHLER: Kann Verzeichnis nicht lesen: {html_dir}")
        continue

    for filename in filenames:
        if not filename.endswith(".html"):
            continue

        date_str = filename.replace(".html", "")
        file_path = os.path.join(html_dir, filename)
        print(f"\n---> VERSUCHE DATEI: {file_path} <---") # Logging

        try:
            with open(file_path, "r", encoding="utf-8", errors='replace') as f:
                soup = BeautifulSoup(f, "html.parser")
            print(f"     Datei geparsed: {filename}") # Logging

            files_processed += 1
            # --- Finde Posts ---
            post_containers = soup.find_all('shreddit-post') # Neuer primärer Selektor
            if not post_containers:
                # Fallbacks für ältere Strukturen
                print("     -> Kein <shreddit-post> gefunden, versuche alte Selektoren...") # Logging
                post_containers = soup.find_all('div', class_=re.compile(r"\bPost\b|scrollerItem\b|_1poyrkZ7g36PawDueRza-J\b"))
            if not post_containers:
                 print("     -> Auch keine alten Div-Container gefunden, versuche <article>...") # Logging
                 post_containers = soup.find_all('article', attrs={'data-testid': 'post-container'})
            print(f"     {len(post_containers)} potenzielle Post-Container gefunden.") # Logging

            posts_in_file = 0
            processed_titles_in_file = set()

            for i, post_container in enumerate(post_containers): # Hinzufügen von Index für Logging
                print(f"       Prüfe Container {i+1}/{len(post_containers)}...") # Logging
                raw_title = None
                tag = None

                # *** Titel Extratktion***
                if post_container.name == 'shreddit-post' and post_container.has_attr('post-title'):
                    raw_title = post_container.get('post-title')
                    # Versuche trotzdem, ein Tag für find_upvotes_near zu finden (das <a> ist gut)
                    title_link = post_container.find('a', slot='title')
                    if title_link:
                        tag = title_link
                else:
                    # Fallback 1: Suche nach <a slot="title"> in neueren Strukturen
                    title_link = post_container.find('a', slot='title')
                    if title_link:
                        raw_title = fix_encoding(title_link.get_text().strip())
                        tag = title_link # Wichtig für find_upvotes_near fallback
                    else:
                        # Fallback 2: Suche nach H3/H2/H1 (alte Methode)
                        tag = post_container.find(["h3", "h2", "h1"])
                        if tag:
                            raw_title = fix_encoding(tag.get_text().strip())
                        else:
                            # Fallback 3: Nimm aria-label vom article (falls es ein article ist oder ein parent)
                            container_tag_name = post_container.name
                            parent_article = post_container if container_tag_name == 'article' else post_container.find_parent('article')
                            if parent_article and parent_article.has_attr('aria-label'):
                                raw_title = fix_encoding(parent_article['aria-label'].strip())
                                # 'tag' bleibt hier None, was für upvote-fallback ok ist
                                print("         -> Titel aus aria-label extrahiert.")
                            else:
                                print("         -> Kein H1/H2/H3/a[slot=title]/aria-label Titel gefunden.") # Logging
                                continue # Zum nächsten Container springen

                if not raw_title:
                    print("         -> Titel ist leer nach Extraktion.")
                    continue

                print(f"         Titel gefunden: '{raw_title[:80]}...'") # Logging

                # --- Keyword Check ---
                if not any(keyword in raw_title.lower() for keyword in KEYWORDS):
                    # print(f"         -> Titel enthält kein Keyword: '{raw_title[:80]}...'") # Optional: Nur für Debugging
                    continue
                print(f"         >>> KEYWORD MATCH: '{raw_title[:80]}...'") # Logging

                # --- Duplikat-Check innerhalb der Datei ---
                if raw_title in processed_titles_in_file:
                    print("         -> Titel bereits in dieser Datei verarbeitet (Skip).") # Logging
                    continue
                processed_titles_in_file.add(raw_title)

                posts_in_file += 1
                try:
                    upvotes = None
                    if post_container.name == 'shreddit-post' and post_container.has_attr('score'):
                        score_attribute = post_container.get('score')
                        upvotes = parse_numeric_string(score_attribute)
                        print(f"           Upvotes aus Attribut 'score': {upvotes}") # Logging

                    # Fallback: Wenn 'score'-Attribut fehlt/nicht parsebar ODER wir eh schon ein 'tag' haben
                    if upvotes is None and tag:
                        print("           Versuche Upvotes mit find_upvotes_near (Fallback)...") # Logging
                        upvotes = find_upvotes_near(tag) # braucht 'tag' als Startpunkt
                        print(f"           Upvotes aus Fallback-Suche: {upvotes}") # Logging
                    elif upvotes is None:
                         print("           Kein 'score'-Attribut und kein Fallback-Titel-Tag für Upvote-Suche gefunden.") # Logging


                    # --- Rest der Analyse (Sentiment, etc.) ---
                    cleaned = clean_text(raw_title)
                    vader_scores = analyze_sentiment(raw_title) # Nutzt Fallback, wenn Modul fehlt
                    vader_compound_score = vader_scores.get('compound', 0.0) # Sicherer Zugriff

                    try:
                        subjectivity = TextBlob(cleaned).sentiment.subjectivity
                    except Exception as tb_err:
                         subjectivity = None

                    emojis_found = extract_emojis(raw_title)
                    emoji_count = len(emojis_found)
                    caps_ratio = (sum(1 for c in raw_title if c.isupper()) / len(raw_title)) if raw_title else 0
                    sentiment_final_adjusted = amplify_sentiment_by_caps(vader_compound_score, caps_ratio)

                    label = "unknown"
                    if subjectivity is not None:
                        label = "opinionated" if (subjectivity >= 0.5 or vader_compound_score > 0.4 or "!" in raw_title or any(word in raw_title.lower() for word in ["thank", "love", "hate", "feel"])) else "factual"

                    entities = []
                    if nlp:
                        try:
                            doc = nlp(cleaned)
                            entities = [ent.text for ent in doc.ents if ent.label_ in {"ORG", "PERSON", "GPE", "PRODUCT", "MONEY"}]
                        except Exception as spacy_err:
                            pass
                    user_mentions = re.findall(r"u\/[A-Za-z0-9_-]+", raw_title)
                    entities.extend(user_mentions)

                    topics = []
                    for tag_label, keywords in TOPIC_PATTERNS.items():
                        if any(kw in cleaned.lower() for kw in keywords) or any(kw in raw_title.lower() for kw in keywords):
                            topics.append(tag_label)

                    image_urls = set()
                    # Suche nach Bildern (Klassen können sich ändern, flexibel bleiben)
                    img_tags = post_container.find_all('img', class_=re.compile(r'ImageBox-image|_2_tDEnGMLxpM6uOa2kaDB3|media-element|preview-img')) # 'preview-img' hinzugefügt
                    for img in img_tags:
                        src = img.get('src') or img.get('data-src')
                        # Verbesserter Filter: Ignoriere sehr kleine/generische Icons
                        if src and not src.startswith('data:image') and ('redd.it' in src or 'redditmedia.com' in src) and len(src) > 40: # Längere URL und Domain-Check
                            image_urls.add(src)
                    video_tags = post_container.find_all('video')
                    for video in video_tags:
                        poster = video.get('poster')
                        if poster and not poster.startswith('data:image') and len(poster) > 40:
                            image_urls.add(poster)

                    # --- Daten sammeln ---
                    all_posts_combined.append({
                        "subreddit": subreddit_name,
                        "date": date_str,
                        "title": raw_title,
                        "cleaned": cleaned,
                        "sentiment_vader_comp": vader_compound_score,
                        "sentiment_final": sentiment_final_adjusted,
                        "subjectivity": subjectivity,
                        "subjectivity_label": label,
                        "emojis": emojis_found,
                        "emoji_count": emoji_count,
                        "entities": ", ".join(list(set(entities))),
                        "topic_tags": ", ".join(list(set(topics))),
                        "upvotes": upvotes,
                        "image_urls": "; ".join(list(image_urls)),
                        "title_length": len(cleaned),
                        "capslock_ratio": caps_ratio,
                    })
                    posts_found_in_subreddit += 1
                    print(f"           +++++ Post erfolgreich verarbeitet und hinzugefügt! +++++") # Logging

                except Exception as analyse_err:
                    # Detailliertere Fehlerausgabe
                    print(f"           XXXXX ERROR bei Analyse von '{raw_title[:50]}...' in {filename} XXXXX") # Logging
                    print(f"           Fehlertyp: {type(analyse_err).__name__}")
                    print(f"           Fehlermeldung: {analyse_err}")
                    # traceback.print_exc() # Optional: Kompletter Traceback
                    print("           ----------------------------------------------------") # Logging


            print(f"     -> {posts_in_file} Posts in {filename} verarbeitet.") # Logging

        except FileNotFoundError:
            print(f"!!! Datei nicht gefunden: {file_path}")
        except Exception as parse_err:
             # Detailliertere Fehlerausgabe
             print(f"!!! FATAL ERROR beim Parsen von {filename} !!!") # Logging
             print(f"    Fehlertyp: {type(parse_err).__name__}")
             print(f"    Fehlermeldung: {parse_err}")
             # traceback.print_exc() # Optional: Kompletter Traceback
             print("    -----------------------------------------") # Logging


    print(f"--- {subreddit_name}: {files_processed} Dateien verarbeitet, {posts_found_in_subreddit} Posts initial extrahiert. ---")

print(f"\n=== Alle Subreddits gescraped. Gesamtanzahl Posts initial: {len(all_posts_combined)} ===")

# === NACHVERARBEITUNG UND SPEICHERN ===
if all_posts_combined:
    df_all = pd.DataFrame(all_posts_combined)

    # --- Datenbereinigung und Konvertierung (Gesamt-DataFrame) ---
    print("\n--- Datenbereinigung und Typkonvertierung ---")
    try:
        df_all["date"] = pd.to_datetime(df_all["date"], errors='coerce')
        # Zeilen löschen, bei denen die Datumsumwandlung fehlgeschlagen ist, BEVOR die Länge geprüft wurde
        df_all.dropna(subset=['date'], inplace=True)
        if df_all.empty:
            print("FEHLER: Keine gültigen Datumsangaben nach Konvertierung gefunden. Breche ab.")
            sys.exit(1)
        print(f"Gültige Datumsangaben: {len(df_all)}")
    except Exception as date_err:
        print(f"FEHLER bei globaler Datumskonvertierung: {date_err}. Breche ab.")
        sys.exit(1) # Beendet das Skript mit Fehlercode

    numeric_cols = ['upvotes', 'sentiment_vader_comp', 'sentiment_final', 'subjectivity', 'title_length', 'capslock_ratio', 'emoji_count']
    for col in numeric_cols:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
        else:
             print(f"Warnung: Erwartete numerische Spalte '{col}' nicht im DataFrame gefunden.")

    # --- 1. Speichern der individuellen Subreddit-Dateien (Intra-dedupliziert) ---
    print("\n--- Speichere individuelle Subreddit-Dateien (intra-dedupliziert) ---")
    output_columns_individual = [ # Definiere hier die gewünschte Reihenfolge/Auswahl
            "subreddit", "date", "title", "cleaned", "sentiment_vader_comp", "sentiment_final",
            "subjectivity", "subjectivity_label", "emojis", "emoji_count",
            "entities", "topic_tags", "upvotes", "image_urls", "title_length", "capslock_ratio"
        ]

    for sub_name in SUBREDDIT_SOURCES.keys():
        print(f"Verarbeite und speichere für: {sub_name}")
        df_sub = df_all[df_all['subreddit'] == sub_name].copy()

        if df_sub.empty:
            print(f" -> Keine Daten für {sub_name} nach Filterung.")
            continue

        # Intra-Subreddit Deduplizierung
        initial_count = len(df_sub)
        df_sub.sort_values(by=['title', 'date'], ascending=[True, False], inplace=True, na_position='last')
        df_sub.drop_duplicates(subset=['title'], keep='first', inplace=True)
        df_sub.sort_values(by='date', ascending=True, inplace=True, na_position='last')
        print(f" -> Intra-Deduplizierung: {initial_count} -> {len(df_sub)} Posts")

        # Spalten auswählen, die existieren
        final_columns_sub = [col for col in output_columns_individual if col in df_sub.columns]
        df_sub_output = df_sub[final_columns_sub]

        csv_path_sub = os.path.join(FINAL_RESULTS_DIR, f"reddit_posts_{sub_name}.csv")
        try:
            df_sub_output.to_csv(
                csv_path_sub,
                index=False,
                sep=";",
                encoding="utf-8-sig",
                quoting=csv.QUOTE_ALL
            )
            print(f" -> Gespeichert: {csv_path_sub}")
        except Exception as csv_err:
            print(f"!!! FEHLER beim Speichern der CSV für {sub_name}: {csv_err}")

    # --- 2. Globale Deduplizierung (Gesamt-DataFrame) ---
    print("\n--- Globale Deduplizierung über alle Subreddits ---")
    df_deduplicated = df_all.copy() # Kopieren, um Original nicht zu ändern
    initial_total_count = len(df_deduplicated)
    df_deduplicated.sort_values(by=['title', 'date'], ascending=[True, False], inplace=True, na_position='last')
    df_deduplicated.drop_duplicates(subset=['title'], keep='first', inplace=True)
    df_deduplicated.sort_values(by='date', ascending=True, inplace=True, na_position='last')
    print(f"Globale Deduplizierung: {initial_total_count} -> {len(df_deduplicated)} einzigartige Posts.")

    # --- 3. Speichern der global deduplizierten Gesamt-Postliste ---
    print("\n--- Speichere global deduplizierte Gesamt-Postliste ---")
    csv_path_all_posts = os.path.join(FINAL_RESULTS_DIR, "ALL_reddit_posts_extended.csv")
    try:
        # Verwende dieselbe Spaltenauswahl wie für individuelle Dateien
        final_columns_all = [col for col in output_columns_individual if col in df_deduplicated.columns]
        df_deduplicated_output = df_deduplicated[final_columns_all]
        df_deduplicated_output.to_csv(
            csv_path_all_posts,
            index=False,
            sep=";",
            encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL
        )
        print(f"Gespeichert: {csv_path_all_posts}")
    except Exception as csv_err:
        print(f"!!! FEHLER beim Speichern der globalen Post-CSV: {csv_err}")

    # --- 4. Berechne und speichere aggregierte Tagesmetriken (basierend auf df_deduplicated) ---
    print("\n--- Berechne und speichere aggregierte Tagesmetriken ---")
    daily_agg = {}
    if 'sentiment_final' in df_deduplicated.columns and pd.api.types.is_numeric_dtype(df_deduplicated['sentiment_final']):
        daily_agg['Avg_Sentiment_Final'] = ('sentiment_final', 'mean')
    if 'sentiment_vader_comp' in df_deduplicated.columns and pd.api.types.is_numeric_dtype(df_deduplicated['sentiment_vader_comp']):
        daily_agg['Avg_Sentiment_VADER'] = ('sentiment_vader_comp', 'mean')
    if 'subjectivity' in df_deduplicated.columns and pd.api.types.is_numeric_dtype(df_deduplicated['subjectivity']):
        daily_agg['Avg_Subjectivity'] = ('subjectivity', 'mean')
    if 'upvotes' in df_deduplicated.columns and pd.api.types.is_numeric_dtype(df_deduplicated['upvotes']):
        upvotes_series = df_deduplicated['upvotes'].dropna()
        if not upvotes_series.empty:
             daily_agg['Median_Upvotes'] = ('upvotes', 'median')
             daily_agg['Avg_Upvotes'] = ('upvotes', 'mean')
             daily_agg['Sum_Upvotes'] = ('upvotes', 'sum')
    daily_agg['Post_Count'] = ('title', 'count')

    if daily_agg:
        try:
            daily_data = df_deduplicated.groupby(df_deduplicated['date'].dt.date).agg(**daily_agg)
            daily_data = daily_data.reset_index()
            daily_data.rename(columns={'date': 'Date'}, inplace=True)

            cols_to_round = ['Avg_Sentiment_Final', 'Avg_Sentiment_VADER', 'Avg_Subjectivity', 'Avg_Upvotes']
            for col in cols_to_round:
                if col in daily_data.columns:
                    daily_data[col] = daily_data[col].round(4)

            csv_path_daily = os.path.join(FINAL_RESULTS_DIR, "ALL_reddit_sentiment_daily.csv")
            daily_data.to_csv(
                csv_path_daily,
                index=False,
                sep=";",
                encoding="utf-8-sig",
                quoting=csv.QUOTE_MINIMAL
            )
            print(f"Gespeichert: {csv_path_daily}")
        except Exception as agg_err:
             print(f"!!! FEHLER bei der Tagesaggregation: {agg_err}")
    else:
        print("WARNUNG: Keine numerischen Spalten für die Tagesaggregation gefunden.")

    # --- 5. Berechne und speichere Wortfrequenzen (basierend auf df_deduplicated) ---
    print("\n--- Berechne und speichere Wortfrequenzen ---")
    if 'cleaned' in df_deduplicated.columns:
        try:
            cleaned_texts = df_deduplicated["cleaned"].dropna().astype(str)
            if not cleaned_texts.empty:
                all_words_text = " ".join(cleaned_texts)
                words = re.findall(r"\b[a-z]{3,}\b", all_words_text.lower())
                custom_stopwords = set(ENGLISH_STOP_WORDS).union({'gme', 'amc', 'like', 'just', 'im', 'dont', 'know', 'post', 'thread', 'https', 'www', 'com', 'org'})
                filtered = [w for w in words if w not in custom_stopwords and not w.isdigit()]
                word_freq = Counter(filtered).most_common(150)
                if word_freq:
                    freq_df = pd.DataFrame(word_freq, columns=["word", "count"])
                    csv_path_freq = os.path.join(FINAL_RESULTS_DIR, "ALL_word_frequencies.csv")
                    freq_df.to_csv(
                        csv_path_freq,
                        index=False,
                        sep=";",
                        encoding="utf-8-sig",
                        quoting=csv.QUOTE_MINIMAL
                    )
                    print(f"Gespeichert: {csv_path_freq}")
                else:
                    print("WARNUNG: Keine Wörter für Frequenzanalyse nach Filterung übrig.")
            else:
                print("WARNUNG: Keine Texte für die Wortfrequenzanalyse vorhanden.")
        except Exception as freq_err:
            print(f"!!! FEHLER bei der Wortfrequenzanalyse: {freq_err}")

    print("\n✅ Unified Scraper erfolgreich abgeschlossen.")

else:
    print("⚠️ Keine passenden Beiträge in den verarbeiteten HTML-Dateien gefunden.")