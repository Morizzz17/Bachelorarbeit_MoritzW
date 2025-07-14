# -*- coding: utf-8 -*-
"""
Modularer Local HTML Scraper – Grundgerüst
Zieht H3-Titel aus HTML + berechnet Sentiment und Subjectivity,
Members, passt Sentiment mit Capslock an. WSB/Emoji-Lexikon in sentiment_analysis.py

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
from sentiment_analysis import analyze_sentiment
from data_cleaning import clean_text

# === SETTINGS ===
KEYWORDS = ["gme", "amc", "moass", "short squeeze", "robinhood", "diamond hands", "hedge fund"]
HTML_DIR = r"C:\Users\Moriz\OneDrive - Finance Network - FNI® e.V\Bachelorarbeit\HTML-Snapshots\html_snapshots_bis11-10"
RESULTS_DIR = r"C:\Users\Moriz\OneDrive - Hochschule München\Hochschule\Bachelorarbeit\GME_Coding\GME_Coding\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

all_posts = []

# EMOJI_SENTIMENT Dictionary hier nicht mehr nötig

def fix_encoding(text):
    try:
        return text.encode('latin1').decode('utf-8')
    except:
        return text

# === spaCy NLP-Model laden ===
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy Modell 'en_core_web_sm' nicht gefunden.")
    print("Bitte herunterladen: python -m spacy download en_core_web_sm")
    nlp = None

# === Themen-Tagging ===
TOPIC_PATTERNS = {
    "short squeeze": ["short squeeze", "shortcover", "covering", "gamma squeeze"],
    "robinhood": ["robinhood", "robin hood", "rh"],
    "regulation": ["sec", "regulation", "oversight", "investigation", "ban", "hearing"],
    "meme stocks": ["meme stock", "hype", "trending"],
    "hedge funds": ["hedge fund", "citadel", "melvin", "ken griffin", "fund", "hf"]
}

# === Hilfsfunktionen ===
def parse_numeric_string(text):
    # Deine funktionierende Version
    if not text: return None
    text_lower = str(text).strip().lower()
    if text_lower == 'vote': return 0
    if text_lower == '•': return None
    cleaned_text = re.sub(r'[^\d.km]', '', text_lower)
    if not cleaned_text: return None
    if not cleaned_text[0].isdigit(): return None
    cleaned_text = cleaned_text.replace(",", "")
    multiplier = 1
    if cleaned_text.endswith('k'):
        multiplier = 1000
        cleaned_text = cleaned_text[:-1]
    elif cleaned_text.endswith('m'):
        multiplier = 1_000_000
        cleaned_text = cleaned_text[:-1]
    try:
        number = float(cleaned_text)
        return int(number * multiplier)
    except ValueError: return None

def find_upvotes_near(tag):
    # Deine funktionierende Version
    post_container = tag.find_parent(class_=re.compile(r"\bPost\b"))
    if not post_container: return None
    vote_area_div = post_container.find('div', class_='_23h0-EcaBUorIHC-JZyh6J')
    if not vote_area_div: return None
    vote_container = vote_area_div.find('div', id=lambda x: x and x.startswith('vote-arrows-'))
    if not vote_container: vote_container = vote_area_div.find('div', class_='_1E9mcoVn4MYnuBQSVDt1gC')
    if not vote_container: return None
    score_span_screenreader = vote_area_div.find('span', class_='D6SuXeSnAAagG8dKAb4O4')
    if score_span_screenreader:
        score_text = score_span_screenreader.get_text(strip=True)
        parsed_score = parse_numeric_string(score_text)
        if parsed_score is not None: return parsed_score
    score_div_direct = vote_container.find('div', class_='_1rZYMD_4xY3gRcSS3p8ODO')
    if score_div_direct:
        score_text = score_div_direct.get_text(strip=True)
        parsed_score = parse_numeric_string(score_text)
        if parsed_score is not None: return parsed_score
    return None

def extract_emojis(text):
    """ Extrahiert Emojis aus einem Text mithilfe des emoji.EMOJI_DATA Sets. """
    cleaned_text_for_emojis = fix_encoding(text)
    return "".join(c for c in cleaned_text_for_emojis if c in emoji.EMOJI_DATA)

# Funktion adjust_sentiment_with_emojis wird nicht mehr benötigt

def amplify_sentiment_by_caps(score, capslock_ratio, threshold=0.4, factor=1.1):
    """ Erhöht den Betrag des Scores leicht, wenn viele Großbuchstaben vorkommen. """
    if capslock_ratio > threshold:
        amplified = score * factor
        return max(-1.0, min(1.0, amplified))
    return score

# === SCRAPE LOOP ===
for filename in sorted(os.listdir(HTML_DIR)):
    if not filename.endswith(".html"):
        continue

    date_str = filename.replace(".html", "")
    file_path = os.path.join(HTML_DIR, filename)
    print(f"\n--- Verarbeite: {filename} ---")
    try:
        with open(file_path, "r", encoding="utf-8", errors='replace') as f:
            soup = BeautifulSoup(f, "html.parser")

        h3_tags = soup.find_all("h3")
        # print(f"{date_str}: {len(h3_tags)} H3-Tags gefunden") # Debug

        posts_in_file = 0
        for tag in h3_tags:
            post_container = tag.find_parent(class_=re.compile(r"\bPost\b"))
            if not post_container: continue

            raw_title = fix_encoding(tag.get_text().strip())
            if not raw_title or not any(keyword in raw_title.lower() for keyword in KEYWORDS):
                continue

            posts_in_file += 1
            try:
                cleaned = clean_text(raw_title)
                # Verwende den angepassten VADER auf den ROH-TEXT
                vader_scores = analyze_sentiment(raw_title)
                vader_compound_score = vader_scores['compound'] # Der Score enthält jetzt WSB/Emoji-Infos

                subjectivity = TextBlob(cleaned).sentiment.subjectivity

                emojis_found = extract_emojis(raw_title)
                emoji_count = len(emojis_found)

                caps_ratio = (sum(1 for c in raw_title if c.isupper()) / len(raw_title)) if len(raw_title) > 0 else 0

                # Verstärke den (bereits durch Lexikon/Emojis angepassten) VADER-Score mit Capslock
                sentiment_final_adjusted = amplify_sentiment_by_caps(vader_compound_score, caps_ratio)

                # Subjektivitäts-Label
                label = "opinionated" if (subjectivity >= 0.5 or vader_compound_score > 0.4 or "!" in raw_title or "thank" in raw_title.lower()) else "factual"

                # Entities
                entities = []
                if nlp:
                    doc = nlp(cleaned)
                    entities = [ent.text for ent in doc.ents if ent.label_ in {"ORG", "PERSON", "GPE", "PRODUCT"}]
                user_mentions = re.findall(r"u\/[A-Za-z0-9_-]+", raw_title)
                entities.extend(user_mentions) # extend statt += für Klarheit

                # Topics
                topics = []
                for tag_label, keywords in TOPIC_PATTERNS.items():
                    if any(kw in cleaned.lower() for kw in keywords):
                        topics.append(tag_label)

                # Upvotes
                upvotes = find_upvotes_near(tag)

                # Bild-URLs
                image_urls = []
                img_tags = post_container.find_all('img', class_='ImageBox-image')
                if not img_tags: img_tags = post_container.find_all('img', class_='_2_tDEnGMLxpM6uOa2kaDB3')
                for img in img_tags:
                    src = img.get('src')
                    if src and not src.startswith('data:image') and len(src) > 10: image_urls.append(src)
                video_tags = post_container.find_all('video')
                for video in video_tags:
                    poster = video.get('poster')
                    if poster and not poster.startswith('data:image') and len(poster) > 10 and poster not in image_urls: image_urls.append(poster)

                # Daten sammeln (Spaltennamen angepasst)
                all_posts.append({
                    "date": date_str,
                    "title": raw_title,
                    "cleaned": cleaned,
                    "sentiment_vader_comp": vader_compound_score, # Kombinierter VADER (WSB+Emoji)
                    "sentiment_final": sentiment_final_adjusted, # + Capslock-Verstärkung
                    "subjectivity": subjectivity,
                    "subjectivity_label": label,
                    "emojis": emojis_found,
                    "emoji_count": emoji_count,
                    "entities": ", ".join(list(set(entities))),
                    "topic_tags": ", ".join(list(set(topics))),
                    "upvotes": upvotes,
                    "image_urls": "; ".join(image_urls),
                    "title_length": len(cleaned),
                    "capslock_ratio": caps_ratio,
                })
            except Exception as se:
                print(f"  ERROR bei Analyse von '{raw_title[:50]}...' – {se}")

        print(f"{date_str}: {posts_in_file} passende Posts verarbeitet.")

    except FileNotFoundError:
        print(f"!!! Datei nicht gefunden: {file_path}")
    except Exception as e:
        print(f"!!! FATAL ERROR beim Parsen von {filename} – {e}")

# === SPEICHERN ===
if all_posts:
    df = pd.DataFrame(all_posts)

    if 'date' in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors='coerce')
            # df.dropna(subset=['date'], inplace=True) # Entfernen optional
        except Exception as date_err:
            print(f"WARNUNG: Fehler bei Datumskonvertierung - {date_err}")
            df = pd.DataFrame()

    # Konvertiere numerische Spalten
    for col in ['upvotes', 'sentiment_vader_comp', 'sentiment_final', 'subjectivity', 'title_length', 'capslock_ratio', 'emoji_count']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Deduplizierung
    if 'title' in df.columns and 'date' in df.columns and df['date'].notna().any():
        print(f"\nDataFrame Größe vor Deduplizierung: {len(df)}")
        df.sort_values(by=['title', 'date'], ascending=[True, False], inplace=True, na_position='last')
        df.drop_duplicates(subset=['title'], keep='first', inplace=True)
        df.sort_values(by='date', ascending=True, inplace=True, na_position='last')
        print(f"DataFrame Größe nach Deduplizierung: {len(df)}")
    else:
         print("\nWARNUNG: 'title' oder gültige 'date' Spalte für Deduplizierung nicht ausreichend vorhanden.")

    # Definiere Spaltenreihenfolge
    output_columns_extended = [
        "date", "title", "cleaned", "sentiment_vader_comp", "sentiment_final", # Angepasste Namen
        "subjectivity", "subjectivity_label", "emojis", "emoji_count",
        "entities", "topic_tags", "upvotes", "image_urls", "title_length",
        "capslock_ratio"
    ]
    final_columns = [col for col in output_columns_extended if col in df.columns]
    df_output = df[final_columns]

    # === Haupt-CSV: Extended Posts ===
    csv_path_extended = os.path.join(RESULTS_DIR, "reddit_posts_extended.csv")
    print(f"\nSpeichere Haupt-CSV: {csv_path_extended}")
    try:
        df_output.to_csv(
            csv_path_extended,
            index=False,
            sep=";",
            encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL
        )
    except Exception as csv_err:
        print(f"!!! FEHLER beim Speichern der Haupt-CSV: {csv_err}")

    # === Tagesdurchschnitt ===
    if 'date' in df.columns and df['date'].notna().any():
        daily_agg = {}
        if 'sentiment_final' in df.columns and pd.api.types.is_numeric_dtype(df['sentiment_final']):
            daily_agg['Avg_Sentiment_Final'] = ('sentiment_final', 'mean')
        if 'sentiment_vader_comp' in df.columns and pd.api.types.is_numeric_dtype(df['sentiment_vader_comp']):
             daily_agg['Avg_Sentiment_VADER'] = ('sentiment_vader_comp', 'mean') # Umbenannt
        if 'subjectivity' in df.columns and pd.api.types.is_numeric_dtype(df['subjectivity']):
             daily_agg['Avg_Subjectivity'] = ('subjectivity', 'mean')
        if 'upvotes' in df.columns and pd.api.types.is_numeric_dtype(df['upvotes']):
             daily_agg['Avg_Upvotes'] = ('upvotes', 'mean')

        if daily_agg:
            try:
                df_for_daily = df.dropna(subset=['date'])
                if not df_for_daily.empty:
                    daily = df_for_daily.groupby(df_for_daily['date'].dt.date).agg(**daily_agg).reset_index()
                    cols_to_round = [key for key in daily_agg.keys()]
                    for col in cols_to_round:
                        if col in daily.columns:
                            daily[col] = daily[col].round(4)
                    csv_path_daily = os.path.join(RESULTS_DIR, "reddit_sentiment_daily.csv")
                    print(f"Speichere Tageswerte-CSV: {csv_path_daily}")
                    daily.to_csv(
                        csv_path_daily,
                        index=False,
                        sep=";",
                        encoding="utf-8-sig",
                        quoting=csv.QUOTE_ALL
                    )
                else:
                    print("WARNUNG: Keine gültigen Daten für die Tagesaggregation nach Filterung vorhanden.")
            except Exception as agg_err:
                 print(f"!!! FEHLER bei der Tagesaggregation: {agg_err}")
        else:
            print("WARNUNG: Keine numerischen Spalten für die Tagesaggregation gefunden.")

    # === Wortfrequenzanalyse ===
    if 'cleaned' in df.columns:
        try:
            cleaned_texts = df["cleaned"].dropna().astype(str)
            if not cleaned_texts.empty:
                all_words_text = " ".join(cleaned_texts)
                words = re.findall(r"\b[a-z]{3,}\b", all_words_text)
                filtered = [w for w in words if w not in ENGLISH_STOP_WORDS]
                word_freq = Counter(filtered).most_common(50)
                if word_freq:
                    freq_df = pd.DataFrame(word_freq, columns=["word", "count"])
                    csv_path_freq = os.path.join(RESULTS_DIR, "word_frequencies.csv")
                    print(f"Speichere Wortfrequenz-CSV: {csv_path_freq}")
                    freq_df.to_csv(
                        csv_path_freq,
                        index=False,
                        sep=";",
                        encoding="utf-8-sig",
                        quoting=csv.QUOTE_ALL
                    )
            else:
                print("WARNUNG: Keine Texte für die Wortfrequenzanalyse vorhanden.")
        except Exception as freq_err:
            print(f"!!! FEHLER bei der Wortfrequenzanalyse: {freq_err}")

    print("\n✅ Skript abgeschlossen.")

else:
    print("⚠️ Keine passenden Beiträge gefunden oder alle Posts wurden herausgefiltert.")