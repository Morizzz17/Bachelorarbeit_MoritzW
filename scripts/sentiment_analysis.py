# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:12:19 2025

@author: Moritz
Funktion:
Führt mithilfe von VADER eine Sentiment-Analyse auf einem übergebenen Text durch und gibt ein Dictionary mit den Scores (negativ, neutral, positiv und compound) zurück.
"""

# sentiment_analysis.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- NEU: WSB-spezifisches Lexikon ---
# Werte sind +/- Sentiment-Intensität (ähnlich VADERs Skala, ca. -4 bis +4)
# Muss experimentell angepasst werden!
wsb_lexicon = {
    
    # WSB Begriffe
    "moon": 3.0, 
    "mooning": 3.0, 
    "tendies": 2.5, 
    "yolo": 1.5, # oft neutral bis positiv verwendet
    "rocket": 3.0, # Emoji wird separat behandelt, aber das Wort auch
    "diamond hands": 3.0, 
    "hold": 1.0, 
    "hodl": 1.0, 
    "buy the dip": 2.0, 
    "btfd": 1.5,
    "squeeze": 3.0, 
    "gamma squeeze": 1.5, 
    "short squeeze": 1.5,
    "autist": 0.5,  # Kontextabhängig, oft neutral/positiv auf WSB
    "retard": 1.0,  # Kontextabhängig, oft neutral/positiv auf WSB
    "ape": 1.5,     # Kontextabhängig
    "bullish": 2.5, 
    "bull": 1.0,
    "guh": -3.0, 
    "drill": -2.5, 
    "red": -1.5, # Oft negativ im Kontext von Portfolio/Markt
    "bleed": -2.0,
    "paper hands": -3.0, 
    "bearish": -2.5, 
    "bear": -1.5,
    "puts": -0.5,  # Eher neutral, aber oft in negativen Szenarien
    "calls": 0.5,  # Eher neutral, aber oft in positiven Szenarien
    "fuck": -1.5, # Verstärkt oft Negativität, aber VADER kennt es schon
    "fucking": -1.0, # Als Verstärker
    "citadel": -3.0, # Steht für Citadel LLC, negativ erwähnt
    "melvin": -1.0,  # Oft negativ erwähnt
    "sec": -0.5,    # Oft neutral bis negativ
    "robinhood": -1.5, # Oft negativ, nachtragend aufgrund
    "bagholder": -2.0, 
    "bagholding": -2.0,
    "gme": 1.5,
    "amc": 0.5,
    
    # Emojis (aus EMOJI_SENTIMENT übertragen und auf Valenz-Skala geschätzt)
    '🚀': 4.0,  # Sehr stark positiv
    '🌕': 3.0,  # Stark positiv
    '💎': 3.0,  # Stark positiv (Kontext WSB)
    '🙌': 2.5,  # Positiv (Kontext WSB)
    '📈': 2.0,  # Positiv
    '💰': 2.0,  # Positiv
    '🤑': 2.0,  # Positiv
    '😂': 1.0,  # Eher positiv (Lachen)
    '🤣': 1.0,  # Eher positiv (Lachen)
    '🔥': 1.5,  # Oft positiv ("fire")
    '👍': 1.5,  # Positiv

    '😭': -2.0, # Negativ
    '📉': -2.0, # Negativ
    '🤡': -3.0, # Stark negativ (Kontext WSB)
    '🐻': -2.5, # Negativ (Bär)
    '🌈🐻': -3.5, # Sehr stark negativ (Regenbogen-Bär)
    '💩': -1.5, # Negativ
    '😡': -1.0, # Eher negativ
    '😠': -1.0, # Eher negativ
    '👎': -1.5, # Negativ
    '📄🙌': -3.5, # Paper hands Emoji sehr negativ
}

analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(wsb_lexicon)

def analyze_sentiment(text):
    """
    Führt eine Sentiment-Analyse durch und gibt ein Dictionary mit den Scores zurück.
    """
    return analyzer.polarity_scores(text)
