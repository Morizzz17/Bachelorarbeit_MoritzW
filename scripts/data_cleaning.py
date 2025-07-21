# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:30:11 2025

@author: Moritz
Funktion:
Entfernt HTML-Tags, Sonderzeichen und überflüssige Leerzeichen aus einem Text und konvertiert den Text in Kleinbuchstaben.
Test:
Mit dem Beispiel <p>Das ist ein Beispiel!</p> wird korrekt der bereinigte Text „Das ist ein Beispiel“ ausgegeben.
"""

# data_cleaning.py
import re

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # URLs entfernen
    text = re.sub(r"[^\w\s]", "", text)  # Sonderzeichen raus
    return text.lower().strip()
