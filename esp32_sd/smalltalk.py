"""Tiny deterministic smalltalk layer for the 8 MB chatbot path."""

import re


def _clean(text):
    return re.sub(r"\s+", " ", text.lower()).strip(" .!?")


def smalltalk_answer(question, topic=""):
    q = _clean(question)
    if not q:
        return "Bitte gib eine Frage ein."
    if q in {"hallo", "hi", "hey", "guten tag", "moin", "servus"}:
        return "Hallo! Ich kann plaudern und Fragen aus meiner Wikipedia-Karte beantworten."
    if q in {"danke", "dankeschön", "vielen dank"}:
        return "Gern. Frag ruhig weiter."
    if q in {"tschüss", "tschuess", "ciao", "bis später", "bis spaeter"}:
        return "Bis später."
    if q in {"wie geht es dir", "wie gehts", "wie geht's"}:
        return "Mir geht es gut. Ich bin bereit fuer Fragen von der SD-Karte."
    if q in {"wer bist du", "was bist du"}:
        return "Ich bin eine kleine Offline-Wissens-KI fuer den ESP32-S3."
    if q in {"was kannst du", "was kannst du machen"}:
        return "Ich kann Smalltalk, kurze Fakten aus Wikipedia finden und belegte Antworten geben."
    if q in {"bist du eine ki", "bist du ein chatbot"}:
        return "Ja. Ich bin als kleiner Chatbot mit lokalem Wissen gedacht."
    if q in {"bist du ein mensch", "bist du menschlich"}:
        return "Nein, ich bin Software. Aber ich kann freundlich antworten."
    if q in {"was machen wir gerade", "woran arbeiten wir"}:
        return "Wir bauen zuerst eine kleine, zuverlaessige Wissens-KI fuer 8 MB."
    if q in {"was war das thema", "worueber reden wir", "worüber reden wir"}:
        if topic:
            return f"Gerade ist das Thema: {topic}."
        return "Noch gibt es kein gespeichertes Thema."
    if q in {"hilfe", "help"}:
        return "Frag zum Beispiel: Was ist ein Transistor? Wie hoch ist der Eiffelturm? Oder sag einfach Hallo."
    return None
