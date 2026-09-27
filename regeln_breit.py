"""Breitere Auslöser für die regelbasierte Antwort.

Gemessen an `esp32_sd/extractive_answer.py`: Wo die Regel greift, liegt sie
zu 92 % richtig - aber sie greift zu selten. 81 % der erzeugten Fragen, 22 %
der handgeschriebenen, 4 % der umformulierten. Der Engpass ist die
**Auslösung**, nicht die Genauigkeit.

Zwei Ursachen, beide im Code sichtbar:

    1. `has_typed_answer` verzweigt über sechs feste Frageanfänge per
       startswith. "Welche Höhe erreicht X?" fällt durch, obwohl es
       dieselbe Frage ist.
    2. Für Orte und Personen gibt es gar keine Regel - dabei sind das die
       beiden größten Antwortgruppen im Bestand.

Hier wird die Frage zuerst in eine **Art** übersetzt (viele Formulierungen
je Art), und erst danach greift ein Muster über den Absatz. Das trennt die
zwei Aufgaben, die vorher vermischt waren: die Frage verstehen und die
Antwort finden.

Codex' Datei bleibt unangetastet; beide Wege sind nebeneinander messbar.

    python regeln_breit.py            # Selbsttest über die Fragensätze
"""

import os
import re

# ---- 1. Frage -> Art. Viele Formulierungen, absichtlich ohne startswith.
ARTEN = [
    ("Höhe",   r"wie hoch|welche höhe|höhe (?:hat|von|des|der)|hoch (?:ist|ragt)|"
               r"auf welche höhe|wie weit.{0,12}(?:hinauf|nach oben)"),
    ("Tiefe",  r"wie tief|welche tiefe|tiefe (?:hat|von|des|der)|"
               r"bis zu welcher tiefe|wie weit.{0,20}tiefe"),
    ("Länge",  r"wie lang|welche länge|länge (?:hat|von|des|der)|"
               r"wie weit erstreckt|über welche strecke|auf welche länge"),
    ("Fläche", r"wie groß|wie gross|welche fläche|fläche (?:hat|von|des|der)|"
               r"flächeninhalt|wie viel fläche|welche größe"),
    ("Anzahl", r"wie viele|wieviele|wie viel(?!\s*fläche)|einwohnerzahl|"
               r"wie groß ist die bevölkerung|von wie vielen"),
    ("Jahr",   r"^wann|in welchem jahr|welches jahr|seit wann|aus welchem jahr|"
               r"zur welt|geburtsjahr|todesjahr|wann ist|wann wurde|wann war"),
    ("Person", r"^wer |von wem|wessen|wer hat|wer war|wer steckt|auf wen|"
               r"wem verdankt|urheber|erfinder|aus wessen"),
    ("Ort",    r"^wo |wohin|in welcher (?:gegend|umgebung|stadt|region)|"
               r"an welchem ort|welchem ort|in welchem land|wo genau|"
               r"welches gewässer|zu welcher|wozu gehört|in welchen|"
               r"hauptstadt|welche stadt|welches land|welcher ort"),
    # Auffangart fuer alles, was nach einem Namen fragt, ohne dass die Art
    # feststeht: "Welcher Fluss fliesst durch X?", "Wie heisst das Y?",
    # "Welche Waehrung gilt in Z?". Gemessen war das mit 31,7 % der groesste
    # Einzelposten der Fehlschlaege - groesser als alles andere zusammen.
    ("Name",   r"^welche[rs]? \w+|^wie hei|^was ist (?:der|die|das) \w+|"
               r"^was nennt man|^was bedeutet|^womit|^woraus|"
               r"^welches (?:tier|element|metall|gas|organ|instrument|land)"),
]
ARTEN = [(a, re.compile(m, re.IGNORECASE)) for a, m in ARTEN]

# ---- 2. Art -> Muster über den Absatz. Gruppe 1 ist die Antwort.
MUSTER = {
    "Höhe":   [r"\b(\d{1,4}(?:[.,]\d+)?)\s*(?:m|Meter)\s+(?:hoch|hohe[rnms]?)",
               r"[Hh]öhe\s+von\s+(\d{1,4}(?:[.,]\d+)?)\s*(?:m|Meter)",
               r"\b(\d{1,4}(?:[.,]\d+)?)\s*(?:m|Meter)\s+über\s+(?:dem\s+)?Meer"],
    "Tiefe":  [r"\b(\d{1,4}(?:[.,]\d+)?)\s*(?:m|Meter)\s+(?:tief|tiefe[rnms]?)",
               r"[Tt]iefe\s+von\s+(\d{1,4}(?:[.,]\d+)?)\s*(?:m|Meter)"],
    "Länge":  [r"\b(\d{1,5}(?:[.,]\d+)?)\s*(?:km|Kilometer)\s+(?:lang|lange[rnms]?)",
               r"[Ll]änge\s+von\s+(\d{1,5}(?:[.,]\d+)?)\s*(?:km|Kilometer)"],
    "Fläche": [r"\b(\d{1,3}(?:[.\s]\d{3})*(?:,\d+)?)\s*(?:km²|km2|Quadratkilometer)",
               r"[Ff]läche\s+von\s+(\d{1,3}(?:[.\s]\d{3})*(?:,\d+)?)"],
    "Anzahl": [r"\b(\d{1,3}(?:[.\s]\d{3})*(?:,\d+)?)\s*Einwohner",
               r"\b(\d{1,3}(?:[.\s]\d{3})*)\s*(?:Mitglieder|Personen|Menschen)"],
    # Jahr war mit 43,1 % die schwaechste typisierte Art, weil das Muster
    # jede Jahreszahl im Absatz nahm. Jetzt muss sie beim passenden Vorgang
    # stehen; die nackte Jahreszahl bleibt nur als letzter Ausweg.
    "Jahr":   [r"(1[0-9]{3}|20[0-2][0-9])\b[^.]{0,60}?\bVERB",
               r"\bVERB[^.]{0,60}?\b(1[0-9]{3}|20[0-2][0-9])",
               r"\b(1[0-9]{3}|20[0-2][0-9])\b"],
    "Person": [r"\bvon\s+([A-ZÄÖÜ][\wäöüß-]{2,}(?:\s+[A-ZÄÖÜ][\wäöüß-]{2,}){0,2})\s+"
               r"(?:entwickelt|erfunden|entworfen|gegründet|erbaut|errichtet|"
               r"geschrieben|komponiert|verfasst|konstruiert)",
               r"\bdurch\s+([A-ZÄÖÜ][\wäöüß-]{2,}(?:\s+[A-ZÄÖÜ][\wäöüß-]{2,}){0,2})\s+"
               r"(?:entwickelt|erfunden|gegründet)"],
    # Die Auffangart sucht einen Namen in der Naehe der Frageworte. Sie ist
    # bewusst schwaecher als die typisierten Muster und kommt nur zum Zug,
    # wenn keines von ihnen greift.
    # Absichtlich leer. Die Auffangart erkennt die Frage, liefert aber keine
    # Antwort - gemessen traf sie nur zu 6,7 %, waehrend der Satz-Rueckfall
    # bei 27,7 % liegt. Eine schwache typisierte Antwort ist schlechter als
    # ein zitierter Satz, weil sie ihn verdraengt. Die Art bleibt trotzdem
    # nuetzlich: Sie steuert das Umsortieren.
    "Name":   [],
    "Ort":    [r"\bliegt\s+(?:in|im|am|an der|bei)\s+(?:der\s+|dem\s+|den\s+)?"
               r"([A-ZÄÖÜ][\wäöüß-]{2,}(?:\s+[A-ZÄÖÜ][\wäöüß-]{2,})?)",
               r"\bbefindet sich\s+(?:in|im|am|bei)\s+(?:der\s+|dem\s+)?"
               r"([A-ZÄÖÜ][\wäöüß-]{2,}(?:\s+[A-ZÄÖÜ][\wäöüß-]{2,})?)",
               r"\bmündet\s+(?:in|bei)\s+(?:die\s+|den\s+|das\s+)?"
               r"([A-ZÄÖÜ][\wäöüß-]{2,})",
               r"\bgehört\s+zu(?:m|r)?\s+"
               r"([A-ZÄÖÜ][\wäöüß-]{2,}(?:\s+[A-ZÄÖÜ][\wäöüß-]{2,})?)",
               r"\bHauptstadt\s+(?:ist|war)\s+([A-ZÄÖÜ][\wäöüß-]{2,})",
               r"\bim\s+(?:Bundesland|Land|Kanton|Landkreis)\s+"
               r"([A-ZÄÖÜ][\wäöüß-]{2,}(?:\s+[A-ZÄÖÜ][\wäöüß-]{2,})?)"],
}
# Welches Verb im Absatz zu welchem Frageverb gehoert. Ohne diese Bindung
# nimmt die Jahresregel die erstbeste Zahl - bei "Wann wurde X gegruendet?"
# also womoeglich das Jahr eines Umbaus.
JAHR_VERB = [
    (r"geboren|zur welt|geburtsjahr", r"geboren"),
    (r"gestorben|starb|todesjahr|endete das leben", r"(?:starb|gestorben|Tod)"),
    (r"gegründet|gruendung|entstand|datiert", r"(?:gegründet|Gründung|entstand)"),
    (r"eröffnet|in betrieb|gebaut|erbaut|errichtet|seit wann",
     r"(?:eröffnet|erbaut|errichtet|Betrieb|fertiggestellt)"),
    (r"entdeckt", r"entdeckt"),
    # "Wann fiel die Berliner Mauer?" lieferte 1961 - das Baujahr. Ohne
    # dieses Verb greift die Regel die erstbeste Jahreszahl im Absatz, und
    # die gehoert beim selben Gegenstand oft zum gegenteiligen Ereignis.
    (r"\bfiel|fall der|geöffnet|ende der|aufgelöst|abgerissen",
     r"(?:fiel|Fall|geöffnet|Öffnung|Ende|aufgelöst|abgerissen)"),
    (r"begann|ausbruch|beginn", r"(?:begann|Beginn|Ausbruch|brach)"),
    (r"endete|ende des|schluss", r"(?:endete|Ende|Kapitulation)"),
]
JAHR_VERB = [(re.compile(f, re.IGNORECASE), v) for f, v in JAHR_VERB]

MUSTER = {a: [re.compile(m) if "VERB" not in m else m for m in ms]
          for a, ms in MUSTER.items()}

EINHEIT = {"Höhe": "{a} Meter", "Tiefe": "{a} Meter", "Länge": "{a} Kilometer",
           "Fläche": "{a} Quadratkilometer", "Anzahl": "{a} Einwohner",
           "Jahr": "{a}", "Person": "{a}", "Ort": "{a}", "Name": "{a}"}


def art_der_frage(frage):
    """Welche Art von Antwort verlangt diese Frage? None, wenn unklar."""
    f = frage.lower().strip(" .!?")
    for art, muster in ARTEN:
        if muster.search(f):
            return art
    return None


def antwort(frage, absatz):
    """Kurze, wörtlich belegte Antwort - oder None, wenn nichts passt.

    Bewusst None statt eines Rueckfalls: Der Rueckfall (einen ganzen Satz
    zitieren) gehoert dem Aufrufer, damit beide Anteile getrennt messbar
    bleiben. Genau diese Trennung hat gezeigt, dass die 36,7 % der alten
    Regeln aus 92 % Treffern und 27,7 % Zufall bestehen.
    """
    art = art_der_frage(frage)
    if not art:
        return None
    # Titel abtrennen: "Eiffelturm: Der Eiffelturm ist ..." -> nur den Rumpf
    rumpf = absatz.split(": ", 1)[-1] if ": " in absatz[:60] else absatz
    verb = None
    if art == "Jahr":
        f = frage.lower()
        verb = next((v for m, v in JAHR_VERB if m.search(f)), None)
    for muster in MUSTER[art]:
        if isinstance(muster, str):          # enthaelt noch den Platzhalter
            if verb is None:
                continue                     # ohne Verb ueberspringen
            muster = re.compile(muster.replace("VERB", verb))
        tr = muster.search(rumpf)
        if tr:
            return EINHEIT[art].format(a=tr.group(1))
    return None


if __name__ == "__main__":
    import json
    from chat_test import enthaelt
    from esp32_sd.extractive_answer import has_typed_answer
    from wissen_index import Wissensbasis

    wb = None
    print(f"\n  {'Fragensatz':<24}{'Verfahren':<12}{'greift':>9}{'richtig':>10}")
    for datei, orakel in (("chat_fragen.json", True), ("chat_fragen_um.json", True),
                          ("chat_fragen_1000.json", True), ("chat_fragen_hand.json", False)):
        if not os.path.exists(datei):
            continue                     # aeltere Fragensaetze liegen nicht mehr bei
        roh = json.load(open(datei, encoding="utf-8"))
        if not orakel and wb is None:
            wb = Wissensbasis("wiki_gestuft")
        absaetze = [(e["absatz"][:1000] if orakel else
                     wb.bester_text(e["frage"], max_zeichen=1000, absaetze=2) or "")
                    for e in roh]
        for name, greift, gib in (
                ("Codex", lambda f, a: has_typed_answer(f, a),
                 lambda f, a: __import__("esp32_sd.extractive_answer",
                                         fromlist=["x"]).answer_from_text(f, a)),
                ("breit", lambda f, a: antwort(f, a) is not None, antwort)):
            n_g = ok = 0
            for e, a in zip(roh, absaetze):
                if greift(e["frage"], a):
                    n_g += 1
                    ok += enthaelt(gib(e["frage"], a) or "", e["antwort"])
            print(f"  {datei[:23]:<24}{name:<12}{n_g:>6}/{len(roh)}"
                  f"{ok/max(n_g,1)*100:>9.1f} %")


# ---------------------------------------------------------------- Umsortieren

FRAGEWOERTER = {
    "wie", "was", "wer", "wo", "wann", "welche", "welcher", "welches", "welchem",
    "ist", "hat", "sind", "haben", "der", "die", "das", "ein", "eine", "einen",
    "den", "dem", "des", "von", "vom", "in", "im", "am", "an", "auf", "für",
    "viele", "viel", "man", "es", "und", "oder", "gibt", "war", "wurde", "heißt",
    "besitzt", "erreicht", "kommt", "steht", "liegt", "sich", "seine", "ihre",
}


def inhaltswoerter(frage):
    """Die Woerter, die den Gegenstand der Frage benennen."""
    return {w for w in re.findall(r"[a-zäöüß]{4,}", frage.lower())
            if w not in FRAGEWOERTER}


def waehle_absatz(frage, kandidaten, tiefe=20):
    """Aus vielen Treffern den nehmen, der die Frage beantworten KANN.

    Gemessen: Die Antwort steht bei 81,7 % der Fragen unter den besten 30
    Treffern, aber nur bei 51,7 % unter den besten zwei. Dreissig Punkte
    liegen also allein in der Sortierung.

    BM25 sortiert nach Wortuebereinstimmung und weiss nichts davon, welche
    ART von Antwort gesucht ist. Wir wissen es: `art_der_frage` liefert sie,
    und `antwort` sagt, ob ein Absatz eine solche Angabe ueberhaupt enthaelt.
    Ein Absatz ohne jede Hoehenangabe kann die Frage nach der Hoehe nicht
    beantworten, egal wie gut die Woerter passen.

    Das ist Umsortieren nach Beantwortbarkeit statt nach Aehnlichkeit - und
    es kostet nur die Regelpruefung je Kandidat, kein Modell.

    Zurueck kommt (Absatz, Antwort-oder-None). Faellt keiner durch, bleibt
    der beste BM25-Treffer stehen.
    """
    if not kandidaten:
        return "", None
    if art_der_frage(frage) is None:
        return kandidaten[0], None      # keine Art bekannt, nichts zu sortieren
    # Beantwortbarkeit allein reicht nicht: Ein Absatz ueber irgendeinen Turm
    # enthaelt auch "X Meter hoch". Gemessen brachte die reine Artpruefung
    # nur 38,3 -> 40,0 % und wurde mit wachsender Tiefe schlechter. Es muss
    # also zusaetzlich um den GEGENSTAND der Frage gehen.
    woerter = inhaltswoerter(frage)
    bester = None
    for rang, text in enumerate(kandidaten[:tiefe]):
        a = antwort(frage, text)
        if not a:
            continue
        titel = text.split(":", 1)[0].lower()
        passt = sum(1 for w in woerter if w in titel)
        # Rangnaehe zaehlt mit: BM25 hat ja schon Arbeit geleistet
        note = passt * 3 - rang * 0.1
        if bester is None or note > bester[0]:
            bester = (note, text, a)
        if passt and rang == 0:
            break                        # bester Treffer passt schon, fertig
    if bester and bester[0] > 0:
        return bester[1], bester[2]
    return kandidaten[0], antwort(frage, kandidaten[0])
