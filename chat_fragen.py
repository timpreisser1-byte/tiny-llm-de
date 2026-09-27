"""Fragen mit nachprüfbarer Antwort aus der Wissensbasis erzeugen.

Dreißig handgeschriebene Fragen reichen nicht: Zwei Fragen Unterschied sind
dort sieben Prozentpunkte, und damit ist kein Vergleich auseinanderzuhalten.

Erzeugt wird nach demselben Grundsatz, der im Projekt schon das Lesen gebracht
hat - Vorlagen über den ganzen Bestand. Nur zielt es diesmal nicht auf eine
markierte Lücke, sondern auf die **natürliche Frage**:

    Absatz   Der Eiffelturm ist ein 330 Meter hoher Eisenfachwerkturm ...
    Frage    Wie hoch ist der Eiffelturm?
    Antwort  330

Jede Vorlage bindet eine Frageform an eine Antwortart. "Wie hoch" verlangt
eine Länge, "wann" ein Jahr, "wie viele Einwohner" eine Anzahl. Genau diese
Bindung fehlt dem Modell bisher, und genau sie lässt sich hier messen.

Zwei Bedingungen machen die Fragen brauchbar:

    einmalig   die Antwort kommt genau einmal im Absatz vor, sonst ist die
               Prüfung mehrdeutig
    Titel      die Frage nennt den Artikeltitel, sonst ist sie ohne den
               Absatz gar nicht zu stellen

    python chat_fragen.py --anzahl 200 --aus fragen.json
"""

import argparse
import json
import random
import re
import sys
import time

# (Muster im Absatz, Fragevorlage, Gruppe mit der Antwort)
# Die Muster sind bewusst eng: lieber wenige saubere Fragen als viele schiefe.
# Ein Name: ein oder zwei grossgeschriebene Woerter, keine Satzanfaenge.
N = r"([A-ZÄÖÜ][\wäöüß-]{2,}(?:\s+[A-ZÄÖÜ][\wäöüß-]{2,})?)"

# Namensfragen. Sie sind die eigentliche Luecke: Im Lesetest bringt der Absatz
# bei Zahlen +50 Punkte, bei Namen +6,3. Der auffaelligste Fehler des Chatbots
# ist genau das - "Maerce von Bismarck wurde 1749 in Frankfurt geboren".
VORLAGEN_NAME = [
    (r"\bvon\s+" + N + r"\s+(?:entwickelt|erfunden|entworfen)", "Wer hat {t} entwickelt?", "Person"),
    (r"\bvon\s+" + N + r"\s+(?:gegründet|erbaut|errichtet)", "Wer hat {t} gegründet?", "Person"),
    (r"\bvon\s+" + N + r"\s+(?:komponiert|geschrieben|verfasst)", "Wer hat {t} geschrieben?", "Person"),
    (r"\bliegt\s+(?:in|im)\s+(?:der\s+|dem\s+)?" + N, "Wo liegt {t}?", "Ort"),
    (r"\bbefindet sich\s+(?:in|im)\s+(?:der\s+|dem\s+)?" + N, "Wo befindet sich {t}?", "Ort"),
    (r"\bmündet\s+(?:in|bei)\s+(?:die\s+|den\s+|das\s+)?" + N, "Wo mündet {t}?", "Ort"),
    (r"\b(?:Sitz|Hauptsitz)\s+(?:in|im)\s+" + N, "Wo hat {t} seinen Sitz?", "Ort"),
    (r"\bHauptstadt\s+(?:ist|war)\s+" + N, "Was ist die Hauptstadt von {t}?", "Ort"),
    (r"\bentspringt\s+(?:in|im|am|bei)\s+(?:der\s+|dem\s+)?" + N, "Wo entspringt {t}?", "Ort"),
    (r"\bgehört\s+zu(?:m|r)?\s+" + N, "Wozu gehört {t}?", "Ort"),
]

VORLAGEN = [
    (r"\b(\d[\d.,]*)\s*(?:m|Meter)\s+(?:hoh|hoch)", "Wie hoch ist {t}?", "Höhe"),
    (r"\bHöhe von\s+(\d[\d.,]*)\s*(?:m|Meter)", "Wie hoch ist {t}?", "Höhe"),
    (r"\b(\d[\d.,]*)\s*(?:km|Kilometer)\s+lang", "Wie lang ist {t}?", "Länge"),
    (r"\bLänge von\s+(\d[\d.,]*)\s*(?:km|Kilometer)", "Wie lang ist {t}?", "Länge"),
    (r"\b(?:Tiefe|tief)[^.]{0,30}?(\d[\d.,]*)\s*(?:m|Meter)", "Wie tief ist {t}?", "Tiefe"),
    (r"\b(\d[\d.,]*)\s+Einwohner", "Wie viele Einwohner hat {t}?", "Anzahl"),
    (r"\bFläche von\s+(\d[\d.,]*)\s*(?:km²|Quadratkilometer)", "Wie groß ist die Fläche von {t}?", "Fläche"),
    (r"\b(\d[\d.,]*)\s*km²", "Wie groß ist die Fläche von {t}?", "Fläche"),
    (r"\bwurde\b[^.]{0,60}?\b(1[0-9]{3}|20[0-2][0-9])\b[^.]{0,40}?\bgeboren", "Wann wurde {t} geboren?", "Jahr"),
    (r"\bgeboren[^.]{0,40}?\b(1[0-9]{3}|20[0-2][0-9])\b", "Wann wurde {t} geboren?", "Jahr"),
    (r"\b(1[0-9]{3}|20[0-2][0-9])\b[^.]{0,50}?\bgegründet", "Wann wurde {t} gegründet?", "Jahr"),
    (r"\bgegründet[^.]{0,40}?\b(1[0-9]{3}|20[0-2][0-9])\b", "Wann wurde {t} gegründet?", "Jahr"),
    (r"\b(1[0-9]{3}|20[0-2][0-9])\b[^.]{0,50}?\beröffnet", "Wann wurde {t} eröffnet?", "Jahr"),
    (r"\b(1[0-9]{3}|20[0-2][0-9])\b[^.]{0,50}?\berbaut", "Wann wurde {t} erbaut?", "Jahr"),
    (r"\bstarb[^.]{0,50}?\b(1[0-9]{3}|20[0-2][0-9])\b", "Wann starb {t}?", "Jahr"),
]


# Dieselben Aufgaben, andere Worte. KEINE dieser Formen kommt im Training
# vor - sie ist der ehrliche Gegentest. Das Projekt hat schon einmal
# gemessen, was ohne sie passiert: eine einzige Vorlage bringt +42 Punkte
# auf der geuebten Form und +3 auf jeder anderen. Wer nur auf der geuebten
# Form prueft, misst seine eigene Verengung mit.
UMFORMULIERT = {
    "Wie hoch ist {t}?":                   "Welche Höhe erreicht {t}?",
    "Wie lang ist {t}?":                   "Auf welche Länge kommt {t}?",
    "Wie tief ist {t}?":                   "Welche Tiefe hat {t}?",
    "Wie viele Einwohner hat {t}?":        "Wie groß ist die Bevölkerung von {t}?",
    "Wie groß ist die Fläche von {t}?":    "Welchen Flächeninhalt besitzt {t}?",
    "Wann wurde {t} geboren?":             "In welchem Jahr kam {t} zur Welt?",
    "Wann wurde {t} gegründet?":           "Aus welchem Jahr stammt die Gründung von {t}?",
    "Wann wurde {t} eröffnet?":            "In welchem Jahr ging {t} in Betrieb?",
    "Wann wurde {t} erbaut?":              "Aus welchem Jahr stammt der Bau von {t}?",
    "Wann starb {t}?":                     "In welchem Jahr endete das Leben von {t}?",
    "Wer hat {t} entwickelt?":             "Auf wen geht die Entwicklung von {t} zurück?",
    "Wer hat {t} gegründet?":              "Von wem stammt die Gründung von {t}?",
    "Wer hat {t} geschrieben?":            "Aus wessen Feder stammt {t}?",
    "Wo liegt {t}?":                       "In welcher Gegend findet man {t}?",
    "Wo befindet sich {t}?":               "An welchem Ort steht {t}?",
    "Wo mündet {t}?":                      "In welches Gewässer fließt {t}?",
    "Wo hat {t} seinen Sitz?":             "An welchem Ort ist {t} ansässig?",
    "Was ist die Hauptstadt von {t}?":     "Welche Stadt regiert {t}?",
    "Wo entspringt {t}?":                  "An welcher Stelle beginnt {t}?",
    "Wozu gehört {t}?":                    "Welcher Einheit ist {t} zugeordnet?",
}


def balken(i, n, t0):
    b = 22
    v = int(b * i / max(n, 1))
    sys.stderr.write(f"\r  [{'█'*v}{'░'*(b-v)}] {i:>5}/{n}  {time.time()-t0:5.0f}s")
    sys.stderr.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="wiki/chunks.txt")
    ap.add_argument("--anzahl", type=int, default=200)
    ap.add_argument("--aus", default="chat_fragen.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--je-gruppe", type=int, default=0,
                    help="hoechstens so viele je Antwortart (0 = ohne Grenze)")
    ap.add_argument("--art", default="zahl", choices=["zahl", "name", "beides"])
    ap.add_argument("--umformuliert", action="store_true",
                    help="Frageformen verwenden, auf die NICHT trainiert wurde")
    ap.add_argument("--ohne", help="Fragen aus dieser Datei aussparen "
                                   "(damit Training und Pruefung sich nicht "
                                   "ueberschneiden)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    roh = {"zahl": VORLAGEN, "name": VORLAGEN_NAME,
           "beides": VORLAGEN + VORLAGEN_NAME}[args.art]
    muster = [(re.compile(m), v, g) for m, v, g in roh]
    gesperrt = set()
    if args.ohne:
        gesperrt = {e["absatz"] for e in json.load(open(args.ohne, encoding="utf-8"))}
        print(f"  {len(gesperrt):,} Absätze gesperrt")
    zeilen = open(args.chunks, encoding="utf-8").read().splitlines()
    rng.shuffle(zeilen)
    print(f"\n  {len(zeilen):,} Absätze, {len(muster)} Vorlagen ({args.art})\n")

    fragen, gezaehlt, t0 = [], {}, time.time()
    for i, zeile in enumerate(zeilen, 1):
        if len(fragen) >= args.anzahl:
            break
        if ": " not in zeile or zeile in gesperrt:
            continue
        titel, absatz = zeile.split(": ", 1)
        # Titel wie "Liste der ..." oder Klammerzusaetze taugen nicht als Frage
        # Jahres- und Zahlartikel ergeben Unsinn ("Wann wurde 1579 eroeffnet?")
        if (len(titel) < 3 or len(titel) > 40 or titel.startswith("Liste")
                or "(" in titel or len(absatz) < 200
                or titel[0].isdigit() or not any(z.isalpha() for z in titel)):
            continue
        for m, vorlage, gruppe in muster:
            if args.je_gruppe and gezaehlt.get(gruppe, 0) >= args.je_gruppe:
                continue
            treffer = m.search(absatz)
            if not treffer:
                continue
            antwort = treffer.group(1)
            # einmalig, sonst ist die Pruefung mehrdeutig
            if absatz.count(antwort) != 1 or len(antwort) < 2:
                continue
            # Der Name darf nicht der Titel selbst sein - sonst lautet die
            # Antwort auf "Wo liegt Berlin?" schlicht "Berlin".
            if antwort in titel or titel in antwort:
                continue
            # Bei Namen faengt das Muster oft ein Wort zu viel ein
            # ("Nachbargemeinde Stattegg"). Das letzte Wort gilt als
            # gleichwertig, sonst waere die richtige Antwort "Stattegg"
            # falsch - dann misst man die Musterschwaeche statt das Modell.
            wortlaut = vorlage
            if args.umformuliert:
                wortlaut = UMFORMULIERT.get(vorlage)
                if wortlaut is None:
                    continue
            gueltig = [antwort]
            if " " in antwort:
                gueltig.append(antwort.rsplit(" ", 1)[1])
            fragen.append({"frage": wortlaut.format(t=titel), "antwort": gueltig,
                           "absatz": zeile, "titel": titel, "gruppe": gruppe})
            gezaehlt[gruppe] = gezaehlt.get(gruppe, 0) + 1
            break
        if i % 2000 == 0:
            balken(len(fragen), args.anzahl, t0)
    sys.stderr.write("\r" + " " * 60 + "\r")

    json.dump(fragen, open(args.aus, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  {len(fragen)} Fragen -> {args.aus}")
    for g, n in sorted(gezaehlt.items(), key=lambda x: -x[1]):
        print(f"    {g:<10}{n:>5}")
    print(f"\n  {'Beispiele:':<12}")
    for f in fragen[:6]:
        print(f"    {f['frage']:<44} -> {f['antwort'][0]}")


if __name__ == "__main__":
    main()
