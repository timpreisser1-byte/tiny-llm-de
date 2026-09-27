"""Trainingspaare mit Formvielfalt - und einer offenen Entscheidung.

Erste Runde gemessen: 87 % auf der geübten Frageform, 26,7 % auf
handgeschriebenen Fragen - schlechter als vor dem Training. Der Grund war zu
wenig Formvielfalt: 25 Schablonen, alle nach demselben Bauplan.

Hier bekommt jedes Muster vier Formulierungen, drei fürs Training und eine
zurückgehaltene für die Prüfung (`chat_formen.py`).

Die zweite offene Frage ist die Ablehnung. Runde eins hat gezeigt, dass sie
in beide Richtungen schiefgehen kann:

    zu wenige Ablehnungen   das Modell erfindet          43,3 % erfunden
    zu viele/falsche        es lehnt Beantwortbares ab   "Worum geht es hier?"
                                                         -> "steht nicht im Text"

    python -m training.qa_pairs --anzahl 60000
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
import sys
import time

from lm.chat_format import mit_wissen
from training.question_forms import MUSTER, formen

FORM = {"Höhe": "{a} Meter", "Tiefe": "{a} Meter", "Länge": "{a} Kilometer",
        "Fläche": "{a} Quadratkilometer", "Anzahl": "{a} Einwohner",
        "Jahr": "{a}", "Person": "{a}", "Ort": "{a}"}

ABLEHNUNGEN = [
    "Das steht nicht im Text.", "Der Abschnitt erwähnt das nicht.",
    "Dazu steht hier nichts.", "Das geht aus dem Text nicht hervor.",
    "Im Text finde ich dazu nichts.", "Der Absatz sagt dazu nichts.",
]


def waehle_ablehnungen(treffer, rng, anteil):
    """Welche Beispiele sollen eine Ablehnung als richtige Antwort bekommen?

    `treffer` ist eine Liste von Fundstellen, je ein dict mit:
        frage    die fertige Frage, z. B. "Wie tief ist Bodensee?"
        wissen   der Absatz, in dem die Antwort steht
        antwort  die kurze richtige Antwort, z. B. "251 Meter"
        gruppe   die Antwortart: Höhe, Tiefe, Länge, Anzahl, Fläche,
                 Jahr, Person, Ort

    Zurückgeben: eine Liste neuer dicts im selben Format, aber mit einer
    Ablehnung aus ABLEHNUNGEN als `antwort` und `gruppe` = "Ablehnung".

    `anteil` ist der gewünschte Anteil an der Gesamtmenge (Vorgabe 0.2),
    `rng` ist ein random.Random für reproduzierbare Auswahl.

    Gemessen am 18.09.2026, was das Modell ohne solche Beispiele tut: bei
    einem absichtlich falschen Absatz (Kartoffel) antwortete es auf 14 von 14
    Fragen, davon 13 erfunden -- "Der Eiffelturm ist eine Nutzpflanze aus der
    Familie der Nachtschattengewächse." Es nimmt das Subjekt aus der Frage und
    das Prädikat aus dem Absatz.

    Genau dagegen wird hier trainiert, und zwar in drei Härtegraden:

      schwer  gleiche Antwortart, anderer Gegenstand. Der Absatz enthält eine
              Höhenangabe, nur eben die eines anderen Bergs. Das ist der Fall,
              den das Modell nicht erkennt -- der Absatz sieht beantwortbar
              aus. Deshalb die Hälfte.
      mittel  andere Antwortart, irgendein Absatz. Hier fehlt die Angabe ganz.
      leicht  Frage und Absatz zufällig gepaart.

    Die Verteilung 50/30/20 ist gesetzt, nicht gemessen -- die schweren Fälle
    bekommen das meiste Gewicht, weil sie der beobachtete Fehler sind.
    """
    if not treffer or anteil <= 0:
        return []
    nach_gruppe = defaultdict(list)
    for e in treffer:
        nach_gruppe[e["gruppe"]].append(e)
    gruppen = sorted(nach_gruppe)

    ziel = int(len(treffer) * anteil)
    raus = []
    for _ in range(ziel):
        frage_e = rng.choice(treffer)
        wuerfel = rng.random()
        if wuerfel < 0.5 and len(nach_gruppe[frage_e["gruppe"]]) > 1:
            # schwer: derselbe Antworttyp, anderer Gegenstand
            for _ in range(8):
                fremd = rng.choice(nach_gruppe[frage_e["gruppe"]])
                if fremd["wissen"] != frage_e["wissen"]:
                    break
        elif wuerfel < 0.8 and len(gruppen) > 1:
            # mittel: andere Antwortart
            andere = [g for g in gruppen if g != frage_e["gruppe"]]
            fremd = rng.choice(nach_gruppe[rng.choice(andere)])
        else:
            # leicht: irgendein Absatz
            fremd = rng.choice(treffer)
        if fremd["wissen"] == frage_e["wissen"]:
            continue                     # zufällig doch der richtige Absatz
        # Sicherheitsnetz: steht die Antwort wider Erwarten doch drin, ist es
        # keine Ablehnung, sondern eine falsche Trainingsvorgabe.
        kurz = re.sub(r"[^\wäöüßÄÖÜ,.]+", " ", frage_e["antwort"]).strip()
        kern = kurz.split()[-1] if kurz.split() else ""
        if kern and len(kern) > 2 and kern.lower() in fremd["wissen"].lower():
            continue
        raus.append({"frage": frage_e["frage"],
                     "wissen": fremd["wissen"],
                     "antwort": rng.choice(ABLEHNUNGEN),
                     "gruppe": "Ablehnung"})
    return raus


def balken(i, n, t0):
    b = 22
    v = int(b * i / max(n, 1))
    sys.stderr.write(f"\r  [{'█'*v}{'░'*(b-v)}] {i:>6,}/{n:,}  {time.time()-t0:5.0f}s")
    sys.stderr.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="wiki/chunks.txt")
    ap.add_argument("--anzahl", type=int, default=60000)
    ap.add_argument("--aus", default="data/sft/fragen2.jsonl")
    ap.add_argument("--ohne", action="append", default=[])
    ap.add_argument("--anteil-ablehnung", type=float, default=0.2)
    ap.add_argument("--max-zeichen", type=int, default=900)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kompiliert = [(re.compile(m[0]), m, formen(m, "training")) for m in MUSTER]
    gesperrt = set()
    for p in args.ohne:
        gesperrt |= {e["absatz"] for e in json.load(open(p, encoding="utf-8"))
                     if e.get("absatz")}
    zeilen = open(args.chunks, encoding="utf-8").read().splitlines()
    rng.shuffle(zeilen)
    print(f"\n  {len(zeilen):,} Absätze, {len(MUSTER)} Muster à 3 Formen, "
          f"{len(gesperrt):,} gesperrt\n")

    treffer, t0 = [], time.time()
    for i, zeile in enumerate(zeilen, 1):
        if len(treffer) >= args.anzahl:
            break
        if ": " not in zeile or zeile in gesperrt:
            continue
        titel, absatz = zeile.split(": ", 1)
        if (len(titel) < 3 or len(titel) > 40 or titel.startswith("Liste")
                or "(" in titel or len(absatz) < 200 or titel[0].isdigit()):
            continue
        for rx, m, wortlaute in kompiliert:
            tr = rx.search(absatz)
            if not tr:
                continue
            antwort = next((g for g in tr.groups() if g), None)
            if not antwort or absatz.count(antwort) != 1 or len(antwort) < 2:
                continue
            if antwort in titel or titel in antwort:
                continue
            # Jede der drei Trainingsformen einmal - das ist der ganze Zweck
            for wortlaut in wortlaute:
                treffer.append({"frage": wortlaut.format(t=titel),
                                "wissen": zeile[:args.max_zeichen],
                                "antwort": FORM[m[2]].format(a=antwort),
                                "gruppe": m[2]})
            break
        if i % 5000 == 0:
            balken(len(treffer), args.anzahl, t0)
    sys.stderr.write("\r" + " " * 60 + "\r")

    daneben = waehle_ablehnungen(treffer, rng, args.anteil_ablehnung)
    alle = treffer + daneben
    rng.shuffle(alle)
    os.makedirs(os.path.dirname(args.aus) or ".", exist_ok=True)
    with open(args.aus, "w", encoding="utf-8") as f:
        for e in alle:
            f.write(json.dumps({"messages": [
                {"role": "user", "content": mit_wissen(e["frage"], e["wissen"])},
                {"role": "assistant", "content": e["antwort"]}]},
                ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"  {len(alle):,} Paare -> {args.aus}")
    for g, n in Counter(e["gruppe"] for e in alle).most_common():
        print(f"    {g:<12}{n:>7,}")


if __name__ == "__main__":
    main()
