#!/usr/bin/env python3
"""training/distractor_data.py — üben, den richtigen Absatz unter mehreren zu finden.

Gemessen am 20.09.2026 auf 400 sauberen Fragen, `ckpt_gross/sft.pt`:

    Bedingung                   Antwort da   richtig   gelesen
    Orakel (nur der richtige)       100,0 %    71,5 %    71,5 %
    1 Absatz aus der Suche           45,0 %    31,5 %    70,0 %
    2 Absätze (heute)                70,8 %    45,8 %    64,7 %
    3 Absätze                        86,0 %    44,5 %    51,7 %

Mit jedem weiteren Absatz FÄLLT die Lesefähigkeit: 71,5 auf 64,7 auf 51,7 %.
Bei drei Absätzen steht die Antwort zu 86 % im Text und wird trotzdem nur in
der Hälfte der Fälle gefunden — die Gesamtquote ist deshalb schlechter als
bei zweien.

Könnte das Modell Ablenkung ignorieren, wären es 86,0 x 71,5 = 61,5 % statt
45,8 %. Sechzehn Punkte, ohne bessere Suche und ohne mehr Speicher.

Die vorhandenen Ablehnungsdaten (`chat_daten2.py`) zeigen immer nur EINEN
Absatz, entweder den richtigen oder einen falschen. Das Heraussuchen aus
mehreren wurde nie geübt. Genau das erzeugt dieses Skript:

    richtiger Absatz + 1 bis 3 Ablenker, Reihenfolge gemischt -> kurze Antwort

Die Ablenker sind bewusst schwer: zur Hälfte derselbe Antworttyp mit anderem
Gegenstand. Ein Absatz mit einer Höhenangabe, nur eben der eines anderen
Bergs — der sieht beantwortbar aus, und genau daran scheitert das Modell.

Ein Teil der Beispiele enthält KEINEN richtigen Absatz und soll abgelehnt
werden. Ohne das lernt das Modell, immer eine Antwort zu erzwingen.

    python -m training.distractor_data --anzahl 40000 --ohne evaluation/questions/templated1000.json
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

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


def gesperrte_absaetze(pfade):
    raus = set()
    for p in pfade:
        if not os.path.exists(p):
            continue
        if p.endswith(".jsonl"):
            for z in open(p, encoding="utf-8"):
                try:
                    t = json.loads(z)["messages"][0]["content"]
                except Exception:
                    continue
                for m in re.finditer(r"Wissen:\s*(.+?)\s*(?:\n|Frage:)", t, re.S):
                    raus.add(m.group(1).strip()[:120])
        else:
            for e in json.load(open(p, encoding="utf-8")):
                if e.get("absatz"):
                    raus.add(e["absatz"].strip()[:120])
    return raus


def balken(i, n, t0):
    b = 24
    v = int(b * i / max(n, 1))
    sys.stderr.write(f"\r  [{'█'*v}{'░'*(b-v)}] {i:>6,}/{n:,}  {time.time()-t0:4.0f}s")
    sys.stderr.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="wiki/chunks.txt")
    ap.add_argument("--anzahl", type=int, default=40000)
    ap.add_argument("--aus", default="data/sft/ablenkung.jsonl")
    ap.add_argument("--ohne", action="append", default=[])
    ap.add_argument("--max-zeichen", type=int, default=500,
                    help="je Absatz, wie in der Kette (--zeichen 1000 fuer zwei)")
    ap.add_argument("--anteil-ablehnung", type=float, default=0.15,
                    help="Anteil ohne richtigen Absatz")
    ap.add_argument("--anteil-hinten", type=float, default=0.55,
                    help="Anteil der Beispiele, in denen der richtige Absatz "
                         "in der hinteren Haelfte steht (sonst zufaellig)")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    gesperrt = gesperrte_absaetze(args.ohne)
    print(f"\n  {len(gesperrt):,} Absätze gesperrt\n")

    rng = random.Random(args.seed)
    kompiliert = [(re.compile(m[0]), m, formen(m, "training")) for m in MUSTER]
    zeilen = open(args.chunks, encoding="utf-8").read().splitlines()
    rng.shuffle(zeilen)

    # --- erst Fundstellen sammeln ---
    treffer, t0 = [], time.time()
    for i, zeile in enumerate(zeilen, 1):
        if len(treffer) >= args.anzahl:
            break
        if ": " not in zeile or zeile.strip()[:120] in gesperrt:
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
            treffer.append({"frage": rng.choice(wortlaute).format(t=titel),
                            "absatz": zeile[:args.max_zeichen],
                            "antwort": FORM[m[2]].format(a=antwort),
                            "kern": antwort, "gruppe": m[2]})
            break
        if i % 5000 == 0:
            balken(len(treffer), args.anzahl, t0)
    sys.stderr.write("\r" + " " * 60 + "\r")

    nach_gruppe = defaultdict(list)
    for e in treffer:
        nach_gruppe[e["gruppe"]].append(e)
    gruppen = sorted(nach_gruppe)

    def ablenker(e, n):
        """n Absätze, die die Frage NICHT beantworten."""
        raus = []
        versuche = 0
        while len(raus) < n and versuche < 20 * n:
            versuche += 1
            # zur Hälfte gleiche Antwortart (schwer), sonst irgendeine
            if rng.random() < 0.5 and len(nach_gruppe[e["gruppe"]]) > 1:
                f = rng.choice(nach_gruppe[e["gruppe"]])
            else:
                f = rng.choice(treffer)
            if f["absatz"] == e["absatz"]:
                continue
            # Sicherheitsnetz: der Ablenker darf die Antwort nicht enthalten
            if e["kern"].lower() in f["absatz"].lower():
                continue
            if f["absatz"] in raus:
                continue
            raus.append(f["absatz"])
        return raus

    # --- daraus Mehrfach-Absatz-Beispiele bauen ---
    beispiele, zaehler = [], defaultdict(int)
    for e in treffer:
        n_ablenk = rng.choice([1, 1, 2, 2, 3])        # meist zwei bis drei Absaetze
        ohne_richtigen = rng.random() < args.anteil_ablehnung
        stoerer = ablenker(e, n_ablenk + (1 if ohne_richtigen else 0))
        if len(stoerer) < n_ablenk:
            continue
        teile = list(stoerer) if ohne_richtigen else [e["absatz"]] + stoerer
        # Position: gemessen am 23.09. faellt die Trefferquote von Platz 1 nach
        # Platz 3 um 26 Punkte (61,0 -> 34,8 %), und zwar mit und ohne
        # Ablenkungstraining gleichermassen. Das Modell greift den ersten
        # passenden Absatz. Reines Mischen reicht dagegen nicht: bei drei
        # Absaetzen landet der richtige nur zu einem Drittel hinten.
        # Deshalb wird die hintere Haelfte gezielt uebergewichtet.
        if not ohne_richtigen and len(teile) > 1:
            if rng.random() < args.anteil_hinten:
                stelle = rng.randrange((len(teile) + 1) // 2, len(teile))
            else:
                stelle = rng.randrange(len(teile))
            rest = list(stoerer)
            rng.shuffle(rest)
            teile = rest[:stelle] + [e["absatz"]] + rest[stelle:]
        else:
            rng.shuffle(teile)
        ziel = rng.choice(ABLEHNUNGEN) if ohne_richtigen else e["antwort"]
        beispiele.append({"messages": [
            {"role": "user", "content": mit_wissen(e["frage"], " ".join(teile))},
            {"role": "assistant", "content": ziel}]})
        zaehler["Ablehnung" if ohne_richtigen else e["gruppe"]] += 1
        zaehler[f"{len(teile)} Absätze"] += 1
        if not ohne_richtigen:
            zaehler[f"richtig an Stelle {teile.index(e['absatz'])+1}"] += 1

    rng.shuffle(beispiele)
    os.makedirs(os.path.dirname(args.aus) or ".", exist_ok=True)
    with open(args.aus, "w", encoding="utf-8") as f:
        for b in beispiele:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

    print(f"  {len(beispiele):,} Beispiele -> {args.aus}")
    laenge = sum(len(b["messages"][0]["content"]) for b in beispiele) / max(len(beispiele), 1)
    print(f"  mittlere Eingabelänge {laenge:.0f} Zeichen\n")
    for k in sorted(zaehler, key=lambda x: -zaehler[x]):
        print(f"    {k:<14}{zaehler[k]:>7,}")


if __name__ == "__main__":
    main()
