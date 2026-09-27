#!/usr/bin/env python3
"""evaluation/clean_test_set.py — einen grossen, sauberen Fragensatz bauen.

Am 19.09. war ein Prüfsatz unbrauchbar, weil 163 von 199 Fragen wörtlich in
den Trainingsdaten standen: beide kamen aus denselben Schablonen und demselben
Korpus. Hier wird beides ausgeschlossen:

  * **Formulierung**: nur die je Muster zurückgehaltene Form
    (`chat_formen.formen(m, "pruefung")`), nie eine der drei Trainingsformen.
  * **Absatz**: jeder Absatz, der in einer Trainingsdatei vorkommt, fliegt
    raus. Die Prüfung wird gegen die Dateien gesperrt, die man ihr nennt.

Ohne die zweite Sperre hätte das Modell die Antwort schon einmal zu genau
diesem Absatz gesehen — nur in anderer Formulierung, was ein 13-M-Modell
mühelos auswendig lernt.

    python -m evaluation.clean_test_set --anzahl 1000 --ohne data/sft/ablehnung_sauber.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
import time

from training.question_forms import MUSTER, formen

FORM = {"Höhe": ["{a}"], "Tiefe": ["{a}"], "Länge": ["{a}"],
        "Fläche": ["{a}"], "Anzahl": ["{a}"], "Jahr": ["{a}"],
        "Person": ["{a}"], "Ort": ["{a}"]}


def absaetze_aus(pfad):
    """Alle Absätze einer Trainings- oder Prüfdatei, normiert."""
    raus = set()
    if not os.path.exists(pfad):
        return raus
    if pfad.endswith(".jsonl"):
        for zeile in open(pfad, encoding="utf-8"):
            try:
                inhalt = json.loads(zeile)["messages"][0]["content"]
            except Exception:
                continue
            # "Wissen: <Absatz>\nFrage: <Frage>"
            m = re.search(r"Wissen:\s*(.+?)\s*(?:\n|Frage:)", inhalt, re.S)
            if m:
                raus.add(m.group(1).strip()[:120])
    else:
        for e in json.load(open(pfad, encoding="utf-8")):
            if e.get("absatz"):
                raus.add(e["absatz"].strip()[:120])
    return raus


def balken(i, n, t0):
    b = 24
    v = int(b * i / max(n, 1))
    sys.stderr.write(f"\r  [{'█'*v}{'░'*(b-v)}] {i:>5}/{n}  {time.time()-t0:4.0f}s")
    sys.stderr.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="wiki/chunks.txt")
    ap.add_argument("--anzahl", type=int, default=1000)
    ap.add_argument("--aus", default="evaluation/questions/templated1000.json")
    ap.add_argument("--ohne", action="append", default=[],
                    help="Datei, deren Absätze gesperrt sind (mehrfach)")
    ap.add_argument("--max-zeichen", type=int, default=900)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    gesperrt = set()
    for p in args.ohne:
        n = len(gesperrt)
        gesperrt |= absaetze_aus(p)
        print(f"  gesperrt aus {os.path.basename(p):<28}{len(gesperrt)-n:>7,} Absätze")
    print(f"  {len(gesperrt):,} Absätze insgesamt gesperrt\n")

    rng = random.Random(args.seed)
    kompiliert = [(re.compile(m[0]), m, formen(m, "pruefung")) for m in MUSTER]
    zeilen = open(args.chunks, encoding="utf-8").read().splitlines()
    rng.shuffle(zeilen)

    raus, verworfen, t0 = [], 0, time.time()
    for i, zeile in enumerate(zeilen, 1):
        if len(raus) >= args.anzahl:
            break
        if ": " not in zeile:
            continue
        if zeile.strip()[:120] in gesperrt:
            verworfen += 1
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
            raus.append({"frage": wortlaute[0].format(t=titel),
                         "antwort": [antwort],
                         "absatz": zeile[:args.max_zeichen],
                         "titel": titel,
                         "gruppe": m[2]})
            break
        if i % 2000 == 0:
            balken(len(raus), args.anzahl, t0)
    sys.stderr.write("\r" + " " * 60 + "\r")

    json.dump(raus, open(args.aus, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    from collections import Counter
    print(f"  {len(raus):,} Fragen -> {args.aus}")
    print(f"  {verworfen:,} Absätze wegen Sperre verworfen\n")
    for g, n in Counter(e["gruppe"] for e in raus).most_common():
        print(f"    {g:<12}{n:>6,}")


if __name__ == "__main__":
    main()
