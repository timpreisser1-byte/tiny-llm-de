"""Kuratierter Datensatz: nur muttersprachlich deutsche, saubere Dialoge.

Der Gegenentwurf zu "viel hilft viel". Bisher lernte das Modell aus 273.000
Dialogen, von denen die allermeisten aus dem Englischen uebersetzt sind. Man
sieht es den Antworten an: "Als KI-Sprachmodell habe ich keine Gefuehle" ist
woertlich uebersetztes ChatGPT-Deutsch, "Ich bin ein grosses Sprachmodell"
ebenso. Ein Modell mit 10 Millionen Parametern hat keine Kapazitaet, solche
Floskeln zu ueberwinden - es lernt genau das, was am haeufigsten dasteht.

Deshalb hier: wenige, dafuer gute Daten.

  behalten   german-conv      auf Deutsch erzeugte Alltagsgespraeche
             oasst2_trees     von Deutschen geschrieben und bewertet
             seed/ratgeber/   selbst geschrieben, auf das Projekt zugeschnitten
             smalltalk
  raus       alles aus dem Englischen uebersetzte (dolly, alpaca, dolphin,
             ultra-chat, airoboros, evol-instruct, sharegpt, openschnabeltier)

    python hochwertig.py
"""

import argparse
import json
import os
import random
import re
from collections import Counter

import dialogdaten

# Muttersprachlich deutsche Quellen -> (Datei, Gewicht, max. Frageleange)
# Bei Kontextdaten steht der ganze Wissenstext in der Nutzerzeile, die darf
# dort also viel laenger sein als bei normalen Fragen.
QUELLEN = [
    # german-conv liefert den natuerlichen Plauderton, ist aber so gross, dass
    # es alles andere erdrueckt: nach dem ersten Lauf antwortete das Modell
    # auch auf "Ich kann nicht einschlafen" mit einer Alltagsanekdote. Deshalb
    # gedeckelt - Vielfalt der Quellen schlaegt Menge einer Quelle.
    ("german-conv.parquet", 1, 300),
    ("oasst2_trees.jsonl.gz", 4, 300),
    ("smalltalk_de.jsonl", 60, 300),
    ("seed_de.jsonl", 60, 300),
    ("ratgeber_de.jsonl", 60, 300),
    ("elektronik_de.jsonl", 60, 300),
    # DiscoResearch: auf Deutsch entstandene RAG-Daten, inklusive Faellen,
    # in denen die Antwort NICHT im Text steht
    ("germanrag_chat.jsonl", 4, 900),
]

# Uebersetzungsspuren und Assistenten-Floskeln. Genau diese Saetze hat das
# 5M-Modell auswendig gelernt und gibt sie bei jeder Gelegenheit aus.
FLOSKELN = re.compile(
    r"als (ki|ein ki|eine ki|künstliche)|ki-sprachmodell|sprachmodell|"
    r"open ?assistant|als sprach|ich bin ein (großes|grosses) |"
    r"ich habe keine (gefühle|emotionen|persönlichen)|"
    r"es tut mir leid, aber ich (bin|kann)|"
    r"basierend auf (den|meinen) daten|meine trainingsdaten", re.I)

# Reste aus dem Englischen, die beim Uebersetzen stehenblieben
ENGLISCH = re.compile(r"\b(the|and|with|from|this|that|your|which|would|"
                      r"about|there|their|please|sorry|however)\b", re.I)

# Formatierung, die in gesprochener Sprache nichts zu suchen hat
FORMAT = re.compile(r"https?://|\*\*|\[.*?\]\(|^\s*[-*]\s|\|\s*-{3,}")


def taugt(msgs) -> bool:
    """Strenge Pruefung. Im Zweifel wegwerfen - Menge ist hier nicht das Ziel."""
    for m in msgs:
        t = m["content"]
        if FLOSKELN.search(t) or FORMAT.search(t):
            return False
        if len(ENGLISCH.findall(t)) >= 2:
            return False
    frage, antwort = msgs[0]["content"], msgs[-1]["content"]
    if not (3 <= len(frage) <= 900) or not (15 <= len(antwort) <= 500):
        return False
    if not antwort[-1] in ".!?":          # abgeschnittene Antworten raus
        return False
    if antwort.count("  ") > 2:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/sft")
    ap.add_argument("--out", default="data/sft/hochwertig.jsonl")
    ap.add_argument("--kontext", default="data/sft/kontext_qa.jsonl",
                    help="Wikipedia-Kontextfragen mitnehmen (leer lassen zum Weglassen)")
    ap.add_argument("--kontext-gewicht", type=int, default=2)
    ap.add_argument("--max-plauder", type=int, default=12000,
                    help="Obergrenze fuer german-conv, damit es nicht dominiert")
    args = ap.parse_args()
    random.seed(0)

    alle, grund = [], Counter()
    print("Muttersprachlich deutsche Quellen:\n")
    for datei, gewicht, max_frage in QUELLEN:
        pfad = os.path.join(args.dir, datei)
        if not os.path.exists(pfad):
            print(f"  {datei:26} fehlt")
            continue
        grenze = args.max_plauder if datei.startswith("german-conv") else 10**9
        roh = gut = 0
        for msgs in dialogdaten.lade(pfad, max_frage=max_frage, max_antwort=500):
            roh += 1
            if gut >= grenze:
                break
            if taugt(msgs):
                alle.append((msgs, gewicht))
                gut += 1
            else:
                grund[datei] += 1
        print(f"  {datei:26} {gut:>6,} von {roh:>6,} behalten  x{gewicht}"
              f"  = {gut*gewicht:>7,}")

    if args.kontext and os.path.exists(args.kontext):
        n = 0
        # Der Kontext steht in der Nutzerzeile, die ist hier absichtlich lang
        for msgs in dialogdaten.lade(args.kontext, max_frage=900, max_antwort=500):
            # Kontextfragen: die Antwort steht im mitgelieferten deutschen
            # Wikipedia-Text, das Modell muss sie nur wiedergeben. Uebersetzt
            # ist hier hoechstens die Frage - das faellt kaum ins Gewicht.
            alle.append((msgs, args.kontext_gewicht))
            n += 1
        print(f"  {'kontext_qa.jsonl':26} {n:>6,} Kontextfragen        "
              f"x{args.kontext_gewicht}  = {n*args.kontext_gewicht:>7,}")

    # Erst entdoppeln, dann gewichten. Andersherum loescht die Duplikatpruefung
    # genau die Wiederholungen wieder weg, die das Gewicht ausmachen - dann
    # haben 50 eigene Beispiele mit Gewicht 60 am Ende wieder Gewicht 1.
    gesehen, einmalig = set(), []
    for msgs, gewicht in alle:
        schluessel = msgs[0]["content"][:70].lower()
        if schluessel in gesehen:
            continue
        gesehen.add(schluessel)
        einmalig.append((msgs, gewicht))

    sauber = []
    for msgs, gewicht in einmalig:
        sauber.extend([msgs] * gewicht)
    random.shuffle(sauber)

    with open(args.out, "w", encoding="utf-8") as f:
        for msgs in sauber:
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")

    zeichen = sum(len(m["content"]) for d in sauber for m in d)
    print(f"\n{len(sauber):,} Dialoge -> {args.out}")
    print(f"  {zeichen/1e6:.1f} Mio. Zeichen, rund {zeichen/3.3/1e6:.1f} Mio. Token")
    print(f"  verworfen wegen Floskeln/Format/Englisch: "
          f"{sum(grund.values()):,}")
    if sauber:
        print("\nBeispiel:")
        for m in sauber[0][:2]:
            print(f"  {m['role']:9}: {m['content'][:100]}")


if __name__ == "__main__":
    main()
