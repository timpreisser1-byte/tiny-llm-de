"""Mischkorpus bauen: Wikipedia + deutsche Gespraeche.

Warum: ein Modell, das nur Wikipedia gesehen hat, schreibt Lexikonartikel.
Es kennt "Der Kondensator ist ein passives Bauelement", aber nicht
"Klar, erklaer ich dir das kurz". Damit ein 5-8M-Modell wirklich reden lernt,
muessen die Dialoge schon im Pretraining vorkommen - und zwar in genau dem
Format, das spaeter beim Chat benutzt wird:

    <s>\\n<user>\\n Frage \\n<assistant>\\n Antwort </s>

Danach wieder tokenize_corpus.py laufen lassen.

    python build_mix.py --chat-anteil 0.35
"""

import argparse
import glob
import os
import random

import dialogdaten

# Datei -> wie oft der Datensatz im Mix vorkommt (Qualitaet vor Menge)
GEWICHTE = {
    "ratgeber_de": 30,         # eigene Psychologie- und Alltagstipps
    "kontext_qa": 3,           # mit gegebenem Wissen antworten statt erfinden
    "oasst2_trees": 6,         # muttersprachlich deutsch, von Menschen bewertet
    "german-conv": 3,          # echte Alltagsgespraeche - genau das, was fehlt
    "oasst-chains_de": 3,      # mehrstufige Dialoge mit Rueckfragen
    "seed_de": 40,             # eigene Beispiele, stark gewichtet
    "dolly_de": 2,             # kurz und sachlich, passt gut zu kleinen Modellen
    "oasst_de": 2,
    "sharegpt_de": 2,
    "ultra-chat_de": 1,        # grosse Menge, eher Sachtexte
    "dolphin_de": 1,
    "evol-instruct_de": 1,
    "airoboros-3.0_de": 1,
    "openschnabeltier_de": 1,
    "alpaca_de": 1,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", default="data/corpus.txt")
    ap.add_argument("--sft-dir", default="data/sft")
    ap.add_argument("--out", default="data/corpus_mix.txt")
    ap.add_argument("--chat-anteil", type=float, default=0.30,
                    help="Zielanteil der Gespraeche am Gesamttext (0-1)")
    ap.add_argument("--max-wiederholungen", type=int, default=8,
                    help="wie oft die Dialoge hoechstens wiederholt werden duerfen")
    ap.add_argument("--max-antwort", type=int, default=1200,
                    help="Zeichen; im Pretraining darf mehr durch als beim Finetuning")
    args = ap.parse_args()

    random.seed(0)
    print("Lese Dialogdaten ...")
    chats = []
    dateien = sum((glob.glob(os.path.join(args.sft_dir, f"*.{e}"))
                   for e in ("json", "jsonl", "jsonl.gz", "parquet")), [])
    for datei in sorted(dateien):
        name = os.path.basename(datei).split(".")[0]   # auch bei .jsonl.gz
        if name == "train":                 # Ausgabe von get_sft_data.py
            continue
        gewicht = GEWICHTE.get(name, 1)
        n = 0
        for msgs in dialogdaten.lade(datei, max_antwort=args.max_antwort):
            chats.extend([dialogdaten.als_text(msgs)] * gewicht)
            n += gewicht
        print(f"  {name:22} {n:>9,} Dialoge (Gewicht {gewicht})", flush=True)

    if not chats:
        raise SystemExit("Keine Dialoge gefunden - erst get_sft_data.py laufen lassen.")

    random.shuffle(chats)
    chat_zeichen = sum(len(t) for t in chats)
    wiki_limit = os.path.getsize(args.wiki) if args.chat_anteil < 0.999 else 0
    print(f"\nGespraeche: {len(chats):,} Stueck, {chat_zeichen/1e6:.0f} Mio. Zeichen")
    print(f"Wikipedia:  {wiki_limit/1e6:.0f} Mio. Zeichen")

    if wiki_limit == 0:
        # Reiner Gespraechskorpus. Bewusste Entscheidung: das Modell lernt
        # Dialogform und Alltagssprache, dafuer weniger Wortschatz und weniger
        # Faktenwissen. Fuer den ESP32-Assistenten passt das - Wissen soll
        # ohnehin von der SD-Karte kommen, nicht aus den Gewichten.
        print("Modus: NUR Gespraeche, kein Wikipedia")

    # Wikipedia komplett behalten und stattdessen die Dialoge wiederholen.
    # Wikipedia wegzuwerfen waere teurer: Sprache lernt das Modell dort, das
    # Gespraechsformat ist dagegen kurz und vertraegt Wiederholung.
    if wiki_limit:
        noetig = wiki_limit * args.chat_anteil / max(1 - args.chat_anteil, 1e-6)
        wdh = max(1, min(args.max_wiederholungen, round(noetig / chat_zeichen)))
        chats = chats * wdh
        chat_zeichen *= wdh
        print(f"Ziel {args.chat_anteil:.0%} Gespraech -> Dialoge {wdh}x wiederholt "
              f"= {chat_zeichen/(chat_zeichen+wiki_limit):.0%} tatsaechlich")
        random.shuffle(chats)

    print("\nSchreibe Korpus ...")
    i_chat = 0
    with open(args.out, "w", encoding="utf-8") as out:
        if wiki_limit:
            # Beides ineinander mischen, damit kein Block nur aus einer Quelle besteht
            bloecke = max(1, int(wiki_limit / 2_000_000))
            chat_pro_block = max(1, len(chats) // bloecke)
            geschrieben = 0
            with open(args.wiki, encoding="utf-8") as wiki:
                puffer = []
                for zeile in wiki:
                    if geschrieben >= wiki_limit:
                        break
                    puffer.append(zeile)
                    geschrieben += len(zeile)
                    if len(puffer) > 20000:
                        out.write("".join(puffer)); puffer = []
                        ende = min(i_chat + chat_pro_block, len(chats))
                        for t in chats[i_chat:ende]:
                            out.write(t + "\n\n")
                        i_chat = ende
                        print(f"  {geschrieben/1e6:6.0f} Mio. Zeichen Wikipedia | "
                              f"{i_chat:,} Dialoge", flush=True)
                out.write("".join(puffer))
        for t in chats[i_chat:]:
            out.write(t + "\n\n")

    gesamt = os.path.getsize(args.out)
    print(f"\nFertig: {args.out} ({gesamt/1e6:.0f} MB)")
    print(f"  Anteil Gespraech: {chat_zeichen/gesamt:.0%}")
    print(f"  grob ~{gesamt/3.3/1e6:.0f} Mio. Token")
    print(f"\nWeiter mit: python tokenize_corpus.py --input {args.out}")


if __name__ == "__main__":
    main()
