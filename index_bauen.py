"""index_bauen.py — Suchindex nach Bekanntheit der Artikel statt nach Dateireihenfolge.

Gemessen am 25.09.2026 (Suche@3, strenger Bewerter mit Wortgrenze):

    Index                                   eigene195  MKQA-dev  Mintaka-dev  Groesse
    wiki/  (erste 500k Artikel der Datei)      52,3       14,6       19,2      0,2 GB
    140k bekannteste x 8 Absaetze              55,4       19,6       25,2      0,5 GB
    gestuft 140k x 8 + bis 1 Mio x 2           51,8       23,0       28,6      1,4 GB
    alle 2,85 Mio x 12 (wiki_voll/)            50,3       23,0       29,6      8,1 GB

In der vollen Kette (Regeln + Fakten) hebt der gestufte Index MKQA-dev von
5,0 auf 8,2 % und Mintaka-dev von 8,0 auf 10,8 %. Die eigenen 195 Fragen
merkten davon nichts - sie waren unbemerkt auf den alten Index zugeschnitten.

Warum gestuft: Fragen folgen der Bekanntheit (Zipf). Bekannte Artikel werden
oft und im Detail gefragt -> viele Absaetze; seltene nur nach dem Kern ->
der Leitabsatz reicht. Die Bekanntheit ist die Zahl der Wikipedia-
Sprachversionen (Wikidata-Sitelinks, fakten/bekanntheit.csv).

SD-Kosten je Frage (sd_kosten-Messung): Postinglisten Median 164 KB statt
23 KB, auf dem ESP32-S3 per SPI ~0,1 s mehr. Den Ausschlag gibt die
Nachsortierung (PAAR_TIEFE in wissen_index.py), nicht die Indexgroesse.

    python index_bauen.py 140000:8,1000000:2 wiki_gestuft
    python index_bauen.py 140000:8,1000000:2 wiki_gestuft --dazu wiki/chunks.txt

Quelle ist wiki_voll/chunks.txt (alle Artikel, je bis zu 12 Absaetze).
"""
import argparse
import csv
import json
import os
import time
from array import array
from collections import defaultdict

import numpy as np

from wissen_index import zerlege


def lies_bekanntheit(pfad):
    """Titel -> Zahl der Sitelinks."""
    bek = {}
    with open(pfad, encoding="utf-8") as f:
        for z in csv.reader(f):
            if len(z) == 2 and z[1].isdigit():
                bek[z[0]] = int(z[1])
    return bek


def titel_der_zeile(z, bekannt):
    """Laengstes Praefix vor ": ", das ein bekannter Titel ist (Titel koennen
    selbst ": " enthalten)."""
    titel, start = None, 0
    for _ in range(4):
        k = z.find(": ", start)
        if k < 0:
            break
        if z[:k] in bekannt:
            titel = z[:k]
        start = k + 2
    return titel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stufen", help='z. B. "140000:8,1000000:2": bis Rang 140k je 8 Absaetze, dann bis 1 Mio je 2')
    ap.add_argument("ziel")
    ap.add_argument("--quelle", default="wiki_voll/chunks.txt")
    ap.add_argument("--bekanntheit", default="fakten/bekanntheit.csv")
    ap.add_argument("--dazu", help="chunks.txt eines alten Index: dessen Artikel kommen mit je 3 Absaetzen dazu")
    args = ap.parse_args()

    stufen = [tuple(map(int, s.split(":"))) for s in args.stufen.split(",")]

    def absaetze_fuer(rang):
        for grenze, n in stufen:
            if rang < grenze:
                return n
        return 0

    t0 = time.time()
    sitelinks = lies_bekanntheit(args.bekanntheit)
    dazu = set()
    if args.dazu:
        with open(args.dazu, encoding="utf-8") as f:
            for z in f:
                k = z.find(": ")
                if k > 0:
                    dazu.add(z[:k])
    bekannt = set(sitelinks) | dazu

    # Durchlauf 1: welche Artikel gibt es in der Quelle wirklich? Der Rang muss
    # unter ihnen gelten - die Bekanntheitsliste enthaelt auch Kategorien,
    # Vorlagen und Begriffsklaerungen (3,8 Mio Titel fuer 2,85 Mio Artikel).
    vorhanden = set()
    with open(args.quelle, encoding="utf-8") as f:
        for z in f:
            t = titel_der_zeile(z, bekannt)
            if t:
                vorhanden.add(t)
    folge = sorted(vorhanden, key=lambda t: -sitelinks.get(t, 0))
    rang = {t: r for r, t in enumerate(folge)}
    print(f"  {len(vorhanden):,} Artikel in der Quelle, {len(dazu):,} dazu  {time.time()-t0:.0f}s", flush=True)

    # Durchlauf 2: Absaetze je nach Rang uebernehmen
    gesehen = defaultdict(int)
    zeilen = []
    with open(args.quelle, encoding="utf-8") as f:
        for z in f:
            titel = titel_der_zeile(z, bekannt)
            if titel is None:
                continue
            erlaubt = max(absaetze_fuer(rang[titel]), 3 if titel in dazu else 0)
            if gesehen[titel] >= erlaubt:
                continue
            gesehen[titel] += 1
            zeilen.append((titel, z.rstrip("\n")))
    print(f"  {sum(1 for v in gesehen.values() if v):,} Artikel, {len(zeilen):,} Absaetze  {time.time()-t0:.0f}s", flush=True)

    # Gleicher Aufbau wie wissen_index.bauen: Postings je Wort, Titel getrennt
    os.makedirs(args.ziel, exist_ok=True)
    offsets = array("Q")
    postings = defaultdict(lambda: array("I"))
    titel_postings = defaultdict(lambda: array("I"))
    pos = 0
    with open(os.path.join(args.ziel, "chunks.txt"), "w", encoding="utf-8") as out:
        for n, (titel, text) in enumerate(zeilen):
            zeile = text + "\n"
            out.write(zeile)
            offsets.append(pos)
            pos += len(zeile.encode("utf-8"))
            for w in set(zerlege(text)):
                postings[w].append(n)
            for w in set(zerlege(titel)):
                titel_postings[w].append(n)
    n = len(zeilen)
    grenze = max(1000, int(n * 0.12))
    for w in [w for w, liste in postings.items() if len(liste) > grenze]:
        del postings[w]

    def packe(tab):
        vok, alle = {}, array("I")
        for w in sorted(tab):
            vok[w] = [len(alle), len(tab[w])]
            alle.extend(tab[w])
        return vok, alle

    vok, alle = packe(postings)
    tvok, talle = packe(titel_postings)
    np.frombuffer(offsets, dtype=np.uint64).tofile(os.path.join(args.ziel, "offsets.npy"))
    np.frombuffer(alle, dtype=np.uint32).tofile(os.path.join(args.ziel, "postings.npy"))
    np.frombuffer(talle, dtype=np.uint32).tofile(os.path.join(args.ziel, "titel_postings.npy"))
    with open(os.path.join(args.ziel, "vokabular.json"), "w", encoding="utf-8") as f:
        json.dump({"n_chunks": n, "woerter": vok, "titel": tvok}, f, ensure_ascii=False)
    groesse = sum(os.path.getsize(os.path.join(args.ziel, d)) for d in os.listdir(args.ziel))
    print(f"  fertig {args.ziel}: {groesse/1e6:.0f} MB  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
