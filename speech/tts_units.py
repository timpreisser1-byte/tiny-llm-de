"""Wie viele Lautbausteine braucht deutsche Sprachausgabe wirklich?

Der Weg, den der Sprachplan fuer den ESP32-P4 vorsieht: aus Thorstens
Aufnahmen die Lautuebergaenge herausschneiden und beim Sprechen wieder
zusammensetzen. Kein Netz zur Laufzeit, nur Nachschlagen und Aneinanderfuegen.

Vorher muss man wissen, wie viele Bausteine das sind. Theoretisch gibt es bei
n Lauten n² Uebergaenge - bei rund 45 deutschen Lauten waeren das 2.025. In
echtem Text kommen aber laengst nicht alle vor: "pf" gibt es, "fp" nicht.

Dieses Skript misst es. Es jagt echten deutschen Text durch die Lautumschrift
und zaehlt, welche Uebergaenge tatsaechlich auftreten und wie viele man
braucht, um 95, 99 und 99,9 Prozent aller Uebergaenge abzudecken. Daraus
folgt direkt der Platzbedarf auf der Karte.

    python -m speech.tts_units --saetze 200000
"""

import argparse
import re
import sys
import time
from collections import Counter

from speech.tts_phonemes import wort_zu_lauten

# Laute sind teils zweizeichig (a:, aI, OY, tS). Zum Zaehlen muss die
# Lautkette wieder in einzelne Laute zerlegt werden.
ZWEI = ("a:", "e:", "i:", "o:", "u:", "y:", "E:", "2:", "aI", "aU", "OY",
        "tS", "dZ", "ts", "pf", "ks", "kv", "@n", "@l")


def zerlege(kette):
    laute, i = [], 0
    while i < len(kette):
        if kette[i:i + 2] in ZWEI:
            laute.append(kette[i:i + 2]); i += 2
        else:
            laute.append(kette[i]); i += 1
    return laute


def balken(i, n, t0):
    breite = 28
    voll = int(breite * i / max(n, 1))
    rest = (time.time() - t0) / max(i, 1) * (n - i)
    sys.stderr.write(f"\r  [{'█'*voll}{'░'*(breite-voll)}] {i:>7,}/{n:,}"
                     f"  noch {rest/60:4.1f} min")
    sys.stderr.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="wiki/chunks.txt")
    ap.add_argument("--saetze", type=int, default=200000)
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--ms", type=int, default=110,
                    help="mittlere Laenge eines Bausteins in Millisekunden")
    args = ap.parse_args()

    paare, einzel, woerter = Counter(), Counter(), Counter()
    t0, n = time.time(), 0
    with open(args.chunks, encoding="utf-8") as f:
        for zeile in f:
            if n >= args.saetze:
                break
            for wort in re.findall(r"[A-Za-zÄÖÜäöüß]{2,}", zeile):
                w = wort.lower()
                if w not in woerter:
                    woerter[w] = 0
                woerter[w] += 1
            n += 1
            if n % 5000 == 0:
                balken(n, args.saetze, t0)
    sys.stderr.write("\r" + " " * 60 + "\r")

    # Lautumschrift nur einmal je Wortform, danach mit der Haeufigkeit gewichten
    print(f"  {len(woerter):,} verschiedene Wortformen aus {n:,} Absätzen")
    t0 = time.time()
    for k, (wort, anzahl) in enumerate(woerter.items(), 1):
        laute = zerlege(wort_zu_lauten(wort))
        for l in laute:
            einzel[l] += anzahl
        # Wortgrenze zaehlt mit: der Uebergang von Stille in den ersten Laut
        kette = ["#"] + laute + ["#"]
        for a, b in zip(kette, kette[1:]):
            paare[(a, b)] += anzahl
        if k % 20000 == 0:
            balken(k, len(woerter), t0)
    sys.stderr.write("\r" + " " * 60 + "\r")

    gesamt = sum(paare.values())
    sortiert = paare.most_common()
    print(f"  {len(einzel)} verschiedene Laute, "
          f"{len(paare):,} verschiedene Übergänge\n")

    print(f"  {'Abdeckung':12}{'Bausteine':>11}{'auf SD':>11}{'im Flash':>11}")
    print("  " + "-" * 46)
    bytes_je = args.rate * args.ms / 1000 * 2          # 16 Bit
    summe = 0
    ziele = {0.90: None, 0.95: None, 0.99: None, 0.999: None, 1.0: None}
    for i, (_, k) in enumerate(sortiert, 1):
        summe += k
        for z in list(ziele):
            if ziele[z] is None and summe / gesamt >= z:
                ziele[z] = i
    for z, i in ziele.items():
        if i is None:
            i = len(sortiert)
        mb = i * bytes_je / 1e6
        print(f"  {z*100:9.1f} %{i:>11,}{mb:>9.1f} MB{mb/4:>9.1f} MB")
    print(f"\n  {args.ms} ms je Baustein, {args.rate/1000:.0f} kHz, 16 Bit."
          f"  Flash-Spalte: 4-Bit-ADPCM.")

    print(f"\n  Die zwölf häufigsten Übergänge")
    for (a, b), k in sortiert[:12]:
        print(f"    {a:>3} -> {b:<3}  {k/gesamt*100:5.2f} %")
    fehlend = len([1 for p, k in sortiert if k < gesamt * 1e-7])
    print(f"\n  {fehlend:,} Übergänge kommen seltener als einmal in "
          f"zehn Millionen vor - die kann ein Notbehelf ersetzen.")


if __name__ == "__main__":
    main()
