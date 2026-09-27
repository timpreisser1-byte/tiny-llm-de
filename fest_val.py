#!/usr/bin/env python3
"""fest_val.py — feste Validierungsfalten erzeugen.

Die Validierung waehrend des Trainings zieht zufaellige Stapel und schwankt
dadurch um plus/minus 0,3. Wer oefter validiert, bekommt allein dadurch einen
besseren "Bestwert". Fuer den Vergleich zweier Modelle taugt das nicht.

Hier wird einmal ein fester Satz Falten aus val.bin geschnitten, den jedes
Modell exakt gleich sieht. Die Faltlaenge richtet sich nach dem KLEINSTEN zu
vergleichenden Modell - ein Modell mit laengerem Kontext darf seinen Vorteil
hier nicht ausspielen, sonst vergliche man Kontextlaenge statt Guete.

    python fest_val.py --laenge 384 --anzahl 600
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="data_8mb/val.bin")
    ap.add_argument("--aus", default="data_8mb/val_fest.npz")
    ap.add_argument("--laenge", type=int, default=384)
    ap.add_argument("--anzahl", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    daten = np.memmap(args.val, dtype=np.uint16, mode="r")
    moeglich = (len(daten) - 1) // args.laenge
    n = min(args.anzahl, moeglich)
    if n < args.anzahl:
        print(f"  nur {n} Falten moeglich statt {args.anzahl}")

    # gleichmaessig ueber die Datei verteilt statt zufaellig - deckt mehr ab
    starts = np.linspace(0, len(daten) - args.laenge - 1, n).astype(np.int64)
    x = np.stack([daten[s:s + args.laenge] for s in starts])
    y = np.stack([daten[s + 1:s + args.laenge + 1] for s in starts])
    np.savez_compressed(args.aus, x=x.astype(np.uint16), y=y.astype(np.uint16))
    abdeckung = n * args.laenge / len(daten) * 100
    print(f"  {n} Falten a {args.laenge} Token -> {args.aus}")
    print(f"  {abdeckung:.1f} % von val.bin abgedeckt ({len(daten):,} Token)")


if __name__ == "__main__":
    main()
