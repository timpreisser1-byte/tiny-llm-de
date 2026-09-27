"""Aus Thorstens Aufnahmen die Lautbausteine schneiden.

Der letzte Schritt zur Sprachausgabe. Alles davor steht schon:

    speech/tts_phonemes.py       Text -> Laute, und welcher Buchstabe welchen Laut macht
    speech/tts_align.py  Aufnahme + bekannter Text -> Zeitmarke je Buchstabe
    speech/tts_units.py   welche 878 Uebergaenge neunundneunzig Prozent decken

Hier laufen sie zusammen. Fuer jede Aufnahme:

    1. auf 16 kHz bringen und Merkmale rechnen
    2. mit dem CTC-Erkenner gegen den bekannten Text ausrichten
    3. Buchstaben-Zeitmarken in Laut-Zeitmarken uebersetzen
    4. je Lautpaar ein Stueck schneiden - von der Mitte des einen Lauts
       bis zur Mitte des naechsten

Warum von Mitte zu Mitte: In der Mitte eines Lauts ist das Signal am
stabilsten, an den Raendern geht es in den Nachbarn ueber. Schneidet man
dort, passen die Stuecke beim Zusammensetzen aneinander - genau deshalb
nimmt man Uebergaenge und nicht einzelne Laute.

Je Uebergang wird das Stueck mit der ruhigsten Umgebung behalten: das mit
der geringsten Lautstaerkeschwankung an den Schnittkanten.

    python -m speech.tts_cut --dateien 2
"""

import argparse
import glob
import io
import os
import sys
import time

import numpy as np

from speech.asr_data import RATE, SPRUNG, merkmale
from speech.tts_align import RAHMEN_MS, VERSATZ_MS, richte_aus
from speech.tts_units import zerlege
from speech.tts_phonemes import wort_zu_lauten

WORT = None


def balken(i, n, t0, text=""):
    breite = 26
    voll = int(breite * i / max(n, 1))
    rest = (time.time() - t0) / max(i, 1) * (n - i)
    sys.stderr.write(f"\r  [{'█'*voll}{'░'*(breite-voll)}] {i:>5}/{n}"
                     f"  noch {rest/60:4.1f} min  {text:<22}")
    sys.stderr.flush()


def lautmarken(text, marken):
    """Buchstaben-Zeitmarken -> Laut-Zeitmarken.

    Benutzt dieselbe Wortzerlegung wie das Sprechen (`tts_laute.woerter`).
    Zwei getrennte Wege waren der Konstruktionsfehler, der die Synthese
    unbrauchbar machte - siehe den Kommentar dort.
    """
    from speech.tts_phonemes import woerter, wort_zu_lauten
    raus, zeiger = [], 0
    for wort in woerter(text):
        # bis zum Wortanfang in den Zeitmarken vorspulen, Zwischenzeichen
        # (Leerzeichen, Kommas) werden zur Wortgrenze
        start = text.lower().index(wort.lower(), zeiger) if wort.lower() in text.lower()[zeiger:] else zeiger
        if start > zeiger and zeiger < len(marken):
            g = marken[min(zeiger, len(marken) - 1)]
            raus.append(("#", g[1], g[2]))
        zeiger = start
        erg = wort_zu_lauten(wort, mit_spannen=True)
        kette, spannen = erg if isinstance(erg, tuple) else (erg, [])
        laute = zerlege(kette)
        if len(laute) != len(spannen):
            zeiger += len(wort)
            continue
        for (a, b), laut in zip(spannen, laute):
            teil = marken[zeiger + a: zeiger + b]
            if teil:
                raus.append((laut, teil[0][1], teil[-1][2]))
        zeiger += len(wort)
    return raus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quelle", default="audio/thorsten")
    ap.add_argument("--modell", default="ckpt_asr/ctc_gross.pt")
    ap.add_argument("--out", default="stimme")
    ap.add_argument("--dateien", type=int, default=4)
    ap.add_argument("--zeilen", type=int, default=0, help="0 = alle")
    ap.add_argument("--varianten", type=int, default=4,
                    help="wie viele Fassungen je Übergang behalten werden")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    from speech.asr_model import lade_modell
    from training.pretrain import pick_device

    device = pick_device("cpu")
    netz = lade_modell(args.modell, device)
    dateien = sorted(glob.glob(os.path.join(args.quelle, "**", "*.parquet"),
                               recursive=True))[:args.dateien]
    # Mehrere Varianten je Uebergang statt nur der besten. Gemessen springt
    # das Klangbild an den Naehten 49 Prozent staerker als bei natuerlicher
    # Sprache, und mit mehr gemischten Aufnahmen wird es schlimmer - weil ein
    # Baustein, der fuer sich ruhige Raender hat, trotzdem nicht zu seinen
    # Nachbarn im Satz passen muss. Mit mehreren zur Auswahl kann das
    # Sprechen die Kombination nehmen, deren Naehte zusammenpassen.
    from collections import defaultdict
    beste = defaultdict(list)       # (laut_a, laut_b) -> [(guete, signal), ...]
    n_ges = sum(pq.ParquetFile(d).metadata.num_rows for d in dateien)
    if args.zeilen:
        n_ges = min(n_ges, args.zeilen * len(dateien))
    t0, k = time.time(), 0

    for datei in dateien:
        for stapel in pq.ParquetFile(datei).iter_batches(batch_size=32):
            for zeile in stapel.to_pylist():
                if args.zeilen and k >= args.zeilen * len(dateien):
                    break
                k += 1
                text = zeile["text"].lower()
                try:
                    ton, sr = sf.read(io.BytesIO(zeile["audio"]["bytes"]),
                                      dtype="float32")
                except Exception:
                    continue
                if ton.ndim > 1:
                    ton = ton.mean(axis=1)
                # auf 16 kHz bringen - lineare Abtastung reicht fuer Bausteine
                if sr != RATE:
                    m = int(len(ton) * RATE / sr)
                    ton = np.interp(np.linspace(0, len(ton) - 1, m),
                                    np.arange(len(ton)), ton).astype(np.float32)
                mel = merkmale(ton)
                if len(mel) < 10:
                    continue
                with torch.no_grad():
                    lg = netz(torch.from_numpy(mel.astype(np.float32).T)
                              .unsqueeze(0))[:, 0, :].numpy()
                marken = richte_aus(lg, text)
                if not marken:
                    continue
                lm = lautmarken(text, marken)
                # Mitte jedes Lauts in Abtastwerten
                versatz = int(VERSATZ_MS / 1000 * RATE)
                mitten = [(l, int((a + e) / 2 * RAHMEN_MS / 1000 * RATE) - versatz)
                          for l, a, e in lm]
                mitten = [(l, p) for l, p in mitten if p >= 0]
                for (la, pa), (lb, pb) in zip(mitten, mitten[1:]):
                    if pb <= pa or pb - pa > RATE // 2:
                        continue
                    stueck = ton[pa:pb]
                    if len(stueck) < 80:
                        continue
                    # Guete: ruhige Schnittkanten, also wenig Energie ganz aussen
                    rand = float(abs(stueck[:40]).mean() + abs(stueck[-40:]).mean())
                    mitte = float(abs(stueck).mean()) + 1e-9
                    guete = rand / mitte
                    liste = beste[(la, lb)]
                    liste.append((guete, stueck.copy()))
                    if len(liste) > args.varianten:
                        liste.sort(key=lambda x: x[0])
                        del liste[args.varianten:]
                if k % 25 == 0:
                    balken(k, n_ges, t0, f"{len(beste)} Bausteine")
    sys.stderr.write("\r" + " " * 80 + "\r")

    os.makedirs(args.out, exist_ok=True)
    laengen = [len(sig) for k in beste.values() for _, sig in k]
    roh = sum(laengen) * 2
    print(f"  {k:,} Aufnahmen verarbeitet")
    print(f"  {len(beste):,} verschiedene Übergänge, "
          f"{sum(len(v) for v in beste.values()):,} Varianten "
          f"({sum(len(v) for v in beste.values())/max(len(beste),1):.1f} je Übergang)")
    print(f"  mittlere Länge {np.mean(laengen)/RATE*1000:.0f} ms, "
          f"gesamt {roh/1e6:.2f} MB als 16 Bit, "
          f"{roh/4/1e6:.2f} MB als 4-Bit-ADPCM")
    np.savez_compressed(os.path.join(args.out, "bausteine.npz"),
                        **{f"{a}|{b}|{i}": sig.astype(np.float32)
                           for (a, b), k in beste.items()
                           for i, (_, sig) in enumerate(k)})
    print(f"  geschrieben: {args.out}/bausteine.npz")


if __name__ == "__main__":
    main()
