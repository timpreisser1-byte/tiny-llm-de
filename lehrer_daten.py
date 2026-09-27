"""Lehrerdaten erzeugen: Saetze durch den Piper-Lehrer laufen lassen.

Nach sanoTTS (arXiv 2608.21378) lernt das winzige Geraetemodell nicht aus
Aufnahmen, sondern aus einem grossen Lehrer. Gebraucht werden je Satz drei
Dinge, und alle drei fallen hier an:

    IDs      die Phonemfolge, wie der Lehrer sie sieht
    Dauern   wie viele Rahmen jedes Phonem bekommt - direkt aus dem
             Dauerpfad des Lehrers, nicht aus dem Ton zurueckgerechnet
    Ton      22,05 kHz, das Ziel fuer den Dekodierer

Die Dauern sind der Grund, warum der Graph vorher umgebaut wurde: der
Ausgang von /Ceil wird zusaetzlich nach aussen gefuehrt. Ohne ihn muesste
man den Ton nachtraeglich ausrichten, mit allen Fehlern, die das mit sich
bringt - dieselbe Falle wie bei der Bausteinsynthese.

    python lehrer_daten.py --saetze 14000
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np


def balken(i, n, t0, text=""):
    breite = 26
    voll = int(breite * i / max(n, 1))
    rest = (time.time() - t0) / max(i, 1) * (n - i)
    sys.stderr.write(f"\r  [{'█'*voll}{'░'*(breite-voll)}] {i:>6,}/{n:,}"
                     f"  noch {rest/60:5.1f} min  {text:<18}")
    sys.stderr.flush()


def saetze_aus(pfad, n, min_z=25, max_z=140):
    """Kurze, saubere Saetze - lange kosten Rechenzeit ohne mehr zu lehren."""
    raus = []
    with open(pfad, encoding="utf-8") as f:
        for zeile in f:
            if len(raus) >= n:
                break
            absatz = zeile.split(": ", 1)[-1].strip()
            for s in re.split(r"(?<=[.!?])\s+", absatz):
                s = s.strip()
                if not (min_z <= len(s) <= max_z):
                    continue
                if not re.match(r"^[A-ZÄÖÜ]", s) or s.count(",") > 2:
                    continue
                if re.search(r"[(){}\[\]<>|/\\@#*_=+~^]|\d{5,}", s):
                    continue
                raus.append(s)
                if len(raus) >= n:
                    break
    return raus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quelle", default="wiki/chunks.txt")
    ap.add_argument("--out", default="data_tts")
    ap.add_argument("--saetze", type=int, default=14000)
    ap.add_argument("--modell", default="lehrer/thorsten_mit_dauer.onnx")
    args = ap.parse_args()

    import onnxruntime as ort
    from lehrer_stimme import Lehrer

    os.makedirs(args.out, exist_ok=True)
    L = Lehrer()
    sitzung = ort.InferenceSession(args.modell, providers=["CPUExecutionProvider"])
    skalen = np.array([0.667, 1.0, 0.8], dtype=np.float32)

    texte = saetze_aus(args.quelle, args.saetze)
    print(f"  {len(texte):,} Sätze aus {args.quelle}\n")

    ton_datei = open(os.path.join(args.out, "ton.i16"), "wb")
    eintraege, pos, t0, uebersprungen = [], 0, time.time(), 0
    for i, text in enumerate(texte, 1):
        try:
            p = L.phoneme(text)
            ids = L.ids(p)
            if not 8 <= len(ids) <= 400:
                uebersprungen += 1; continue
            a = np.array([ids], dtype=np.int64)
            aus = sitzung.run(None, {"input": a,
                                     "input_lengths": np.array([len(ids)], np.int64),
                                     "scales": skalen})
            ton = aus[0].squeeze().astype(np.float32)
            dauer = aus[1].squeeze().astype(np.int32)
            if len(ton) < 2000 or len(dauer) != len(ids):
                uebersprungen += 1; continue
            roh = np.clip(ton * 32767, -32768, 32767).astype(np.int16)
            ton_datei.write(roh.tobytes())
            eintraege.append({"text": text, "ids": ids,
                              "dauer": dauer.tolist(), "pos": pos, "n": len(roh)})
            pos += len(roh)
        except Exception:
            uebersprungen += 1
        if i % 50 == 0:
            balken(i, len(texte), t0, f"{pos/22050/60:.0f} min Ton")
    ton_datei.close()
    sys.stderr.write("\r" + " " * 80 + "\r")

    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"rate": 22050, "sprung": 256, "eintraege": eintraege}, f)
    print(f"  {len(eintraege):,} Sätze gespeichert, {uebersprungen} übersprungen")
    print(f"  {pos/22050/60:.0f} Minuten Ton, {pos*2/1e6:.0f} MB")
    print(f"  {sum(len(e['ids']) for e in eintraege):,} Phoneme insgesamt")


if __name__ == "__main__":
    main()
