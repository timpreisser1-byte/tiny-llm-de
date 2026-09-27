"""Synthetische Sprachdaten fuer den Erkenner - aus der eigenen Sprachausgabe.

Der Kreis schliesst sich: Die Stimme, die wir fuer das Geraet bauen, kann den
Erkenner trainieren. Zwei Arbeiten belegen das:

    "Ein TTS-Modell in der Selbstverfeinerungsschleife senkt die benoetigte
     Menge echter Sprachdaten um das Zehnfache."          arXiv 2506.11130
    "Bis zu 25,5 % relative Verbesserung der Wortfehlerrate."
                                                          arXiv 2410.16726

Entscheidend ist die Sprechervielfalt: synthetische Daten von EINEM Sprecher
uebertragen sich schlecht auf echte Stimmen. Deshalb laufen die Saetze hier
durch sechs verschiedene Piper-Stimmen, maennlich und weiblich.

Zwei Warnungen aus denselben Arbeiten sind eingebaut:

    mischen, nicht ersetzen   die synthetischen Merkmale kommen als eigene
                              Portion dazu, die echten bleiben
    Tempo streuen             sonst lernt der Erkenner eine Sprechgeschwindigkeit
                              statt Sprache

    python asr_synth.py --saetze 12000
"""

import argparse
import json
import os
import sys
import time

import numpy as np

STIMMEN = [("eva_k", "x_low"), ("karlsson", "low"), ("kerstin", "low"),
           ("pavoque", "low"), ("ramona", "low"), ("thorsten", "medium")]


def balken(i, n, t0, text=""):
    b = 24
    v = int(b * i / max(n, 1))
    rest = (time.time() - t0) / max(i, 1) * (n - i)
    sys.stderr.write(f"\r  [{'█'*v}{'░'*(b-v)}] {i:>6,}/{n:,}  "
                     f"noch {rest/60:5.1f} min  {text:<22}")
    sys.stderr.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quelle", default="wiki/chunks.txt")
    ap.add_argument("--dir", default="data/asr")
    ap.add_argument("--portion", default="synth")
    ap.add_argument("--saetze", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import onnxruntime as ort
    from asr_daten import MAX_SEKUNDEN, N_MEL, RATE, ZU_INDEX, merkmale
    from lehrer_daten import saetze_aus
    from lehrer_stimme import Lehrer

    rng = np.random.default_rng(args.seed)
    texte = saetze_aus(args.quelle, args.saetze, min_z=30, max_z=120)
    print(f"  {len(texte):,} Sätze, {len(STIMMEN)} Stimmen\n")

    sitzungen = []
    for name, güte in STIMMEN:
        p = f"lehrer/stimmen/de_DE-{name}-{güte}.onnx"
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        c = json.load(open(p + ".json", encoding="utf-8"))
        sitzungen.append((name, s, c["phoneme_id_map"], c["audio"]["sample_rate"]))

    L = Lehrer()                       # nur fuer die Phonemzerlegung
    os.makedirs(args.dir, exist_ok=True)
    f16 = open(os.path.join(args.dir, f"{args.portion}.f16"), "wb")
    aufnahmen, pos, t0, weg = [], 0, time.time(), 0

    for i, text in enumerate(texte, 1):
        name, sitzung, karte, rate = sitzungen[i % len(sitzungen)]
        # Der Erkenner arbeitet auf Kleinschreibung ohne Satzzeichen
        ziel = "".join(z for z in text.lower() if z in ZU_INDEX)
        ziele = [ZU_INDEX[z] for z in ziel]
        if not 10 <= len(ziele) <= 300:
            weg += 1; continue
        try:
            p = L.phoneme(text)
            ids = [karte["^"][0]]
            for z in p:
                if z in karte:
                    ids.extend(karte[z]); ids.extend(karte["_"])
            ids.append(karte["$"][0])
            if not 8 <= len(ids) <= 400:
                weg += 1; continue
            # Tempo streuen, sonst lernt der Erkenner eine Geschwindigkeit
            tempo = float(rng.uniform(0.85, 1.20))
            ton = sitzung.run(None, {
                "input": np.array([ids], dtype=np.int64),
                "input_lengths": np.array([len(ids)], dtype=np.int64),
                "scales": np.array([0.667, tempo, 0.8], np.float32)})[0].squeeze()
            if rate != RATE:                     # auf 16 kHz bringen
                m = int(len(ton) * RATE / rate)
                ton = np.interp(np.linspace(0, len(ton) - 1, m),
                                np.arange(len(ton)), ton).astype(np.float32)
            if not 0.5 * RATE < len(ton) < MAX_SEKUNDEN * RATE:
                weg += 1; continue
            mel = merkmale(ton.astype(np.float32))
            if len(mel) < len(ziele):            # CTC braucht mehr Rahmen als Zeichen
                weg += 1; continue
            f16.write(mel.astype(np.float16).tobytes())
            aufnahmen.append({"pos": pos, "rahmen": len(mel),
                              "ziele": ziele, "text": ziel, "stimme": name})
            pos += len(mel) * N_MEL
        except Exception:
            weg += 1
        if i % 50 == 0:
            balken(i, len(texte), t0, f"{pos/N_MEL*320/RATE/3600:.1f} h")
    f16.close()
    sys.stderr.write("\r" + " " * 80 + "\r")

    with open(os.path.join(args.dir, f"{args.portion}.json"), "w",
              encoding="utf-8") as f:
        json.dump({"aufnahmen": aufnahmen}, f)
    stunden = sum(a["rahmen"] for a in aufnahmen) * 320 / RATE / 3600
    print(f"  {len(aufnahmen):,} Aufnahmen, {weg} übersprungen")
    print(f"  {stunden:.1f} Stunden synthetische Sprache, "
          f"{pos*2/1e6:.0f} MB Merkmale")
    from collections import Counter
    for s, n in Counter(a["stimme"] for a in aufnahmen).most_common():
        print(f"    {s:14}{n:>7,}")


if __name__ == "__main__":
    main()
