"""Der Erkenner benotet die Stimme - eine Zahl statt eines Höreindrucks.

Bisher wurde die Sprachausgabe nach Gehör bewertet ("klingt 5 Prozent wie das
Original"). Das ist weder wiederholbar noch vergleichbar. Das saanoTTS-Papier
(arXiv 2608.21378) macht es anders und gibt für seine eingebettete Stufe
14,8 % normalisierte Wortfehlerrate an: Man lässt einen Erkenner auf die
erzeugte Sprache los und misst, wie viel davon ankommt.

Wir haben einen Erkenner. Der Aufbau hier ist streng kontrolliert - dieselben
Sätze laufen in drei Fassungen durch dasselbe Netz:

    echt      der Mensch aus dem Testteil          Bezugspunkt nach unten
    Lehrer    Piper, 15,7M Parameter               Bezugspunkt nach oben
    Schüler   unsere 0,78 MB                       das Prüfstück

Nur die Differenz zählt. Die absolute Fehlerrate des Erkenners (rund 53 %)
kürzt sich dabei weitgehend heraus - sie trifft alle drei Fassungen gleich.

    python -m speech.tts_grade --anzahl 60
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def balken(i, n, t0, text=""):
    b = 22
    v = int(b * i / max(n, 1))
    rest = (time.time() - t0) / max(i, 1) * (n - i)
    sys.stderr.write(f"\r  [{'█'*v}{'░'*(b-v)}] {i:>4}/{n}  "
                     f"noch {rest/60:4.1f} min  {text:<30}")
    sys.stderr.flush()


def auf_16k(ton, rate, ziel):
    if rate == ziel:
        return np.ascontiguousarray(ton, dtype=np.float32)
    m = int(len(ton) * ziel / rate)
    return np.interp(np.linspace(0, len(ton) - 1, m),
                     np.arange(len(ton)), ton).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anzahl", type=int, default=60)
    ap.add_argument("--erkenner", default="ckpt_asr/ctc_gross_tern12.pt")
    ap.add_argument("--tts", default="ckpt_tts")
    ap.add_argument("--dir", default="data/asr")
    ap.add_argument("--portion", default="test")
    ap.add_argument("--min-woerter", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from speech.asr_data import N_MEL, RATE, merkmale
    from speech.asr_model import gierig, lade_modell, wortfehlerrate
    from speech.teacher_voice import Lehrer
    from training.pretrain import pick_device

    device = pick_device(args.device)
    netz = lade_modell(args.erkenner, device)
    L = Lehrer()

    hat_schueler = os.path.exists(os.path.join(args.tts, "dekoder.pt"))
    schueler = None
    if hat_schueler:
        from speech.tts_speaker import lade, sprich
        from speech.tts_model import RATE as TTS_RATE
        schueler = (lade(args.tts, device), sprich, TTS_RATE)
    else:
        print(f"  ! {args.tts}/dekoder.pt fehlt - Schüler wird übersprungen\n")

    eintraege = json.load(open(f"{args.dir}/{args.portion}.json",
                               encoding="utf-8"))["aufnahmen"]
    eintraege = [e for e in eintraege
                 if len(e["text"].split()) >= args.min_woerter][:args.anzahl]
    roh = np.memmap(f"{args.dir}/{args.portion}.f16", dtype=np.float16, mode="r")
    print(f"  {len(eintraege)} Sätze, Erkenner {os.path.basename(args.erkenner)}\n")

    @torch.no_grad()
    def hoere(mel):
        x = torch.from_numpy(np.ascontiguousarray(mel, dtype=np.float32))
        return gierig(netz(x.T[None].to(device))[:, 0])

    fassungen = ["echt", "lehrer"] + (["schueler"] if schueler else [])
    fehler = {f: [] for f in fassungen}
    beispiele = []
    t0 = time.time()

    for i, e in enumerate(eintraege, 1):
        soll = e["text"]
        zeile = {}
        # echt: die Merkmale liegen schon fertig in der Datei
        mel = np.array(roh[e["pos"]:e["pos"] + e["rahmen"] * N_MEL]).reshape(-1, N_MEL)
        zeile["echt"] = hoere(mel)
        # Lehrer
        ton = L.sprich(soll)[0]
        zeile["lehrer"] = hoere(merkmale(auf_16k(ton, L.rate, RATE)))
        # Schüler
        if schueler:
            netze, sprich, tr = schueler
            ton, _ = sprich(soll, netze, L, device)
            zeile["schueler"] = hoere(merkmale(auf_16k(ton, tr, RATE)))
        for f in fassungen:
            fehler[f].append(wortfehlerrate(soll, zeile[f]))
        if len(beispiele) < 3:
            beispiele.append((soll, zeile))
        if i % 2 == 0 or i == len(eintraege):
            balken(i, len(eintraege), t0,
                   " ".join(f"{f[:4]} {np.mean(fehler[f])*100:.0f}%"
                            for f in fassungen))
    sys.stderr.write("\r" + " " * 90 + "\r")

    print("                     Wortfehlerrate   gegen echt")
    basis = np.mean(fehler["echt"])
    for f in fassungen:
        w = np.mean(fehler[f])
        d = "" if f == "echt" else f"{(w-basis)*100:+8.1f} Punkte"
        print(f"  {f:<16}{w*100:>10.1f} %{d:>20}")
    print()
    for soll, z in beispiele:
        print(f"  soll     : {soll[:76]}")
        for f in fassungen:
            print(f"  {f:<9}: {z[f][:76]}")
        print()

    with open("tts_note.json", "w", encoding="utf-8") as f:
        json.dump({"n": len(eintraege), "erkenner": args.erkenner,
                   "wer": {k: float(np.mean(v)) for k, v in fehler.items()}}, f,
                  indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
