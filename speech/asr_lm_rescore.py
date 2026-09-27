"""Das eigene Sprachmodell bewertet nach, was der Erkenner gehört hat.

Der CTC-Erkenner hoert lautlich richtig und schreibt orthografisch falsch:

    soll   denken sie soeben weilten meine gedanken bei ihnen
    roh    ige sie rebendwalten meine gedanken bei inen

Das Woerterbuch faengt davon wenig ab, weil "inen" und "ihnen" beide keine
Frage der Rechtschreibung sind, sondern des Zusammenhangs. Genau dafuer ist
ein Sprachmodell da - und eines liegt in diesem Projekt ohnehin herum.

Das ist der Vorteil dieses Geraets gegenueber jedem gekauften Erkenner: der
Assistent bewertet seine eigene Eingabe mit demselben Modell, mit dem er
spaeter antwortet. Kein zweites Netz, keine zweite Datei auf der SD-Karte.

Ablauf: Die Strahlsuche liefert zehn Kandidaten samt akustischer Bewertung,
das Sprachmodell bewertet jeden, beides wird verrechnet:

    gesamt = akustik + alpha * sprachmodell + beta * anzahl_woerter

Die beiden Gewichte werden auf der ersten Haelfte der Aufnahmen eingestellt
und auf der zweiten gemessen - sonst misst man sich selbst.

    python -m speech.asr_lm_rescore --anzahl 200
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from speech.asr_data import ZEICHEN, indizes_zu_text
from speech.asr_model import Sprachdaten, gierig as gierig_dekodieren, lade_modell
from speech.asr_beam import Baum, strahlsuche
from lm.model import Config, TinyLM
from training.pretrain import pick_device

GRUEN, ROT, GRAU, AUS = "\033[92m", "\033[91m", "\033[90m", "\033[0m"


def balken(i, n, t0, text=""):
    breite = 28
    voll = int(breite * i / max(n, 1))
    rest = (time.time() - t0) / max(i, 1) * (n - i)
    sys.stderr.write(f"\r  [{'█'*voll}{'░'*(breite-voll)}] {i:>4}/{n}"
                     f"  noch {rest/60:4.1f} min  {text:<22}")
    sys.stderr.flush()


def wer(soll, ist):
    """Wortfehlerrate nach Levenshtein auf Wortebene."""
    a, b = soll.split(), ist.split()
    d = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(a) + 1)
    d[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1,
                          d[i-1, j-1] + (a[i-1] != b[j-1]))
    return d[len(a), len(b)], len(a)


@torch.no_grad()
def lm_bewertung(netz, tok, device, text, block_size):
    """Gesamte log-Wahrscheinlichkeit des Satzes. Groesser ist besser."""
    if not text.strip():
        return -1e9
    ids = tok.encode(text.strip(), add_special_tokens=False).ids[:block_size]
    if len(ids) < 2:
        return -1e9
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    _, v = netz(x, y)
    return -v.item() * (len(ids) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modell", default="ckpt_asr/ctc_gross.pt")
    ap.add_argument("--llm", default="ckpt_56m/sft_v5.pt")
    ap.add_argument("--tokenizer", default="tokenizer/de_bpe.json")
    ap.add_argument("--woerterbuch", default="asr_woerter.json")
    ap.add_argument("--dir", default="data/asr")
    ap.add_argument("--daten", default="test")
    ap.add_argument("--anzahl", type=int, default=200)
    ap.add_argument("--breite", type=int, default=24)
    ap.add_argument("--nbest", type=int, default=10)
    ap.add_argument("--kandidaten", type=int, default=6)
    ap.add_argument("--wort-bonus", type=float, default=2.0)
    ap.add_argument("--lm-gewicht", type=float, default=0.8)
    ap.add_argument("--woerter", type=int, default=60000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = pick_device(args.device)

    erkenner = lade_modell(args.modell, device)

    with open(args.woerterbuch, encoding="utf-8") as f:
        wb = json.load(f)
    paare = sorted(wb["woerter"].items(), key=lambda x: -x[1])[:args.woerter]
    baum = Baum(dict(paare), sum(n for _, n in paare))

    lck = torch.load(args.llm, map_location="cpu", weights_only=False)
    cfg = Config(**lck["config"])
    sprachmodell = TinyLM(cfg)
    art = lck.get("art") or ("ternaer" if lck.get("ternaer") else None)
    if art:
        from lm.ternary import beschraenke
        beschraenke(sprachmodell, art)
    sprachmodell.load_state_dict(lck["model"])
    sprachmodell.to(device).eval()
    tok = Tokenizer.from_file(args.tokenizer)

    daten = Sprachdaten(args.dir, args.daten)
    n = min(args.anzahl, len(daten))
    print(f"\n  Erkenner {args.modell}  ·  Sprachmodell {args.llm}")
    print(f"  {n} Aufnahmen, {args.nbest} Kandidaten je Aufnahme\n")

    # ---- einmal durchlaufen, alles sammeln
    proben = []
    t0 = time.time()
    for i in range(n):
        mel, ziele, text = daten.hole(i)
        x = torch.from_numpy(mel.T).unsqueeze(0).to(device)
        with torch.no_grad():
            aus = erkenner(x)[:, 0, :]
        roh = gierig_dekodieren(aus)
        logits = aus.cpu().numpy()
        kand = strahlsuche(logits, baum, args.breite, args.kandidaten,
                           args.wort_bonus, args.lm_gewicht, nbest=args.nbest)
        mit_lm = [(t, a, lm_bewertung(sprachmodell, tok, device, t, cfg.block_size),
                   len(t.split())) for t, a in kand]
        proben.append({"soll": text, "gierig": roh, "kand": mit_lm})
        if (i + 1) % 5 == 0 or i + 1 == n:
            balken(i + 1, n, t0, "erkennen + bewerten")
    sys.stderr.write("\r" + " " * 90 + "\r")

    def rate(auswahl):
        f = g = 0
        for p, t in zip(proben, auswahl):
            a, b = wer(p["soll"], t)
            f += a; g += b
        return f / max(g, 1)

    halb = len(proben) // 2
    # ---- Gewichte auf der ersten Haelfte einstellen
    bestes, beste_rate = (0.0, 0.0), 1e9
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5):
        for beta in (0.0, 1.0, 2.0, 4.0, 6.0):
            wahl = [max(p["kand"], key=lambda k: k[1] + alpha * k[2] + beta * k[3])[0]
                    for p in proben[:halb]]
            f = g = 0
            for p, t in zip(proben[:halb], wahl):
                a, b = wer(p["soll"], t)
                f += a; g += b
            r = f / max(g, 1)
            if r < beste_rate:
                beste_rate, bestes = r, (alpha, beta)
    alpha, beta = bestes

    # ---- auf der zweiten Haelfte messen
    zweite = proben[halb:]
    def messe(fn):
        f = g = 0
        for p in zweite:
            a, b = wer(p["soll"], fn(p))
            f += a; g += b
        return f / max(g, 1) * 100

    r_gierig = messe(lambda p: p["gierig"])
    r_strahl = messe(lambda p: p["kand"][0][0])
    r_llm = messe(lambda p: max(p["kand"],
                                key=lambda k: k[1] + alpha * k[2] + beta * k[3])[0])
    # Obergrenze: der beste Kandidat, den die Liste ueberhaupt hergibt
    r_orakel = messe(lambda p: min(p["kand"], key=lambda k: wer(p["soll"], k[0])[0])[0])

    print(f"  Gewichte auf der ersten Hälfte eingestellt: "
          f"alpha {alpha}, beta {beta}")
    print(f"  Gemessen auf {len(zweite)} ungesehenen Aufnahmen\n")
    print(f"  {'Verfahren':34}{'Wortfehler':>12}{'ggü. gierig':>14}")
    print("  " + "-" * 62)
    for name, r in (("gierig", r_gierig),
                    ("Strahl + Wörterbuch", r_strahl),
                    ("Strahl + Wörterbuch + Sprachmodell", r_llm),
                    ("Orakel (beste Wahl möglich)", r_orakel)):
        d = r - r_gierig
        farbe = GRUEN if d < -1 else (ROT if d > 0.5 else "")
        print(f"  {name:34}{farbe}{r:11.1f}%{AUS}{d:+13.1f}")
    print()
    for p in zweite[:3]:
        gewaehlt = max(p["kand"], key=lambda k: k[1] + alpha * k[2] + beta * k[3])[0]
        print(f"  {GRAU}soll  : {p['soll'][:78]}{AUS}")
        print(f"  {GRAU}gierig: {p['gierig'][:78]}{AUS}")
        print(f"  mit LM: {gewaehlt[:78]}\n")


if __name__ == "__main__":
    main()
