"""Modelle auf EINEM festen Validierungssatz vergleichen.

Beobachtet: Die Validierung waehrend des Trainings schwankt um plus/minus 0,3,
weil sie 60 zufaellige Staepel aus einer kleinen Datei zieht. Der "beste Wert"
eines Laufs ist damit ein Bestwert aus vielen verrauschten Messungen - und je
mehr Messungen, desto niedriger faellt er aus. Ein Lauf mit 26 Validierungen
schlaegt einen mit 14 allein durch die Zahl der Ziehungen.

Hier sieht jedes Modell exakt dieselben 600 Faltungen, einmal, ohne Ziehung.
Ein Unterschied im Ergebnis ist dann ein Unterschied im Modell.

    python -m evaluation.compare_models --ckpt a.pt --ckpt b.pt
"""

import argparse
import os

import numpy as np
import torch

from lm.model import Config, TinyLM


def lade(pfad, device):
    ck = torch.load(pfad, map_location="cpu", weights_only=False)
    cfg = Config(**ck["config"])
    netz = TinyLM(cfg)
    art = ck.get("art") or ("ternaer" if ck.get("ternaer") else None)
    if art:
        from lm.ternary import beschraenke
        beschraenke(netz, art)
    netz.load_state_dict(ck["model"])
    return netz.to(device).eval(), cfg, art


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--val", default="data_8mb/val_fest.npz")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    d = np.load(args.val)
    X = torch.from_numpy(d["x"].astype(np.int64))
    Y = torch.from_numpy(d["y"].astype(np.int64))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n  %d feste Faltungen a %d Token, auf %s\n" % (len(X), X.shape[1], dev))
    kopf = "  %-34s %9s %8s %9s %8s" % ("Modell", "val loss", "ppl", "Parameter", "Packung")
    print(kopf)
    print("  " + "-" * (len(kopf) - 2))

    for pfad in args.ckpt:
        if not os.path.exists(pfad):
            print("  %-34s %9s" % (pfad[-33:], "fehlt"))
            continue
        netz, cfg, art = lade(pfad, dev)
        n = sum(x.numel() for x in netz.parameters())
        summe = 0.0
        with torch.no_grad():
            for i in range(0, len(X), args.batch):
                x, y = X[i:i + args.batch].to(dev), Y[i:i + args.batch].to(dev)
                summe += netz(x, y)[1].item() * len(x)
        v = summe / len(X)
        print("  %-34s %9.4f %8.1f %8.2fM %8s"
              % (pfad[-33:], v, float(np.exp(v)), n / 1e6, art or "float"))


if __name__ == "__main__":
    main()
