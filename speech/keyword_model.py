"""Kommandowort-Erkennung: kleines Faltungsnetz auf Mel-Merkmalen.

Der Weg vom Mikrofon zum Kommando, so wie ihn spaeter auch der ESP32 geht:

    16 kHz Audio, 1 Sekunde
        |
    Mel-Spektrogramm  (40 Baender x 49 Zeitschritte)
        |
    kleines Faltungsnetz  (~60.000 Parameter)
        |
    11 Wahrscheinlichkeiten -> das wahrscheinlichste Kommando

Warum Mel-Merkmale und kein rohes Audio: Das Spektrogramm macht aus 16.000
Zahlen ein Bild mit 40x49 Punkten. Das Netz muss also 8-mal weniger verarbeiten
und die Merkmalsberechnung ist auf dem ESP32 in Festkomma machbar.

    python -m speech.keyword_model trainieren
    python -m speech.keyword_model testen --datei audio/licht_an/00000.wav
    python -m speech.keyword_model hoeren            # Mikrofon
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RATE, DAUER = 16000, 1.0
N_MEL, N_FFT, SPRUNG = 40, 512, 320     # 320 = 20 ms Schritt -> 49 Zeitschritte


# ------------------------------------------------------------- Merkmale

def mel_filter(n_mel=N_MEL, n_fft=N_FFT, rate=RATE):
    """Dreiecksfilter auf der Mel-Skala - die Frequenzaufloesung des Gehoers."""
    def hz_zu_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_zu_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    punkte = mel_zu_hz(np.linspace(hz_zu_mel(20), hz_zu_mel(rate / 2), n_mel + 2))
    bins = np.floor((n_fft + 1) * punkte / rate).astype(int)
    filter = np.zeros((n_mel, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mel):
        links, mitte, rechts = bins[i], bins[i + 1], bins[i + 2]
        for j in range(links, mitte):
            filter[i, j] = (j - links) / max(mitte - links, 1)
        for j in range(mitte, rechts):
            filter[i, j] = (rechts - j) / max(rechts - mitte, 1)
    return torch.from_numpy(filter)


MEL = mel_filter()
FENSTER = torch.hann_window(N_FFT)


def merkmale(audio: torch.Tensor) -> torch.Tensor:
    """Audio (B, 16000) -> Mel-Spektrogramm (B, 40, 49), logarithmiert."""
    # Fenster muss auf demselben Geraet liegen wie das Audio
    fenster = FENSTER.to(audio.device)
    spek = torch.stft(audio, N_FFT, hop_length=SPRUNG, window=fenster,
                      return_complex=True, center=False)
    leistung = spek.abs() ** 2
    mel = MEL.to(audio.device) @ leistung
    return torch.log(mel + 1e-6)


# ------------------------------------------------------------- Modell

class Erkenner(nn.Module):
    """Vier Faltungsschichten, dann Mittelwert ueber die Zeit.

    Bewusst klein und ohne Rekursion: Faltung und Mittelwert lassen sich in C
    mit ein paar Schleifen nachbauen, ein LSTM waere deutlich muehsamer.
    """

    def __init__(self, n_klassen, breite=32):
        super().__init__()
        b = breite
        self.netz = nn.Sequential(
            nn.Conv2d(1, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 20 x 24
            nn.Conv2d(b, b * 2, 3, padding=1), nn.BatchNorm2d(b * 2), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 10 x 12
            nn.Conv2d(b * 2, b * 2, 3, padding=1), nn.BatchNorm2d(b * 2), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 5 x 6
            nn.Conv2d(b * 2, b * 4, 3, padding=1), nn.BatchNorm2d(b * 4), nn.ReLU(),
        )
        self.aus = nn.Linear(b * 4, n_klassen)

    def forward(self, audio):
        x = merkmale(audio).unsqueeze(1)          # (B, 1, 40, 49)
        x = self.netz(x)
        x = x.mean(dim=(2, 3))                    # Mittelwert ueber Zeit und Frequenz
        return self.aus(x)


# ------------------------------------------------------------- Daten

def lade_daten(ordner="audio"):
    import soundfile as sf

    with open(os.path.join(ordner, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    klassen = index["klassen"]
    nummer = {k: i for i, k in enumerate(klassen)}
    n = int(RATE * DAUER)

    X = np.zeros((len(index["beispiele"]), n), dtype=np.float32)
    y = np.zeros(len(index["beispiele"]), dtype=np.int64)
    for i, b in enumerate(index["beispiele"]):
        d, _ = sf.read(b["datei"], dtype="float32")
        X[i, :min(len(d), n)] = d[:n]
        y[i] = nummer[b["klasse"]]
        if i % 1000 == 0:
            print(f"  {i:>6,} / {len(X):,} geladen", end="\r", flush=True)
    print(" " * 40, end="\r")
    return torch.from_numpy(X), torch.from_numpy(y), klassen


# ------------------------------------------------------------- Training

def trainieren(args):
    from training.pretrain import pick_device

    device = pick_device(args.device)
    print(f"Lade Daten aus {args.daten}/ ...")
    X, y, klassen = lade_daten(args.daten)
    print(f"{len(X):,} Beispiele, {len(klassen)} Klassen, {X.shape[1]} Abtastwerte")

    torch.manual_seed(0)
    misch = torch.randperm(len(X))
    X, y = X[misch], y[misch]
    n_test = int(len(X) * 0.15)
    Xtest, ytest, Xtrain, ytrain = X[:n_test], y[:n_test], X[n_test:], y[n_test:]
    print(f"{len(Xtrain):,} zum Trainieren, {len(Xtest):,} zum Testen\n")

    netz = Erkenner(len(klassen), args.breite).to(device)
    n_par = sum(p.numel() for p in netz.parameters())
    print(f"Modell: {n_par:,} Parameter  (int8 etwa {n_par/1e3:.0f} KB)\n")

    opt = torch.optim.AdamW(netz.parameters(), lr=args.lr, weight_decay=0.01)
    plan = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochen * (len(Xtrain) // args.batch + 1))

    for epoche in range(args.epochen):
        netz.train()
        summe = treffer = gesamt = 0
        for i in range(0, len(Xtrain), args.batch):
            xb = Xtrain[i:i + args.batch].to(device)
            yb = ytrain[i:i + args.batch].to(device)
            logits = netz(xb)
            verlust = F.cross_entropy(logits, yb, label_smoothing=0.05)
            opt.zero_grad(); verlust.backward(); opt.step(); plan.step()
            summe += verlust.item() * len(xb)
            treffer += (logits.argmax(1) == yb).sum().item()
            gesamt += len(xb)

        netz.eval()
        with torch.no_grad():
            t_treffer = 0
            for i in range(0, len(Xtest), 256):
                xb = Xtest[i:i + 256].to(device)
                t_treffer += (netz(xb).argmax(1).cpu() == ytest[i:i + 256]).sum().item()
        print(f"Epoche {epoche+1:2}/{args.epochen}  loss {summe/gesamt:.3f}  "
              f"Training {treffer/gesamt*100:5.1f} %  Test {t_treffer/len(Xtest)*100:5.1f} %")

    torch.save({"model": netz.state_dict(), "klassen": klassen,
                "breite": args.breite}, args.out)
    print(f"\nGespeichert: {args.out}")

    # Wo verwechselt es was?
    netz.eval()
    with torch.no_grad():
        vorher = torch.cat([netz(Xtest[i:i+256].to(device)).argmax(1).cpu()
                            for i in range(0, len(Xtest), 256)])
    print("\nFehler pro Klasse:")
    for k, name in enumerate(klassen):
        maske = ytest == k
        if maske.sum() == 0:
            continue
        quote = (vorher[maske] == k).float().mean().item()
        falsch = ""
        if quote < 0.98:
            verwechselt = vorher[maske][vorher[maske] != k]
            if len(verwechselt):
                haeufig = torch.bincount(verwechselt, minlength=len(klassen)).argmax()
                falsch = f"  meist verwechselt mit '{klassen[haeufig]}'"
        print(f"  {name:14} {quote*100:5.1f} %{falsch}")


# ------------------------------------------------------------- Anwenden

def lade_modell(pfad, device):
    ck = torch.load(pfad, map_location="cpu", weights_only=False)
    netz = Erkenner(len(ck["klassen"]), ck.get("breite", 32))
    netz.load_state_dict(ck["model"])
    netz.to(device).eval()
    return netz, ck["klassen"]


def testen(args):
    import soundfile as sf
    from training.pretrain import pick_device

    device = pick_device(args.device)
    netz, klassen = lade_modell(args.modell, device)
    d, rate = sf.read(args.datei, dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    n = int(RATE * DAUER)
    audio = np.zeros(n, dtype=np.float32)
    audio[:min(len(d), n)] = d[:n]
    with torch.no_grad():
        p = F.softmax(netz(torch.from_numpy(audio[None]).to(device))[0], dim=-1)
    for wert, i in zip(*p.topk(3)):
        print(f"  {klassen[i]:14} {wert*100:5.1f} %")


def hoeren(args):
    import sounddevice as sd
    from training.pretrain import pick_device

    device = pick_device(args.device)
    netz, klassen = lade_modell(args.modell, device)
    n = int(RATE * DAUER)
    print(f"Hoere zu. Sprich eines der Kommandos:\n  {', '.join(klassen[:-1])}")
    print("Abbruch mit Strg+C\n")
    puffer = np.zeros(n, dtype=np.float32)
    try:
        with sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                            blocksize=SPRUNG * 5) as strom:
            while True:
                block, _ = strom.read(SPRUNG * 5)
                puffer = np.concatenate([puffer[len(block):], block[:, 0]])
                if np.abs(puffer).max() < 0.02:      # Stille
                    continue
                with torch.no_grad():
                    p = F.softmax(netz(torch.from_numpy(puffer[None]).to(device))[0], -1)
                wert, i = p.max(0)
                if klassen[i] != "_unbekannt" and wert > args.schwelle:
                    print(f"  erkannt: {klassen[i]:14} ({wert*100:.0f} %)")
                    puffer[:] = 0
                    time.sleep(0.4)
    except KeyboardInterrupt:
        print("\nBeendet.")


def main():
    ap = argparse.ArgumentParser()
    unter = ap.add_subparsers(dest="befehl", required=True)

    t = unter.add_parser("trainieren")
    t.add_argument("--daten", default="audio")
    t.add_argument("--out", default="ckpt_stt/kommandos.pt")
    t.add_argument("--epochen", type=int, default=12)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--breite", type=int, default=32)
    t.add_argument("--device", default="auto")

    p = unter.add_parser("testen")
    p.add_argument("--datei", required=True)
    p.add_argument("--modell", default="ckpt_stt/kommandos.pt")
    p.add_argument("--device", default="auto")

    h = unter.add_parser("hoeren")
    h.add_argument("--modell", default="ckpt_stt/kommandos.pt")
    h.add_argument("--schwelle", type=float, default=0.7)
    h.add_argument("--device", default="auto")

    args = ap.parse_args()
    if args.befehl == "trainieren":
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        trainieren(args)
    elif args.befehl == "testen":
        testen(args)
    else:
        hoeren(args)


if __name__ == "__main__":
    main()
