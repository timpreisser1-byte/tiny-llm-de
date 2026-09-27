"""Phase 2: CTC-Spracherkenner - freie Transkription statt fester Kommandos.

Aufbau, bewusst ohne Transformer und ohne Rekursion:

    Mel 40 x T
       |
    Conv1d mit Schrittweite 2   ->  T/2 Zeitschritte, 25 pro Sekunde
       |
    mehrere Restblöcke aus Faltungen
       |
    je Zeitschritt 34 Werte     ->  33 Zeichen + CTC-Leerzeichen
       |
    CTC -> "wie hoch ist der eiffelturm"

CTC loest das Grundproblem der Spracherkennung: Niemand weiss vorher, welcher
Zeitschritt zu welchem Buchstaben gehoert. Die Verlustfunktion summiert ueber
alle moeglichen Zuordnungen und braucht deshalb keine Ausrichtung in den Daten.

    python -m speech.asr_model trainieren --daten 1_hours --epochen 60
    python -m speech.asr_model testen --daten 1_hours
    python -m speech.asr_model hoeren
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from speech.asr_data import N_MEL, RATE, SPRUNG, ZEICHEN, indizes_zu_text, merkmale


# ------------------------------------------------------------------ Modell

class Block(nn.Module):
    """Faltung mit Abkuerzung. Die Abkuerzung haelt tiefe Netze trainierbar."""

    def __init__(self, kanaele, kern=5, dilatation=1):
        super().__init__()
        rand = (kern - 1) // 2 * dilatation
        self.conv1 = nn.Conv1d(kanaele, kanaele, kern, padding=rand,
                               dilation=dilatation)
        self.norm1 = nn.BatchNorm1d(kanaele)
        self.conv2 = nn.Conv1d(kanaele, kanaele, kern, padding=rand,
                               dilation=dilatation)
        self.norm2 = nn.BatchNorm1d(kanaele)

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.relu(x + h)


class CTCErkenner(nn.Module):
    def __init__(self, n_zeichen, kanaele=192, bloecke=6):
        super().__init__()
        self.eingang = nn.Sequential(
            # Schrittweite 2: aus 50 Rahmen pro Sekunde werden 25.
            # Mehr Raffung waere fatal - deutsche Sprache hat rund 15 Zeichen
            # pro Sekunde, und CTC braucht mehr Zeitschritte als Zeichen.
            nn.Conv1d(N_MEL, kanaele, 5, stride=2, padding=2),
            nn.BatchNorm1d(kanaele), nn.ReLU(),
        )
        # wachsende Dilatation: spaetere Bloecke sehen einen groesseren
        # Zeitausschnitt, ohne dass die Rechenlast steigt
        self.bloecke = nn.Sequential(*[
            Block(kanaele, 5, dilatation=2 ** (i % 3)) for i in range(bloecke)])
        self.aus = nn.Conv1d(kanaele, n_zeichen, 1)

    def forward(self, mel):
        """mel: (B, 40, T) -> log-Wahrscheinlichkeiten (T/2, B, n_zeichen)"""
        x = self.eingang(mel)
        x = self.bloecke(x)
        x = self.aus(x)                       # (B, n_zeichen, T/2)
        return F.log_softmax(x.permute(2, 0, 1), dim=-1)


# ------------------------------------------------------------------ Daten

class Sprachdaten:
    """Eine oder mehrere Portionen. "train0,train1,train2" legt sie zusammen -
    fuer den grossen Lauf braucht es Dutzende Shards, aber nur eine Datenquelle."""

    def __init__(self, ordner, portionen):
        self.aufnahmen, self.speicher = [], []
        for nr, portion in enumerate(str(portionen).split(",")):
            portion = portion.strip()
            with open(os.path.join(ordner, f"{portion}.json"), encoding="utf-8") as f:
                meta = json.load(f)
            self.speicher.append(np.memmap(os.path.join(ordner, f"{portion}.f16"),
                                           dtype=np.float16, mode="r"))
            for a in meta["aufnahmen"]:
                a = dict(a); a["quelle"] = nr
                self.aufnahmen.append(a)

    def __len__(self):
        return len(self.aufnahmen)

    def hole(self, i):
        a = self.aufnahmen[i]
        speicher = self.speicher[a.get("quelle", 0)]
        m = np.array(speicher[a["pos"]:a["pos"] + a["rahmen"] * N_MEL],
                     dtype=np.float32).reshape(a["rahmen"], N_MEL)
        return m, a["ziele"], a["text"]

    @staticmethod
    def verdecke(x, zeit_max=25, frequenz_max=8, wie_oft=2):
        """SpecAugment: zufaellige Streifen im Spektrogramm ausblenden.

        Das Modell darf sich dann nicht auf einzelne Stellen verlassen und
        lernt stattdessen den Zusammenhang. Bei Spracherkennung ist das die
        wirksamste Massnahme gegen Auswendiglernen und kostet nichts.
        """
        b, mel, t = x.shape
        for _ in range(wie_oft):
            for k in range(b):
                breite = np.random.randint(0, zeit_max)
                if breite and t > breite:
                    start = np.random.randint(0, t - breite)
                    x[k, :, start:start + breite] = 0
                hoehe = np.random.randint(0, frequenz_max)
                if hoehe:
                    start = np.random.randint(0, mel - hoehe)
                    x[k, start:start + hoehe, :] = 0
        return x

    def stapel(self, indizes, device, augmentieren=False):
        """Aufnahmen auf gleiche Laenge bringen - CTC braucht die echten Laengen."""
        stuecke = [self.hole(i) for i in indizes]
        max_t = max(len(m) for m, _, _ in stuecke)
        x = torch.zeros(len(stuecke), N_MEL, max_t)
        eingabe_laengen, ziel_laengen, ziele = [], [], []
        for k, (m, z, _) in enumerate(stuecke):
            x[k, :, :len(m)] = torch.from_numpy(m).T
            eingabe_laengen.append(len(m) // 2)     # nach der Schrittweite 2
            ziel_laengen.append(len(z))
            ziele.extend(z)
        if augmentieren:
            x = self.verdecke(x)
        return (x.to(device), torch.tensor(ziele),
                torch.tensor(eingabe_laengen), torch.tensor(ziel_laengen))


# ------------------------------------------------------------------ Dekodieren

def gierig(logits):
    """Einfachste Dekodierung: pro Zeitschritt das wahrscheinlichste Zeichen,
    dann Wiederholungen und Leerzeichen zusammenfassen."""
    beste = logits.argmax(dim=-1)
    raus, vorher = [], -1
    for i in beste.tolist():
        if i != vorher and i != 0:
            raus.append(i)
        vorher = i
    return indizes_zu_text(raus)


def wortfehlerrate(soll: str, ist: str) -> float:
    """Levenshtein auf Wortebene - der Standardwert für Spracherkennung."""
    a, b = soll.split(), ist.split()
    if not a:
        return 1.0 if b else 0.0
    d = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        neu = [i]
        for j, wb in enumerate(b, 1):
            neu.append(min(d[j] + 1, neu[j - 1] + 1, d[j - 1] + (wa != wb)))
        d = neu
    return d[-1] / len(a)


# ------------------------------------------------------------------ Anzeige

def balken(anteil: float, breite: int = 22) -> str:
    voll = int(anteil * breite)
    return "█" * voll + "░" * (breite - voll)


def zeit(sekunden: float) -> str:
    if sekunden < 90:
        return f"{sekunden:.0f}s"
    if sekunden < 5400:
        return f"{sekunden/60:.0f} min"
    return f"{sekunden/3600:.1f} h"


# ------------------------------------------------------------------ Training

def bewerte(netz, daten, device, max_n=60):
    netz.eval()
    fehler, gesamt, beispiele = 0.0, 0, []
    with torch.no_grad():
        for i in range(min(max_n, len(daten))):
            m, _, text = daten.hole(i)
            x = torch.from_numpy(m).T[None].to(device)
            ausgabe = netz(x)[:, 0]
            erkannt = gierig(ausgabe)
            fehler += wortfehlerrate(text, erkannt)
            gesamt += 1
            if len(beispiele) < 2:
                beispiele.append((text, erkannt))
    netz.train()
    return fehler / max(gesamt, 1), beispiele


def trainieren(args):
    from training.pretrain import pick_device

    device = pick_device(args.device)
    daten = Sprachdaten(args.dir, args.daten)
    print(f"{len(daten):,} Aufnahmen aus {args.daten}")

    netz = CTCErkenner(len(ZEICHEN) + 1, args.kanaele, args.bloecke)
    if getattr(args, "ternaer", False):
        from lm.ternary import zu_ternaer_conv
        k = zu_ternaer_conv(netz)
        n = sum(p.numel() for p in netz.parameters())
        print(f"  ternaer: {k} Faltungen, Export {n*2/8/1e6:.2f} MB "
              f"statt {n*4/1e6:.2f} MB")
    netz = netz.to(device)
    n = sum(p.numel() for p in netz.parameters())
    print(f"Modell: {n:,} Parameter  (int8 etwa {n/1e6:.1f} MB)")

    start_epoche = 1
    if args.fortsetzen and os.path.exists(args.out):
        ck = torch.load(args.out, map_location="cpu", weights_only=False)
        netz.load_state_dict(ck["model"])
        start_epoche = ck.get("epoche", 0) + 1
        print(f"Fortsetzung ab Epoche {start_epoche}")
    print()

    verlustfunktion = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(netz.parameters(), lr=args.lr, weight_decay=0.01)
    # nach Laenge sortiert stapeln spart viel Fuellmaterial
    reihenfolge = sorted(range(len(daten)), key=lambda i: daten.aufnahmen[i]["rahmen"])
    stapel = [reihenfolge[i:i + args.batch]
              for i in range(0, len(reihenfolge), args.batch)]
    plan = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochen * len(stapel), pct_start=0.2)

    t0 = time.time()
    gesamt_stapel = (args.epochen - start_epoche + 1) * len(stapel)
    fertig = 0
    for epoche in range(start_epoche, args.epochen + 1):
        np.random.shuffle(stapel)
        summe = 0.0
        for nummer, gruppe in enumerate(stapel, 1):
            x, ziele, e_laengen, z_laengen = daten.stapel(
                gruppe, device, augmentieren=not args.ohne_augmentierung)
            ausgabe = netz(x)
            # Die CTC-Verlustfunktion fehlt auf Apples MPS. Die Ausgabe ist mit
            # (T/2, B, 34) winzig, der Umweg über die CPU kostet also kaum Zeit -
            # das Netz selbst rechnet weiter auf der GPU.
            verlust = verlustfunktion(ausgabe.cpu(), ziele, e_laengen, z_laengen)
            opt.zero_grad(); verlust.backward()
            torch.nn.utils.clip_grad_norm_(netz.parameters(), 5.0)
            opt.step(); plan.step()
            summe += verlust.item()

            # Laufende Anzeige: Balken über den gesamten Lauf, nicht nur die
            # Epoche - so sieht man von Anfang an, wie lange es noch dauert.
            fertig += 1
            if nummer % 5 == 0 or nummer == len(stapel):
                anteil = fertig / gesamt_stapel
                vergangen = time.time() - t0
                rest = vergangen / max(anteil, 1e-6) - vergangen
                print(f"\r  {balken(anteil)} {anteil*100:5.1f} %  "
                      f"Epoche {epoche:>3}/{args.epochen}  "
                      f"Verlust {summe/nummer:6.3f}  "
                      f"noch {zeit(rest):>7}  ", end="", flush=True)
        print("\r" + " " * 78 + "\r", end="")
        if epoche % args.sichern == 0 or epoche == args.epochen:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({"model": netz.state_dict(), "kanaele": args.kanaele,
                        "bloecke": args.bloecke, "zeichen": ZEICHEN,
                        "ternaer": getattr(args, "ternaer", False),
                        "epoche": epoche}, args.out)
        if epoche % args.zeige == 0 or epoche == args.epochen:
            wer, beispiele = bewerte(netz, daten, device)
            print(f"Epoche {epoche:3}/{args.epochen}  CTC-Verlust {summe/len(stapel):6.3f}"
                  f"  Wortfehler {wer*100:5.1f} %  {(time.time()-t0)/60:5.1f} min")
            if epoche == args.epochen or wer < 0.35:
                for soll, ist in beispiele[:1]:
                    print(f"    soll: {soll[:78]}")
                    print(f"    ist : {ist[:78]}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": netz.state_dict(), "kanaele": args.kanaele,
                "bloecke": args.bloecke, "zeichen": ZEICHEN,
                "ternaer": getattr(args, "ternaer", False)}, args.out)
    print(f"\nGespeichert: {args.out}")


def lade_modell(pfad, device):
    ck = torch.load(pfad, map_location="cpu", weights_only=False)
    netz = CTCErkenner(len(ck["zeichen"]) + 1, ck["kanaele"], ck["bloecke"])
    if ck.get("ternaer"):
        from lm.ternary import zu_ternaer_conv
        zu_ternaer_conv(netz)
    netz.load_state_dict(ck["model"])
    netz.to(device).eval()
    return netz


def testen(args):
    from training.pretrain import pick_device

    device = pick_device(args.device)
    netz = lade_modell(args.modell, device)

    if args.datei:
        import soundfile as sf
        audio, rate = sf.read(args.datei, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != RATE:
            m = int(len(audio) * RATE / rate)
            audio = np.interp(np.linspace(0, len(audio) - 1, m),
                              np.arange(len(audio)), audio).astype(np.float32)
        mel = merkmale(np.ascontiguousarray(audio)).astype(np.float32)
        with torch.no_grad():
            print(gierig(netz(torch.from_numpy(mel).T[None].to(device))[:, 0]))
        return

    daten = Sprachdaten(args.dir, args.daten)
    wer, beispiele = bewerte(netz, daten, device, max_n=args.anzahl)
    print(f"Wortfehlerrate auf {args.daten}: {wer*100:.1f} %\n")
    for soll, ist in beispiele:
        print(f"  soll: {soll[:80]}")
        print(f"  ist : {ist[:80]}\n")


def main():
    ap = argparse.ArgumentParser()
    unter = ap.add_subparsers(dest="befehl", required=True)

    t = unter.add_parser("trainieren")
    t.add_argument("--dir", default="data/asr")
    t.add_argument("--daten", default="1_hours")
    t.add_argument("--out", default="ckpt_asr/ctc.pt")
    t.add_argument("--epochen", type=int, default=60)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--kanaele", type=int, default=192)
    t.add_argument("--bloecke", type=int, default=6)
    t.add_argument("--zeige", type=int, default=5)
    t.add_argument("--ohne-augmentierung", action="store_true",
                   help="SpecAugment abschalten (nur zum Vergleichen)")
    t.add_argument("--fortsetzen", action="store_true",
                   help="an einem vorhandenen Checkpoint weitermachen")
    t.add_argument("--sichern", type=int, default=2,
                   help="alle wie viele Epochen speichern")
    t.add_argument("--ternaer", action="store_true",
                   help="Faltungen ternaer trainieren - 2 Bit statt 32")
    t.add_argument("--device", default="auto")

    p = unter.add_parser("testen")
    p.add_argument("--dir", default="data/asr")
    p.add_argument("--daten", default="1_hours")
    p.add_argument("--modell", default="ckpt_asr/ctc.pt")
    p.add_argument("--datei", default=None, help="eigene wav-Datei")
    p.add_argument("--anzahl", type=int, default=60)
    p.add_argument("--device", default="auto")

    args = ap.parse_args()
    if args.befehl == "trainieren":
        trainieren(args)
    else:
        testen(args)


if __name__ == "__main__":
    main()
