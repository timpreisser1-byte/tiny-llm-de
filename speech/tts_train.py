"""Den Stimmen-Schueler trainieren, ein Netz nach dem anderen.

Warum nacheinander und nicht zusammen: die drei Netze haben voellig
verschiedene Ziele und Fehlermasse. Getrennt trainiert sieht man sofort,
welches klemmt - und genau diese Sichtbarkeit hat mir bei der
Bausteinsynthese den ganzen Tag gefehlt.

    python -m speech.tts_train dauer   --schritte 4000
    python -m speech.tts_train klang   --schritte 20000
    python -m speech.tts_train dekoder --schritte 20000
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from speech.tts_model import (Dauer, Dekoder, FFT, Klang, Kritiker, MEL, RATE,
                        SPRUNG, erzeuger_verlust, kritiker_verlust,
                        merkmal_verlust,
                        mel_bank, zu_mel)


class Daten:
    """Lehrerdaten: Phonem-IDs, Dauern, Ton."""

    def __init__(self, ordner="data_tts", device="cpu"):
        with open(os.path.join(ordner, "index.json"), encoding="utf-8") as f:
            d = json.load(f)
        self.e = d["eintraege"]
        self.ton = np.memmap(os.path.join(ordner, "ton.i16"), dtype=np.int16, mode="r")
        self.device = device
        self.rng = np.random.default_rng(1234)
        print(f"  {len(self.e):,} Sätze, {len(self.ton)/RATE/60:.0f} min Ton")

    def stapel(self, n, max_phon=180):
        """n Saetze, auf gleiche Laenge aufgefuellt."""
        gewaehlt, versuche = [], 0
        while len(gewaehlt) < n and versuche < n * 20:
            versuche += 1
            e = self.e[int(self.rng.integers(0, len(self.e)))]
            if 8 <= len(e["ids"]) <= max_phon and sum(e["dauer"]) > 8:
                gewaehlt.append(e)
        p = max(len(e["ids"]) for e in gewaehlt)
        ids = torch.zeros(len(gewaehlt), p, dtype=torch.long)
        dau = torch.zeros(len(gewaehlt), p, dtype=torch.long)
        maske = torch.zeros(len(gewaehlt), p)
        for i, e in enumerate(gewaehlt):
            k = len(e["ids"])
            ids[i, :k] = torch.tensor(e["ids"])
            dau[i, :k] = torch.tensor(e["dauer"])
            maske[i, :k] = 1
        r = max(sum(e["dauer"]) for e in gewaehlt)
        ton = torch.zeros(len(gewaehlt), r * SPRUNG)
        for i, e in enumerate(gewaehlt):
            w = np.array(self.ton[e["pos"]:e["pos"] + e["n"]], dtype=np.float32) / 32768
            m = min(len(w), r * SPRUNG)
            ton[i, :m] = torch.from_numpy(w[:m])
        return (ids.to(self.device), dau.to(self.device), maske.to(self.device),
                ton.to(self.device))


def balken(i, n, t0, text=""):
    b = 24
    v = int(b * i / max(n, 1))
    rest = (time.time() - t0) / max(i, 1) * (n - i)
    sys.stderr.write(f"\r  [{'█'*v}{'░'*(b-v)}] {i:>6,}/{n:,}  "
                     f"noch {rest/60:5.1f} min  {text:<26}")
    sys.stderr.flush()


def mehrfach_stft(a, b, groessen=(512, 1024, 2048)):
    """Fehler ueber mehrere Fenstergroessen - haelt Grobes und Feines im Blick."""
    v = 0.0
    for g in groessen:
        f = torch.hann_window(g, device=a.device)
        A = torch.stft(a, g, g // 4, window=f, return_complex=True).abs()
        B = torch.stft(b, g, g // 4, window=f, return_complex=True).abs()
        v = v + F.l1_loss(torch.log(A + 1e-5), torch.log(B + 1e-5))
    return v / len(groessen)


BAUFORMEN = {
    # saanoTTS veroeffentlicht drei Ausbaustufen, und die Kurve hat einen
    # scharfen Knick: 0,57M Parameter -> Guete 2,54, aber 1,45M -> 4,13.
    # Von dort auf 1,83M bringt nur noch 0,03. Der Knick liegt bei 1,45M,
    # und genau so viel Platz ist im Flash-Plan fuer die Stimme reserviert.
    "klein": {"dauer": {}, "klang": {"breite": 48}, "dekoder": {"breite": 76}},
    "gross": {"dauer": {}, "klang": {"breite": 70}, "dekoder": {"breite": 136}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teil", choices=["dauer", "klang", "dekoder"])
    ap.add_argument("--schritte", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--daten", default="data_tts")
    ap.add_argument("--out", default="ckpt_tts")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--warm", type=int, default=2000,
                    help="Schritte ohne Gegenspieler, damit er nichts zerreisst")
    ap.add_argument("--misch", type=float, default=0.5,
                    help="Anteil Schritte mit vorhergesagtem statt echtem Mel")
    ap.add_argument("--g-stft", type=float, default=45.0)
    ap.add_argument("--g-gegen", type=float, default=1.0)
    ap.add_argument("--g-merkmal", type=float, default=2.0)
    ap.add_argument("--ohne-gegner", action="store_true")
    ap.add_argument("--bau", choices=["klein", "gross"], default="gross",
                    help="klein = 0,61 MB wie saanoTTS eingebettet, "
                         "gross = 1,44 MB wie deren Pareto-Stufe")
    ap.add_argument("--weiter", action="store_true",
                    help="an einem abgebrochenen Lauf anknuepfen")
    ap.add_argument("--sichern", type=int, default=1000,
                    help="alle wieviel Schritte ein Pruefpunkt geschrieben wird")
    args = ap.parse_args()

    from training.pretrain import pick_device
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    d = Daten(args.daten, device)
    bank = mel_bank().to(device)
    fenster = torch.hann_window(FFT, device=device)

    # Die Bauform muss im Pruefpunkt stehen, sonst weiss der Sprecher spaeter
    # nicht, wie breit die Netze waren, und laedt in die falsche Form.
    bau = BAUFORMEN[args.bau][args.teil]
    netz = {"dauer": Dauer, "klang": Klang, "dekoder": Dekoder}[args.teil](**bau)
    netz = netz.to(device)
    n = sum(p.numel() for p in netz.parameters())
    print(f"  {args.teil}: {n:,} Parameter ({args.bau}, {bau}) auf {device}")
    opt = torch.optim.AdamW(netz.parameters(), lr=args.lr, weight_decay=0.01)
    plan = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, args.schritte, pct_start=0.1)

    ab = 1
    pfad = f"{args.out}/{args.teil}.pt"
    if args.weiter and os.path.exists(pfad):
        ck = torch.load(pfad, map_location=device, weights_only=False)
        netz.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
            plan.load_state_dict(ck["plan"])
        ab = ck["schritt"] + 1
        if ab > args.schritte:
            print(f"  bereits fertig ({ck['schritt']:,} von {args.schritte:,} "
                  f"Schritten) - nichts zu tun")
            return
        print(f"  weiter ab Schritt {ab:,} von {args.schritte:,}")

    # ---- Gegenspieler und Vorgaenger nur fuer den Dekoder
    kritiker = opt_k = klang_netz = None
    if args.teil == "dekoder" and not args.ohne_gegner:
        kritiker = Kritiker().to(device)
        opt_k = torch.optim.AdamW(kritiker.parameters(), lr=args.lr,
                                  betas=(0.8, 0.99), weight_decay=0.0)
        pk = os.path.join(args.out, "kritiker.pt")
        if args.weiter and os.path.exists(pk):
            ckk = torch.load(pk, map_location=device, weights_only=False)
            kritiker.load_state_dict(ckk["model"])
            if "opt" in ckk:
                opt_k.load_state_dict(ckk["opt"])
        print(f"  Gegenspieler: {sum(p.numel() for p in kritiker.parameters()):,} "
              f"Parameter, ab Schritt {args.warm:,}")
    if args.teil == "dekoder" and args.misch > 0:
        p = os.path.join(args.out, "klang.pt")
        if os.path.exists(p):
            ckl = torch.load(p, map_location="cpu", weights_only=False)
            klang_netz = Klang(**ckl.get("bau", {})).to(device).eval()
            klang_netz.load_state_dict(ckl["model"])
            for x in klang_netz.parameters():
                x.requires_grad_(False)
            print(f"  Klang geladen - {args.misch:.0%} der Schritte mit "
                  f"vorhergesagtem Mel")
        else:
            print(f"  ! {p} fehlt - Dekoder uebt nur auf echtem Mel")
    print()

    t0, verlauf, verlauf_k = time.time(), [], []
    for schritt in range(ab, args.schritte + 1):
        ids, dau, maske, ton = d.stapel(args.batch)
        if args.teil == "dauer":
            vor = netz(ids)
            ziel = torch.log(dau.float().clamp(min=1))
            verlust = (F.huber_loss(vor, ziel, reduction="none") * maske).sum() / maske.sum()
        else:
            with torch.no_grad():
                mel = zu_mel(ton, bank, fenster)
            if args.teil == "klang":
                vor = netz(ids, dau.cpu().numpy())
                k = min(vor.shape[2], mel.shape[2])
                verlust = F.l1_loss(vor[:, :, :k], mel[:, :, :k])
            else:
                # Der Dekoder bekommt beim Sprechen nie echtes Mel, sondern
                # das vorhergesagte - und das ist deutlich glatter. Uebt er
                # nur auf echtem, trifft ihn der Unterschied erst zur Laufzeit.
                eingang = mel
                if (klang_netz is not None
                        and np.random.rand() < args.misch * min(1.0, schritt / max(args.warm, 1))):
                    with torch.no_grad():
                        v = klang_netz(ids, dau.cpu().numpy())
                    j = min(v.shape[2], mel.shape[2])
                    eingang = v[:, :, :j]
                aus = netz(eingang)
                k = min(aus.shape[1], ton.shape[1])
                aus, echt_ton = aus[:, :k], ton[:, :k]
                verlust = mehrfach_stft(aus, echt_ton)

                if kritiker is not None and schritt > args.warm:
                    # 1. der Kritiker lernt zu unterscheiden
                    w_e, _ = kritiker(echt_ton)
                    w_f, _ = kritiker(aus.detach())
                    v_k = kritiker_verlust(w_e, w_f)
                    opt_k.zero_grad(); v_k.backward()
                    torch.nn.utils.clip_grad_norm_(kritiker.parameters(), 5.0)
                    opt_k.step()
                    verlauf_k.append(v_k.item())
                    # 2. der Erzeuger versucht ihn zu taeuschen
                    w_f, m_f = kritiker(aus)
                    with torch.no_grad():
                        _, m_e = kritiker(echt_ton)
                    verlust = (args.g_stft * verlust
                               + args.g_gegen * erzeuger_verlust(w_f)
                               + args.g_merkmal * merkmal_verlust(m_e, m_f))
        opt.zero_grad(); verlust.backward()
        torch.nn.utils.clip_grad_norm_(netz.parameters(), 1.0)
        opt.step(); plan.step()
        verlauf.append(verlust.item())
        if schritt % 25 == 0:
            zusatz = (f"  Kritiker {np.mean(verlauf_k[-100:]):.3f}"
                      if verlauf_k else "")
            balken(schritt, args.schritte, t0,
                   f"Verlust {np.mean(verlauf[-100:]):.4f}{zusatz}")
        if schritt % args.sichern == 0 or schritt == args.schritte:
            torch.save({"model": netz.state_dict(), "teil": args.teil,
                        "bau": bau, "schritt": schritt,
                        "opt": opt.state_dict(), "plan": plan.state_dict()},
                       f"{args.out}/{args.teil}.pt")
            if kritiker is not None:
                torch.save({"model": kritiker.state_dict(),
                            "opt": opt_k.state_dict()},
                           f"{args.out}/kritiker.pt")
    sys.stderr.write("\r" + " " * 80 + "\r")
    print(f"  fertig: Verlust {np.mean(verlauf[-200:]):.4f} "
          f"nach {(time.time()-t0)/60:.0f} min -> {args.out}/{args.teil}.pt")


if __name__ == "__main__":
    main()
