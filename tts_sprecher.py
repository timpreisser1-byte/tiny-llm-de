"""Die drei Schuelernetze zum Sprechen zusammensetzen.

    Text  --eSpeak-->  Phoneme  --Dauer-->  Rahmenzahl je Laut
                                 --Klang-->  40 Werte je Rahmen
                                 --Dekoder-> Betrag, Phase, inverse FFT -> Ton

Auf Wunsch spricht der Lehrer denselben Satz mit, damit man den Abstand hoert.

    python tts_sprecher.py --ckpt ckpt_tts "Guten Morgen"
"""

import argparse
import os

import numpy as np
import torch

from tts_modell import Dauer, Dekoder, Klang, RATE


def lade(ordner, device):
    netze = {}
    for name, klasse in (("dauer", Dauer), ("klang", Klang), ("dekoder", Dekoder)):
        p = os.path.join(ordner, f"{name}.pt")
        if os.path.exists(p):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            n = klasse(**ck.get("bau", {}))   # Breite steht im Pruefpunkt
            n.load_state_dict(ck["model"])
        else:
            print(f"  ! {p} fehlt - nimmt untrainierte Gewichte")
            n = klasse()
        netze[name] = n.to(device).eval()
    return netze


@torch.no_grad()
def sprich(text, netze, lehrer, device, tempo=1.0):
    ids = torch.tensor([lehrer.ids(lehrer.phoneme(text))],
                       dtype=torch.long, device=device)
    log_d = netze["dauer"](ids)
    dauern = torch.clamp(torch.exp(log_d) * tempo, min=1).round().long().cpu().numpy()
    mel = netze["klang"](ids, dauern)
    ton = netze["dekoder"](mel).squeeze().cpu().numpy()
    spitze = float(np.abs(ton).max()) + 1e-9
    return (ton / spitze * 0.9).astype(np.float32), int(dauern.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--ckpt", default="ckpt_tts")
    ap.add_argument("--lehrer-auch", action="store_true")
    ap.add_argument("--tempo", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import soundfile as sf
    from lehrer_stimme import Lehrer
    from train import pick_device

    device = pick_device(args.device)
    netze = lade(args.ckpt, device)
    L = Lehrer()
    n = sum(sum(p.numel() for p in x.parameters()) for x in netze.values())
    print(f"\n  Schüler: {n:,} Parameter = {n/1e6:.2f} MB als int8\n")

    texte = args.text or ["Guten Morgen, wie geht es dir?"]
    for i, t in enumerate(texte, 1):
        ton, rahmen = sprich(t, netze, L, device, args.tempo)
        sf.write(f"probe_{i}_schueler.wav", ton, RATE)
        zeile = f"  {t}\n    Schüler: {len(ton)/RATE:.2f} s, {rahmen} Rahmen"
        if args.lehrer_auch:
            lt, _ = L.sprich(t)
            sf.write(f"probe_{i}_lehrer.wav", lt, L.rate)
            zeile += f"   Lehrer: {len(lt)/L.rate:.2f} s"
        print(zeile)
    print()


if __name__ == "__main__":
    main()
