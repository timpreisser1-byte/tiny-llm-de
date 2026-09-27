"""Das ganze Gerät, gesprochen: hören, nachschlagen, antworten, sprechen.

Hier laufen zum ersten Mal alle vier Teile am Stück, so wie sie spaeter auf
dem ESP32-P4 zusammenlaufen sollen:

    Mikrofon
       |  Erkenner (ternaer, 1,33 MB)          asr_modell.py
    Text der Frage
       |  Wortindex auf der Karte               wissen_index.py
    zwei Absaetze aus der Wikipedia
       |  Sprachmodell (Sherry, 2,93 MB)        model.py
    Antwort als Text
       |  Lautschrift und Bausteine (1,13 MB)   tts_laute.py, tts_sprechen.py
    Lautsprecher

Jede Stufe wird einzeln ausgegeben, damit man sieht, wo es hakt. Die
haeufigste Fehlerquelle ist die erste: bei 53 Prozent Wortfehlerrate kommt
die Frage oft verstuemmelt an, und dann kann der Rest nichts mehr retten.

    python geraet_sprache.py --aufnehmen 4      # vier Sekunden Mikrofon
    python geraet_sprache.py --datei frage.wav
    python geraet_sprache.py --text "Wie tief ist der Bodensee?"
"""

import argparse
import sys
import time

import numpy as np
import torch

GRUEN, GELB, GRAU, FETT, AUS = "\033[92m", "\033[93m", "\033[90m", "\033[1m", "\033[0m"


def stufe(name, text, dauer, farbe=""):
    print(f"  {GRAU}{name:22}{AUS}{farbe}{text}{AUS}")
    print(f"  {GRAU}{'':22}{dauer*1000:.0f} ms{AUS}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None, help="Frage als Text, ohne Erkenner")
    ap.add_argument("--datei", default=None, help="WAV-Datei mit der Frage")
    ap.add_argument("--aufnehmen", type=float, default=0, help="Sekunden Mikrofon")
    ap.add_argument("--ckpt", default="ckpt_8mb/sft_kurz.pt")
    ap.add_argument("--erkenner", default="ckpt_asr/ctc_gross_tern12.pt")
    ap.add_argument("--bausteine", default="stimme/bausteine.npz")
    ap.add_argument("--wissen", default="wiki")
    ap.add_argument("--out", default="stimme/antwort.wav")
    ap.add_argument("--absaetze", type=int, default=2)
    ap.add_argument("--gierig", action="store_true",
                    help="nur das wahrscheinlichste Zeichen je Zeitschritt - "
                         "schnell, aber orthografisch kaputt")
    ap.add_argument("--breite", type=int, default=48)
    ap.add_argument("--nbest", type=int, default=10)
    args = ap.parse_args()

    import soundfile as sf
    from tokenizers import Tokenizer
    from asr_daten import RATE, merkmale
    from geraet import antworte, lade
    from train import pick_device
    from tts_sprechen import lade as lade_bausteine, sprich
    from wissen_index import Wissensbasis

    device = pick_device("cpu")
    print(f"\n  {GRAU}lade Erkenner, Sprachmodell, Index und Stimme ...{AUS}", flush=True)
    t = time.time()
    netz, cfg = lade(args.ckpt, device)
    tok = Tokenizer.from_file("tokenizer/de_bpe.json")
    wb = Wissensbasis(args.wissen)
    bausteine = lade_bausteine(args.bausteine)
    erkenner, baum = None, None
    if not args.text:
        import json
        from asr_modell import lade_modell
        from asr_strahl import Baum, strahlsuche
        from asr_llm import lm_bewertung
        globals().update(strahlsuche=strahlsuche, lm_bewertung=lm_bewertung)
        erkenner = lade_modell(args.erkenner, device)
        if not args.gierig:
            wb_ = json.load(open("asr_woerter.json", encoding="utf-8"))
            paare = sorted(wb_["woerter"].items(), key=lambda x: -x[1])[:60000]
            baum = Baum(dict(paare), sum(n for _, n in paare))
    print(f"  {GRAU}bereit in {time.time()-t:.1f} s{AUS}\n")

    # ---- 1. hören
    if args.text:
        frage, t_hoeren = args.text, 0.0
    else:
        if args.aufnehmen:
            import sounddevice as sd
            # In der Frequenz des Geraets aufnehmen und danach umrechnen.
            # Das eingebaute Mikrofon laeuft mit 48 kHz und lehnt 16 kHz mit
            # PaErrorCode -9986 ab - der Fehler sieht nach fehlender
            # Berechtigung aus, ist aber nur die Abtastrate.
            geraet = sd.query_devices(kind="input")
            sr_auf = int(geraet["default_samplerate"])
            print(f"  {GRAU}{geraet['name']} bei {sr_auf} Hz{AUS}")
            print(f"  {FETT}Sprich jetzt ({args.aufnehmen:.0f} s) ...{AUS}", flush=True)
            ton = sd.rec(int(args.aufnehmen * sr_auf), samplerate=sr_auf,
                         channels=1, dtype="float32")
            sd.wait()
            ton = ton[:, 0]
            if sr_auf != RATE:
                m = int(len(ton) * RATE / sr_auf)
                ton = np.interp(np.linspace(0, len(ton) - 1, m),
                                np.arange(len(ton)), ton).astype(np.float32)
            spitze = float(np.abs(ton).max())
            print(f"  {GRAU}aufgenommen: {len(ton)/RATE:.1f} s, Aussteuerung "
                  f"{spitze:.2f}{' - sehr leise!' if spitze < 0.05 else ''}{AUS}")
            sf.write("/tmp/aufnahme.wav", ton, RATE)
        else:
            ton, sr = sf.read(args.datei, dtype="float32")
            if ton.ndim > 1:
                ton = ton.mean(axis=1)
            if sr != RATE:
                m = int(len(ton) * RATE / sr)
                ton = np.interp(np.linspace(0, len(ton) - 1, m),
                                np.arange(len(ton)), ton).astype(np.float32)
        t = time.time()
        with torch.no_grad():
            lg = erkenner(torch.from_numpy(merkmale(ton).astype(np.float32).T)
                          .unsqueeze(0))[:, 0, :]
        if args.gierig:
            from asr_modell import gierig
            frage = gierig(lg)
        else:
            # Strahlsuche zwingt die Ausgabe auf echte deutsche Woerter, das
            # Sprachmodell waehlt danach unter den Kandidaten. Gemessen holt
            # das Woerterbuch rund einen Punkt, das Sprachmodell noch einen -
            # von 4,8, die in der Kandidatenliste stecken.
            kand = strahlsuche(lg.cpu().numpy(), baum, args.breite, 8, 2.0, 0.8,
                               nbest=args.nbest)
            bewertet = [(a + 0.3 * lm_bewertung(netz, tok, device, txt,
                                                cfg.block_size)
                         + 2.0 * len(txt.split()), txt) for txt, a in kand]
            frage = max(bewertet)[1] if bewertet else ""
        t_hoeren = time.time() - t
    stufe("gehört", frage or "(nichts)", t_hoeren, GELB)

    # ---- 2. nachschlagen
    t = time.time()
    absatz = wb.bester_text(frage, 1000, absaetze=args.absaetze)
    t_suche = time.time() - t
    stufe("von der Karte", (absatz or "(nichts gefunden)")[:150] + "...", t_suche)

    # ---- 3. antworten
    t = time.time()
    antwort = antworte(netz, tok, cfg, device, frage, absatz)
    t_denken = time.time() - t
    stufe("Antwort", antwort[:200], t_denken, GRUEN)

    # ---- 4. sprechen
    t = time.time()
    ausgabe = sprich(antwort, bausteine)
    sf.write(args.out, ausgabe, RATE)
    t_sprechen = time.time() - t
    stufe("gesprochen", f"{len(ausgabe)/RATE:.1f} s -> {args.out}", t_sprechen)

    gesamt = t_hoeren + t_suche + t_denken + t_sprechen
    print(f"  {FETT}{gesamt:.1f} s insgesamt{AUS}   "
          f"{GRAU}hören {t_hoeren:.1f} · suchen {t_suche:.2f} · "
          f"denken {t_denken:.1f} · sprechen {t_sprechen:.1f}{AUS}\n")


if __name__ == "__main__":
    main()
