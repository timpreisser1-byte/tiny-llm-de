"""Phase 4: Erkanntes auf echte deutsche Wörter zwingen.

Der CTC-Erkenner rät Buchstaben einzeln und schreibt "degnichen kenntnichen".
Lautlich ist das nah dran, orthografisch Unsinn. Ein Wörterbuch beim
Dekodieren macht daraus "technischen kenntnissen" - ohne dass das Modell
irgendetwas dazulernen muss.

Warum das so viel bringt: Deutsche Wörter sind im Raum aller Buchstabenfolgen
extrem selten. Von den Millionen moeglichen Achtbuchstaben-Folgen sind nur ein
paar Tausend echte Woerter. Wer die Ausgabe auf diese Menge zwingt, korrigiert
die meisten Buchstabenfehler von allein.

Das Wörterbuch entsteht aus den Daten, die ohnehin daliegen: den Transkripten
der Sprachdaten und dem Wikipedia-Index. Auf der SD-Karte braucht es wenige
Megabyte, und die Suche ist auf dem ESP32 in C nachbaubar.

    python asr_woerterbuch.py bauen
    python asr_woerterbuch.py testen --modell ckpt_asr/ctc_gross.pt
"""

import argparse
import json
import math
import os
import re
import time
from collections import Counter

WORT = re.compile(r"[a-zäöüß]+")


# ------------------------------------------------------------------ Aufbau

def bauen(args):
    zaehler = Counter()

    # 1) Transkripte der Sprachdaten - genau die Wörter, die vorkommen werden
    for datei in sorted(os.listdir(args.asr_dir)):
        if not datei.endswith(".json"):
            continue
        with open(os.path.join(args.asr_dir, datei), encoding="utf-8") as f:
            daten = json.load(f)
        for a in daten.get("aufnahmen", []):
            zaehler.update(WORT.findall(a["text"]))
        print(f"  {datei:16} {len(zaehler):>8,} verschiedene Wörter")

    # 2) Wikipedia dazu - deckt Namen und Fachbegriffe ab
    wiki = os.path.join(args.wiki_dir, "chunks.txt")
    if os.path.exists(wiki) and not args.ohne_wiki:
        gelesen = 0
        with open(wiki, encoding="utf-8", errors="ignore") as f:
            for zeile in f:
                zaehler.update(WORT.findall(zeile.lower()))
                gelesen += len(zeile)
                if gelesen > args.wiki_bytes:
                    break
        print(f"  {'Wikipedia':16} {len(zaehler):>8,} verschiedene Wörter")

    # Nur Wörter behalten, die mehrfach vorkommen - Tippfehler und
    # Erkennungsmüll aus den Quellen sollen nicht ins Wörterbuch
    haeufig = [(w, n) for w, n in zaehler.items() if n >= args.mindestens]
    haeufig.sort(key=lambda x: -x[1])
    haeufig = haeufig[:args.groesse]

    gesamt = sum(n for _, n in haeufig)
    woerterbuch = {w: n for w, n in haeufig}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"gesamt": gesamt, "woerter": woerterbuch}, f, ensure_ascii=False)

    print(f"\n{len(woerterbuch):,} Wörter -> {args.out} "
          f"({os.path.getsize(args.out)/1e6:.1f} MB)")
    print("Häufigste:", ", ".join(w for w, _ in haeufig[:12]))


# ------------------------------------------------------------------ Korrektur

class Woerterbuch:
    """Findet zu einem verstümmelten Wort das plausibelste echte Wort."""

    def __init__(self, pfad="asr_woerter.json", max_abstand=2):
        with open(pfad, encoding="utf-8") as f:
            daten = json.load(f)
        self.gesamt = daten["gesamt"]
        self.woerter = daten["woerter"]
        self.max_abstand = max_abstand
        # Nach Länge und Anfangsbuchstabe vorsortieren. Ohne diese Vorauswahl
        # müsste jedes Wort gegen alle 100.000 verglichen werden.
        self.eimer = {}
        for w in self.woerter:
            self.eimer.setdefault((len(w), w[0]), []).append(w)

    def abstand(self, a, b, grenze):
        """Levenshtein mit Abbruch, sobald die Grenze überschritten ist."""
        if abs(len(a) - len(b)) > grenze:
            return grenze + 1
        vorher = list(range(len(b) + 1))
        for i, za in enumerate(a, 1):
            aktuell = [i]
            bestes = i
            for j, zb in enumerate(b, 1):
                aktuell.append(min(vorher[j] + 1, aktuell[j - 1] + 1,
                                   vorher[j - 1] + (za != zb)))
                bestes = min(bestes, aktuell[j])
            if bestes > grenze:
                return grenze + 1
            vorher = aktuell
        return vorher[-1]

    def kandidaten(self, wort):
        """Wörter ähnlicher Länge mit ähnlichem Anfang."""
        raus = []
        for laenge in range(len(wort) - self.max_abstand, len(wort) + self.max_abstand + 1):
            for anfang in {wort[0], wort[1] if len(wort) > 1 else wort[0]}:
                raus.extend(self.eimer.get((laenge, anfang), ()))
        return raus

    def korrigiere_wort(self, wort):
        if wort in self.woerter or len(wort) <= 2:
            return wort
        bestes, beste_bewertung = wort, -1e9
        for kandidat in self.kandidaten(wort):
            d = self.abstand(wort, kandidat, self.max_abstand)
            if d > self.max_abstand:
                continue
            # Nähe zählt stark, Häufigkeit entscheidet bei Gleichstand
            bewertung = -d * 4 + math.log(self.woerter[kandidat] / self.gesamt)
            if bewertung > beste_bewertung:
                bestes, beste_bewertung = kandidat, bewertung
        return bestes

    def korrigiere(self, text):
        return " ".join(self.korrigiere_wort(w) for w in text.split())


# ------------------------------------------------------------------ Test

def testen(args):
    import torch
    from asr_modell import Sprachdaten, gierig, lade_modell, wortfehlerrate
    from train import pick_device

    device = pick_device(args.device)
    netz = lade_modell(args.modell, device)
    daten = Sprachdaten(args.dir, args.daten)
    wb = Woerterbuch(args.woerterbuch, args.max_abstand)
    print(f"{len(wb.woerter):,} Wörter im Wörterbuch, "
          f"max. {args.max_abstand} Buchstaben Abstand\n")

    roh_summe = korr_summe = 0.0
    t0 = time.time()
    beispiele = []
    with torch.no_grad():
        for i in range(min(args.anzahl, len(daten))):
            m, _, soll = daten.hole(i)
            x = torch.from_numpy(m).T[None].to(device)
            roh = gierig(netz(x)[:, 0])
            korr = wb.korrigiere(roh)
            roh_summe += wortfehlerrate(soll, roh)
            korr_summe += wortfehlerrate(soll, korr)
            if len(beispiele) < 3 and roh != korr:
                beispiele.append((soll, roh, korr))
            if (i + 1) % 20 == 0:
                print(f"\r  {i+1}/{args.anzahl} geprüft", end="", flush=True)
    n = min(args.anzahl, len(daten))
    print(f"\r{' '*30}\r")

    for soll, roh, korr in beispiele:
        print(f"  soll: {soll[:76]}")
        print(f"  roh : {roh[:76]}")
        print(f"  korr: {korr[:76]}\n")

    roh_wer, korr_wer = roh_summe / n * 100, korr_summe / n * 100
    print(f"{'':22}{'Wortfehlerrate':>16}")
    print(f"  {'ohne Wörterbuch':20}{roh_wer:14.1f} %")
    print(f"  {'mit Wörterbuch':20}{korr_wer:14.1f} %")
    print(f"\n  Verbesserung: {roh_wer-korr_wer:+.1f} Prozentpunkte "
          f"({(1-korr_wer/roh_wer)*100:.0f} % weniger Fehler)")
    print(f"  {(time.time()-t0)/n*1000:.0f} ms je Aufnahme")


def main():
    ap = argparse.ArgumentParser()
    unter = ap.add_subparsers(dest="befehl", required=True)

    b = unter.add_parser("bauen")
    b.add_argument("--asr-dir", default="data/asr")
    b.add_argument("--wiki-dir", default="wiki")
    b.add_argument("--out", default="asr_woerter.json")
    b.add_argument("--groesse", type=int, default=120000)
    b.add_argument("--mindestens", type=int, default=3)
    b.add_argument("--wiki-bytes", type=int, default=120_000_000)
    b.add_argument("--ohne-wiki", action="store_true")

    t = unter.add_parser("testen")
    t.add_argument("--modell", default="ckpt_asr/ctc_gross.pt")
    t.add_argument("--woerterbuch", default="asr_woerter.json")
    t.add_argument("--dir", default="data/asr")
    t.add_argument("--daten", default="test")
    t.add_argument("--anzahl", type=int, default=100)
    t.add_argument("--max-abstand", type=int, default=2)
    t.add_argument("--device", default="auto")

    args = ap.parse_args()
    bauen(args) if args.befehl == "bauen" else testen(args)


if __name__ == "__main__":
    main()
