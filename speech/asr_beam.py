"""Phase 4, richtig: Strahlsuche mit Wörterbuch während der Dekodierung.

Die nachträgliche Wortkorrektur brachte nur 4 Prozentpunkte. Der Grund liegt
in den Beispielen:

    soll   erfindungsgabe
    roh    erfindung egabe      <- Wortgrenze falsch, Korrektur sieht das nicht
    soll   kenntnissen
    roh    kenntnichen          <- wird zu "kennzeichen", auch ein echtes Wort

Beides sind Fehler, die man nur vermeiden kann, solange die
Wahrscheinlichkeiten noch da sind. Die gierige Dekodierung wirft sie weg: Sie
nimmt pro Zeitschritt den besten Buchstaben und vergisst alle Alternativen.

Die Strahlsuche behält stattdessen die besten Teilergebnisse gleichzeitig und
laesst nur Buchstabenfolgen zu, die im Woerterbuch weiterfuehren. Aus 34
moeglichen Buchstaben werden dadurch oft nur zwei oder drei - der Rest fuehrt
in kein deutsches Wort.

Der Baum (Trie) ist dabei das Kernstueck: Nach "erfindun" gibt es nur noch
"g" als moegliche Fortsetzung, nach "erfindung" kann ein Wortende folgen -
oder ein "s", weil "erfindungsgabe" auch im Woerterbuch steht.

    python -m speech.asr_beam --anzahl 40 --breite 24
"""

import argparse
import json
import math
import time
from collections import defaultdict

import numpy as np
import torch

from speech.asr_data import ZEICHEN

BLANK = 0
LEER = ZEICHEN.index(" ") + 1        # Index des Leerzeichens im Alphabet


class Baum:
    """Praefixbaum ueber alle Woerter. Knoten sind verschachtelte dicts."""

    ENDE = "#"

    def __init__(self, woerter: dict, gesamt: int):
        self.wurzel = {}
        self.gesamt = gesamt
        for wort, anzahl in woerter.items():
            knoten = self.wurzel
            for zeichen in wort:
                knoten = knoten.setdefault(zeichen, {})
            # Wortende mit Haeufigkeit - daraus wird die Sprachmodellbewertung
            knoten[self.ENDE] = math.log(anzahl / gesamt)

    def kinder(self, knoten):
        return [z for z in knoten if z != self.ENDE]

    def ist_wortende(self, knoten):
        return self.ENDE in knoten

    def bewertung(self, knoten):
        return knoten.get(self.ENDE, -20.0)


def logsumexp(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hoch = max(a, b)
    return hoch + math.log(math.exp(a - hoch) + math.exp(b - hoch))


def strahlsuche(logits, baum, breite=24, kandidaten=6, wort_bonus=2.0,
                lm_gewicht=0.8, nbest=1):
    """CTC-Strahlsuche mit Woerterbuchzwang.

    logits: (T, n_zeichen) als log-Wahrscheinlichkeiten
    Rueckgabe: der beste Text

    Jeder Strahl merkt sich: die fertigen Woerter, das angefangene Wort, den
    Knoten im Baum und zwei Wahrscheinlichkeiten - eine fuer Pfade, die auf
    Leerzeichen enden, eine fuer Pfade, die auf einem Buchstaben enden. Diese
    Trennung ist der Kern von CTC: nur so lassen sich doppelte Buchstaben von
    wiederholten Zeitschritten unterscheiden.
    """
    wurzel = baum.wurzel
    # Schluessel: (fertige Woerter, angefangenes Wort) -> [p_blank, p_zeichen, knoten, lm]
    strahlen = {((), ""): [0.0, -math.inf, wurzel, 0.0]}

    T = logits.shape[0]
    for t in range(T):
        rahmen = logits[t]
        # nur die aussichtsreichsten Buchstaben betrachten
        beste = np.argpartition(-rahmen, kandidaten)[:kandidaten]
        neu = defaultdict(lambda: [-math.inf, -math.inf, None, 0.0])

        for (woerter, teil), (p_b, p_z, knoten, lm) in strahlen.items():
            gesamt = logsumexp(p_b, p_z)

            # 1) Leerzeichen im CTC-Sinn: Pfad bleibt, wie er ist
            e = neu[(woerter, teil)]
            e[0] = logsumexp(e[0], gesamt + float(rahmen[BLANK]))
            e[2], e[3] = knoten, lm

            for c in beste:
                c = int(c)
                if c == BLANK:
                    continue
                p_c = float(rahmen[c])
                zeichen = ZEICHEN[c - 1]

                # 2) Wortgrenze
                if zeichen == " ":
                    if not teil or not baum.ist_wortende(knoten):
                        continue                     # kein gueltiges Wort
                    schluessel = (woerter + (teil,), "")
                    e = neu[schluessel]
                    e[1] = logsumexp(e[1], gesamt + p_c)
                    e[2] = wurzel
                    e[3] = lm + lm_gewicht * baum.bewertung(knoten) + wort_bonus
                    continue

                # 3) derselbe Buchstabe nochmal: entweder Wiederholung im
                #    selben Zeitschritt oder ein echter Doppelbuchstabe
                if teil and zeichen == teil[-1]:
                    e = neu[(woerter, teil)]
                    e[1] = logsumexp(e[1], p_z + p_c)   # Wiederholung
                    e[2], e[3] = knoten, lm
                    if zeichen in knoten:               # echter Doppelbuchstabe
                        s = (woerter, teil + zeichen)
                        e2 = neu[s]
                        e2[1] = logsumexp(e2[1], p_b + p_c)
                        e2[2], e2[3] = knoten[zeichen], lm
                    continue

                # 4) neuer Buchstabe - nur wenn er im Woerterbuch weiterfuehrt
                if zeichen not in knoten:
                    continue
                s = (woerter, teil + zeichen)
                e = neu[s]
                e[1] = logsumexp(e[1], gesamt + p_c)
                e[2], e[3] = knoten[zeichen], lm

        # beschneiden: nur die besten Strahlen weiterverfolgen
        bewertet = sorted(neu.items(),
                          key=lambda kv: -(logsumexp(kv[1][0], kv[1][1]) + kv[1][3]))
        strahlen = {k: v for k, v in bewertet[:breite]}

    # Am Ende zaehlt nur, was ein vollstaendiges Wort ergibt
    fertig = []
    for (woerter, teil), (p_b, p_z, knoten, lm) in strahlen.items():
        bewertung = logsumexp(p_b, p_z) + lm
        volltext = list(woerter)
        if teil:
            if not baum.ist_wortende(knoten):
                continue
            volltext.append(teil)
            bewertung += lm_gewicht * baum.bewertung(knoten) + wort_bonus
        fertig.append((" ".join(volltext), bewertung))
    fertig.sort(key=lambda tb: -tb[1])
    if nbest > 1:
        # fuer die Nachbewertung durch das Sprachmodell: mehrere Kandidaten
        # samt akustischer Bewertung, damit sich beides verrechnen laesst
        return fertig[:nbest] or [("", -math.inf)]
    return fertig[0][0] if fertig else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modell", default="ckpt_asr/ctc_gross.pt")
    ap.add_argument("--woerterbuch", default="asr_woerter.json")
    ap.add_argument("--dir", default="data/asr")
    ap.add_argument("--daten", default="test")
    ap.add_argument("--anzahl", type=int, default=40)
    ap.add_argument("--breite", type=int, default=24)
    ap.add_argument("--kandidaten", type=int, default=6)
    ap.add_argument("--wort-bonus", type=float, default=2.0)
    ap.add_argument("--lm-gewicht", type=float, default=0.8)
    ap.add_argument("--woerter", type=int, default=60000,
                    help="wie viele der haeufigsten Woerter in den Baum")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from speech.asr_model import Sprachdaten, gierig, lade_modell, wortfehlerrate
    from training.pretrain import pick_device

    device = pick_device(args.device)
    netz = lade_modell(args.modell, device)
    daten = Sprachdaten(args.dir, args.daten)

    with open(args.woerterbuch, encoding="utf-8") as f:
        wb = json.load(f)
    paare = sorted(wb["woerter"].items(), key=lambda x: -x[1])[:args.woerter]
    baum = Baum(dict(paare), sum(n for _, n in paare))
    print(f"{len(paare):,} Wörter im Baum, Strahlbreite {args.breite}\n")

    gierig_summe = strahl_summe = 0.0
    beispiele, t0 = [], time.time()
    with torch.no_grad():
        for i in range(min(args.anzahl, len(daten))):
            m, _, soll = daten.hole(i)
            x = torch.from_numpy(m).T[None].to(device)
            ausgabe = netz(x)[:, 0]
            g = gierig(ausgabe)
            s = strahlsuche(ausgabe.cpu().numpy(), baum, args.breite,
                            args.kandidaten, args.wort_bonus, args.lm_gewicht)
            gierig_summe += wortfehlerrate(soll, g)
            strahl_summe += wortfehlerrate(soll, s)
            if len(beispiele) < 3:
                beispiele.append((soll, g, s))
            print(f"\r  {i+1}/{args.anzahl}", end="", flush=True)
    n = min(args.anzahl, len(daten))
    print(f"\r{' '*20}\r")

    for soll, g, s in beispiele:
        print(f"  soll  : {soll[:76]}")
        print(f"  gierig: {g[:76]}")
        print(f"  Strahl: {s[:76]}\n")

    gw, sw = gierig_summe / n * 100, strahl_summe / n * 100
    print(f"{'':24}{'Wortfehlerrate':>16}")
    print(f"  {'gierig':22}{gw:14.1f} %")
    print(f"  {'Strahl + Wörterbuch':22}{sw:14.1f} %")
    print(f"\n  Verbesserung: {gw-sw:+.1f} Prozentpunkte "
          f"({(1-sw/max(gw,1e-9))*100:.0f} % weniger Fehler)")
    print(f"  {(time.time()-t0)/n*1000:.0f} ms je Aufnahme")


if __name__ == "__main__":
    main()
