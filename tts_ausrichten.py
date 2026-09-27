"""Wo liegt welcher Laut? Ausrichtung mit dem eigenen Erkenner.

Um aus Aufnahmen Lautbausteine zu schneiden, braucht man Zeitmarken: ab
welcher Millisekunde klingt das "a", wo endet es. Ueblicherweise nimmt man
dafuer ein eigenes Ausrichtungswerkzeug - hier reicht der CTC-Erkenner, den
das Projekt ohnehin hat.

Der Trick: CTC gibt je Zeitschritt eine Wahrscheinlichkeit fuer jedes
Zeichen. Beim Erkennen sucht man die beste Zeichenfolge. Bei bekannter
Zeichenfolge sucht man stattdessen den besten *Weg* durch die Zeit - welcher
Zeitschritt gehoert zu welchem Buchstaben. Das ist dieselbe Rechnung
rueckwaerts gelesen und heisst erzwungene Ausrichtung.

Der Weg fuehrt durch eine erweiterte Folge, in der zwischen je zwei Zeichen
ein Leerzeichen steht:

    _ h _ a _ l _ l _ o _

In jedem Schritt darf man stehenbleiben, ein Feld weiter gehen, oder - wenn
das naechste Zeichen ein anderes ist - zwei Felder ueberspringen. Viterbi
sucht darin den wahrscheinlichsten Weg.

    python tts_ausrichten.py --pruefen
"""

import argparse
import sys

import numpy as np
import torch

from asr_daten import ZEICHEN, ZU_INDEX, SPRUNG, RATE

BLANK = 0
RAHMEN_MS = SPRUNG / RATE * 1000 * 2       # Schrittweite 2 im Erkenner

# CTC darf die Zeichenausgabe verzoegern - die Verlustfunktion erzwingt nur
# die Reihenfolge, nicht den Zeitpunkt. Gemessen an drei Aufnahmen liegt die
# Ausrichtung 80 bis 140 ms zu spaet, im Mittel 105. Fuer die Erkennung ist
# das egal, fuers Schneiden von Lautbausteinen nicht: der Versatz ist groesser
# als ein ganzer Baustein (Median 80 ms), jedes Stueck enthaelt dann den
# falschen Laut.
VERSATZ_MS = 105.0


def erweitere(text):
    """"hallo" -> [_, h, _, a, _, l, _, l, _, o, _] als Indizes."""
    ids = [ZU_INDEX[z] for z in text if z in ZU_INDEX]
    raus = [BLANK]
    for i in ids:
        raus += [i, BLANK]
    return raus, ids


def richte_aus(logits, text):
    """logits (T, C) als log-Wahrscheinlichkeiten, text bekannt.

    Rueckgabe: Liste (zeichen, start_rahmen, ende_rahmen).
    """
    folge, _ = erweitere(text)
    T, S = logits.shape[0], len(folge)
    if T < S // 2:
        return []                       # Aufnahme zu kurz fuer den Text
    NEG = -1e30
    punkte = np.full((T, S), NEG, dtype=np.float64)
    woher = np.zeros((T, S), dtype=np.int8)
    punkte[0, 0] = logits[0, folge[0]]
    if S > 1:
        punkte[0, 1] = logits[0, folge[1]]
    for t in range(1, T):
        for s in range(S):
            beste, wo = punkte[t - 1, s], 0            # stehenbleiben
            if s > 0 and punkte[t - 1, s - 1] > beste:
                beste, wo = punkte[t - 1, s - 1], 1    # ein Feld weiter
            # zwei Felder ueberspringen: nur ueber ein Leerzeichen hinweg und
            # nur wenn die beiden Zeichen verschieden sind - sonst verschmelzen
            # doppelte Buchstaben wie in "hallo"
            if (s > 1 and folge[s] != BLANK and folge[s] != folge[s - 2]
                    and punkte[t - 1, s - 2] > beste):
                beste, wo = punkte[t - 1, s - 2], 2
            if beste <= NEG / 2:
                continue
            punkte[t, s] = beste + logits[t, folge[s]]
            woher[t, s] = wo
    # zurueckverfolgen vom besseren der beiden letzten Felder
    s = S - 1 if punkte[T - 1, S - 1] >= punkte[T - 1, S - 2] else S - 2
    weg = np.zeros(T, dtype=np.int32)
    for t in range(T - 1, -1, -1):
        weg[t] = s
        # int() ist noetig: woher ist int8, und s wuerde sonst auf int8
        # heruntergestuft werden und bei Folgen ueber 127 ueberlaufen
        s = int(s) - int(woher[t, s])
    # Felder zu Zeichen zusammenfassen, Leerzeichen ueberspringen
    raus, offen = [], None
    for t in range(T):
        s = weg[t]
        if folge[s] == BLANK:
            if offen is not None:
                raus.append(offen); offen = None
            continue
        z = ZEICHEN[folge[s] - 1]
        if offen is None or offen[0] != z or offen[3] != s:
            if offen is not None:
                raus.append(offen)
            offen = [z, t, t + 1, s]
        else:
            offen[2] = t + 1
    if offen is not None:
        raus.append(offen)
    return [(z, a, e) for z, a, e, _ in raus]


def pruefen(modell="ckpt_asr/ctc_gross.pt", anzahl=5):
    from asr_modell import Sprachdaten, lade_modell
    from train import pick_device
    device = pick_device("cpu")
    netz = lade_modell(modell, device)
    daten = Sprachdaten("data/asr", "test")
    print(f"\n  {RAHMEN_MS:.0f} ms je Rahmen\n")
    for i in range(anzahl):
        mel, _, text = daten.hole(i)
        with torch.no_grad():
            lg = netz(torch.from_numpy(mel.T).unsqueeze(0))[:, 0, :].numpy()
        marken = richte_aus(lg, text)
        if not marken:
            print(f"  {i}: zu kurz"); continue
        dauer = len(mel) * SPRUNG / RATE
        gedeckt = (marken[-1][2] - marken[0][1]) * RAHMEN_MS / 1000
        print(f"  {text[:52]:54}{dauer:5.1f} s Aufnahme, "
              f"{len(marken):3} Zeichen ausgerichtet, {gedeckt:.1f} s belegt")
        zeig = "  ".join(f"{z}@{a*RAHMEN_MS/1000:.2f}" for z, a, _ in marken[:9])
        print(f"    {zeig}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pruefen", action="store_true")
    ap.add_argument("--modell", default="ckpt_asr/ctc_gross.pt")
    ap.add_argument("--anzahl", type=int, default=5)
    args = ap.parse_args()
    pruefen(args.modell, args.anzahl)


if __name__ == "__main__":
    main()
