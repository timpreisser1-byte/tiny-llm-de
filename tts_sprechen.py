"""Text zu Sprache: die Bausteine wieder zusammensetzen.

Das Gegenstueck zu tts_schneiden.py. Der Text wird in Laute zerlegt, fuer
jedes Lautpaar das passende Stueck geholt und die Stuecke aneinandergehaengt.

Zwei Kunstgriffe halten die Naehte leise:

  Ueberblenden   die letzten und ersten paar Millisekunden zweier Stuecke
                 werden ineinander geblendet statt hart gestossen
  Angleichen     jedes Stueck wird auf eine gemeinsame Lautstaerke gebracht,
                 sonst schwankt die Stimme von Silbe zu Silbe

Fehlt ein Uebergang, wird auf den naechstbesten ausgewichen: erst derselbe
Anfangslaut mit beliebigem Folgelaut, dann Stille. Bei 878 Stuecken fuer
neunundneunzig Prozent kommt das selten vor.

    python tts_sprechen.py "Guten Morgen, wie geht es dir?"
"""

import argparse
import os

import numpy as np

from asr_daten import RATE
from tts_bausteine import zerlege
from tts_laute import lautfolge, zu_lauten

BLENDE = 160                     # 10 ms bei 16 kHz
GRENZE = 40                      # Fenster, in dem der Randpegel gemessen wird


def lade(pfad="stimme/bausteine.npz"):
    """Sammlung laden. Neues Format hat mehrere Varianten je Uebergang."""
    from collections import defaultdict
    d = np.load(pfad)
    raus = defaultdict(list)
    for k in d.files:
        teile = k.split("|")
        raus[(teile[0], teile[1])].append(d[k])
    return dict(raus)


def rand_klang(x, links=True, n=480):
    """Klangbild am Rand eines Stuecks - 30 ms als Mel-Merkmale.

    Damit wird gemessen, ob zwei Stuecke zueinander passen. Ein Sprung im
    Klangbild an der Naht ist das, was man als verwaschenes Wort hoert:
    gemessen liegen die staerksten Naehte 49 Prozent ueber dem, was
    natuerliche Sprache zeigt.
    """
    from asr_daten import merkmale
    teil = x[:n] if links else x[-n:]
    if len(teil) < 512:
        teil = np.pad(teil, (0, 512 - len(teil)))
    m = merkmale(teil.astype(np.float32))
    return m.mean(axis=0) if len(m) else np.zeros(40, dtype=np.float32)


def waehle(bausteine, laute):
    """Aus mehreren Varianten je Uebergang die beste Kette suchen.

    Klassische Bausteinauswahl: jede Naht kostet so viel, wie sich das
    Klangbild dort aendert. Gesucht ist die Kette mit den kleinsten Kosten -
    das ist eine Wegsuche (Viterbi), kein gieriges Nacheinander, denn eine
    schlechte Wahl vorne kann hinten teuer werden.
    """
    folge = [hole_alle(bausteine, a, b) for a, b in zip(laute, laute[1:])]
    if not folge:
        return []
    kosten = [0.0] * len(folge[0])
    woher = []
    rechts = [rand_klang(v, links=False) for v in folge[0]]
    for stufe in folge[1:]:
        links = [rand_klang(v, links=True) for v in stufe]
        neu, zeiger = [], []
        for j, l in enumerate(links):
            beste, arg = 1e18, 0
            for i, r in enumerate(rechts):
                c = kosten[i] + float(np.abs(r - l).mean())
                if c < beste:
                    beste, arg = c, i
            neu.append(beste); zeiger.append(arg)
        kosten, woher = neu, woher + [zeiger]
        rechts = [rand_klang(v, links=False) for v in stufe]
    # zurueckverfolgen
    j = int(np.argmin(kosten))
    weg = [j]
    for zeiger in reversed(woher):
        j = zeiger[j]
        weg.append(j)
    weg.reverse()
    return [folge[k][j] for k, j in enumerate(weg)]


def hole_alle(bausteine, a, b):
    """Alle Varianten eines Uebergangs, oder Ersatz."""
    if (a, b) in bausteine:
        return bausteine[(a, b)]
    endet = [v for (x, y), k in bausteine.items() if y == a for v in k]
    beginnt = [v for (x, y), k in bausteine.items() if x == b for v in k]
    if endet and beginnt:
        li, re = max(endet, key=len), max(beginnt, key=len)
        return [np.concatenate([li[len(li)//2:], re[:len(re)//2]])]
    for (x, y), k in bausteine.items():
        if x == a:
            return [k[0]]
    return [np.zeros(int(0.04 * RATE), dtype=np.float32)]


def hole(bausteine, a, b):
    """Den Uebergang a->b holen, oder ihn aus Halbstuecken bauen.

    Gemessen ueber fuenf Saetze: jeder zehnte Uebergang fehlt in der Sammlung.
    Der alte Notbehelf nahm irgendeinen Baustein mit demselben ANFANGSLAUT -
    bei "A->m" also etwa "A->k". Das ist an dieser Stelle schlicht der falsche
    Laut und klingt wie ein Stolperer.

    Besser: aus zwei vorhandenen Stuecken zusammensetzen. Die zweite Haelfte
    eines Uebergangs, der auf `a` endet, plus die erste Haelfte eines
    Uebergangs, der mit `b` beginnt. Beide Haelften enthalten dann den
    richtigen Laut, nur die Verbindung dazwischen ist erfunden - und die
    liegt genau in der Mitte, wo das Signal am stabilsten ist.
    """
    if (a, b) in bausteine:
        return bausteine[(a, b)]

    endet_auf_a = [v for (x, y), v in bausteine.items() if y == a]
    beginnt_mit_b = [v for (x, y), v in bausteine.items() if x == b]
    if endet_auf_a and beginnt_mit_b:
        # die laengsten nehmen, die haben die stabilsten Mitten
        links = max(endet_auf_a, key=len)
        rechts = max(beginnt_mit_b, key=len)
        return np.concatenate([links[len(links)//2:], rechts[:len(rechts)//2]])

    for (x, y), v in bausteine.items():   # letzter Ausweg wie bisher
        if x == a:
            return v
    return np.zeros(int(0.04 * RATE), dtype=np.float32)



def grundfrequenz(x, tief=75, hoch=180):
    """Tonhoehe eines Stuecks, oder 0 bei stimmlos.

    Zwei Dinge waren noetig, damit die Zahl brauchbar ist:

      Suchbereich   75 bis 180 Hz statt 70 bis 320. Eine Maennerstimme liegt
                    bei 85 bis 155; alles darueber war ohnehin falsch.
      Oktavschutz   die Autokorrelation greift gern die zweite oder dritte
                    Oberwelle. Gemessen an einer durchgehenden Aufnahme lagen
                    18 Prozent der Werte ueber 200 Hz - bei EINER Stimme. Der
                    Gegenzug: erst die tiefste Periode nehmen, die noch nah am
                    Maximum liegt, statt einfach das Maximum.
    """
    if len(x) < RATE // tief * 2:
        return 0.0
    y = x - x.mean()
    if float(np.sqrt((y ** 2).mean())) < 1e-4:
        return 0.0
    k = np.correlate(y, y, mode="full")[len(y) - 1:]
    k = k / (k[0] + 1e-12)
    a, b = RATE // hoch, min(RATE // tief, len(k) - 1)
    if b <= a:
        return 0.0
    fenster = k[a:b]
    spitze = float(fenster.max())
    if spitze < 0.30:
        return 0.0
    # die LAENGSTE Periode, die noch 85 Prozent der Spitze erreicht - damit
    # gewinnt die Grundwelle gegen ihre Oberwellen
    gut = np.where(fenster >= 0.85 * spitze)[0]
    return RATE / (a + int(gut.max()))


def auf_tonhoehe(x, f0, ziel):
    """Stueck auf die Zieltonhoehe umrechnen.

    Durch Neuabtastung: hoeher heisst schneller, tiefer heisst langsamer.
    Das aendert nebenbei die Dauer - bei Stuecken von rund hundert
    Millisekunden faellt das kaum auf und ist den gewonnenen gleichmaessigen
    Stimmklang wert. Sauberer waere periodenweises Dehnen, aber das ist ein
    eigenes Verfahren.
    """
    if f0 <= 0 or ziel <= 0:
        return x
    faktor = float(np.clip(f0 / ziel, 0.72, 1.4))   # hoechstens eine Terz
    n = max(16, int(len(x) / faktor))
    return np.interp(np.linspace(0, len(x) - 1, n),
                     np.arange(len(x)), x).astype(np.float32)


def naht(links, rechts, suche=180):
    """Wo passen die beiden Stuecke am besten aneinander?

    Der eigentliche Grund, warum Bausteinsynthese ohne diesen Schritt nach
    Krach klingt: In stimmhaften Lauten schwingt die Stimme periodisch, rund
    hundertmal je Sekunde. Stoesst das Ende des einen Stuecks auf eine andere
    Phase als der Anfang des naechsten, knackt es an jeder Naht - und bei
    dreissig Naehten je Satz bleibt davon nichts Verstaendliches uebrig.

    Deshalb wird die Naht verschoben: das Ende von `links` wird gegen den
    Anfang von `rechts` korreliert, und geschnitten wird dort, wo beide am
    aehnlichsten schwingen. Das ist die einfache Form dessen, was in der
    Literatur pitch-synchrones Ueberlappen heisst.

    Rueckgabe: um wie viele Abtastwerte `rechts` nach hinten geschoben wird.
    """
    n = min(BLENDE, len(links), len(rechts))
    if n < 16 or len(rechts) < suche + n:
        return 0
    ende = links[-n:]
    ende = ende - ende.mean()
    beste, bester_wert = 0, -1e18
    for v in range(0, min(suche, len(rechts) - n)):
        stueck = rechts[v:v + n]
        stueck = stueck - stueck.mean()
        norm = float(np.sqrt((ende ** 2).sum() * (stueck ** 2).sum())) + 1e-9
        wert = float((ende * stueck).sum()) / norm
        if wert > bester_wert:
            bester_wert, beste = wert, v
    return beste


def randpegel(x, links=True):
    """Lautstaerke an einem Ende des Stuecks."""
    n = min(GRENZE, len(x))
    teil = x[:n] if links else x[-n:]
    return float(np.sqrt((teil ** 2).mean())) + 1e-9


def sprich(text, bausteine, ziel_lautstaerke=0.08, ziel_f0=0.0):
    """Bausteine aneinanderfuegen.

      Phase    die Naht wird dorthin geschoben, wo beide Stuecke gleich
               schwingen. Ohne das knackt jede der dreissig Naehte je Satz,
               und nichts ist zu verstehen - das war der grosse Sprung.
      Pegel    jedes Stueck wird auf einen FESTEN Zielwert gebracht, nicht
               auf seinen Vorgaenger. Die Kopplung an den Vorgaenger war ein
               Fehler: die Faktoren multiplizieren sich ueber dreissig Naehte
               auf, und der Satz sackte von 100 auf 7 Prozent Lautstaerke ab.
               Gegen die verbleibende Haerte hilft eine milde Angleichung -
               die Wurzel des Faktors statt des Faktors selbst.
    """
    laute = lautfolge(text)
    # Die Tonhoehenumrechnung ist wieder draussen. Gemessen brachte sie
    # nichts: die Streuung der Ausgabe sank nur von 29 auf 27 Hz, und eine
    # echte Aufnahme von Thorsten hat mit 27 Hz genau dieselbe. Natuerliche
    # Sprache schwankt eben so stark - der stoerende Eindruck kommt woanders
    # her. Der Hoervergleich mit und ohne war nicht zu unterscheiden.
    stuecke = []
    for s in waehle(bausteine, laute):
        s = s.astype(np.float32).copy()
        p = float(np.sqrt((s ** 2).mean())) + 1e-9
        # Wurzel statt voller Angleichung: laesst leise Laute leise bleiben,
        # ohne dass die Unterschiede den Satz zerreissen
        stuecke.append(s * float(np.sqrt(ziel_lautstaerke / p)))
    if not stuecke:
        return np.zeros(1, dtype=np.float32)
    raus = stuecke[0]
    for s in stuecke[1:]:
        # Die Nahtsuche darf NICHTS wegwerfen. Frueher stand hier s = s[v:],
        # und damit fiel an jeder der vierzig Naehte der Anfang des Bausteins
        # weg - gemessen 0,7 von 3,1 Sekunden, fast ein Viertel des Signals
        # und an jeder Stelle echter Lautinhalt. Stattdessen wird der
        # Ueberlappungsbereich um v verlaengert: was verschoben wird, wird
        # ueberblendet statt geloescht.
        v = naht(raus, s)
        n = min(BLENDE + v, len(raus), len(s))
        if n < 8:
            raus = np.concatenate([raus, s]); continue
        rampe = np.linspace(0, 1, n, dtype=np.float32)
        ueber = raus[-n:] * (1 - rampe) + s[:n] * rampe
        raus = np.concatenate([raus[:-n], ueber, s[n:]])
    spitze = float(np.abs(raus).max()) + 1e-9
    return (raus / spitze * 0.9).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--bausteine", default="stimme/bausteine.npz")
    ap.add_argument("--out", default="stimme/probe.wav")
    args = ap.parse_args()
    import soundfile as sf

    text = " ".join(args.text) or "Guten Morgen, wie geht es dir?"
    b = lade(args.bausteine)
    ton = sprich(text, b)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sf.write(args.out, ton, RATE)
    laute = zu_lauten(text)
    fehlend = 0
    kette = ["#"] + [l for w in laute.split() for l in zerlege(w)] + ["#"]
    for a, x in zip(kette, kette[1:]):
        fehlend += (a, x) not in b
    print(f"\n  {text}")
    print(f"  {laute}")
    print(f"  {len(ton)/RATE:.2f} s, {len(kette)-1} Übergänge, "
          f"{fehlend} davon ersetzt")
    print(f"  geschrieben: {args.out}\n")


if __name__ == "__main__":
    main()
