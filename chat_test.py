"""Der Chatbot am Stück: Frage rein, Antwort raus, gezählt wird die Antwort.

Bisher wurden die Glieder einzeln gemessen - die Suche für sich, das Lesen für
sich. Das sagt nichts darüber, ob das Gerät eine Frage beantwortet. Hier läuft
die ganze Kette, und gezählt wird nur, was hinten herauskommt:

    Frage -> BM25-Index auf der Karte -> Absatz -> Sprachmodell -> Antwort

Vier Zahlen kommen heraus, und die dritte ist die wichtigste:

    gefunden    die Antwort steht im geholten Absatz   -> misst den Index
    beantwortet die Antwort steht in der Ausgabe       -> misst die Kette
    gelesen     beantwortet, sofern gefunden           -> misst NUR das Modell
    erfunden    ein Name in der Ausgabe steht nirgends im Absatz

"erfunden" ist der Wert, der den auffälligsten Fehler dieses Modells fasst:

    Frage    Wann wurde Goethe geboren?
    Absatz   Johann Wolfgang Goethe wurde 1749 hier ... in Frankfurt geboren.
    Antwort  Märce von Bismarck wurde 1749 in Frankfurt geboren.

Zahl richtig, Ort richtig, Name frei erfunden. Genau das deckt sich mit dem
Lesetest: Zahlen 66 %, Namen kaum über dem Zufall.

    python chat_test.py --ckpt ckpt_8mb/sft_kurz.pt
"""

import argparse
import json
import re
import sys
import time

import torch

GRUEN, ROT, GELB, GRAU, FETT, AUS = ("\033[92m", "\033[91m", "\033[93m",
                                     "\033[90m", "\033[1m", "\033[0m")

# Fragen mit kurzer, nachprüfbarer Antwort. Mehrere Schreibweisen sind
# erlaubt, weil "330" und "330 Meter" dieselbe Antwort sind.
PRUEFUNG = [
    ("Wie hoch ist der Eiffelturm?",              ["330"]),
    ("Wie lang ist der Rhein?",                   ["1232", "1.232"]),
    ("Was ist die Hauptstadt von Bayern?",        ["München"]),
    ("Wann wurde Goethe geboren?",                ["1749"]),
    ("Wann wurde die Berliner Mauer gebaut?",     ["1961"]),
    ("Wer hat die Relativitätstheorie entwickelt?", ["Einstein"]),
    ("Wie tief ist der Bodensee?",                ["251", "254"]),
    ("Wo liegt Quedlinburg?",                     ["Harz", "Sachsen-Anhalt"]),
    ("Wann fiel die Berliner Mauer?",             ["1989"]),
    ("Wie viele Bundesländer hat Deutschland?",   ["16", "sechzehn"]),
    ("Wann begann der Zweite Weltkrieg?",         ["1939"]),
    ("Wann endete der Zweite Weltkrieg?",         ["1945"]),
    ("Wer schrieb den Faust?",                    ["Goethe"]),
    ("Wie hoch ist die Zugspitze?",               ["2962", "2.962"]),
    ("Welcher Fluss fließt durch Hamburg?",       ["Elbe"]),
    ("Wann wurde die Bundesrepublik gegründet?",  ["1949"]),
    ("Wie heißt die Hauptstadt von Österreich?",  ["Wien"]),
    ("Wie lang ist die Donau?",                   ["2857", "2.857", "2850"]),
    ("Wer war der erste Bundeskanzler?",          ["Adenauer"]),
    ("Wann wurde die Titanic versenkt?",          ["1912"]),
    ("Wie viele Einwohner hat Berlin?",           ["3,7", "3.6", "3,6", "Millionen"]),
    ("Was ist ein Transistor?",                   ["Halbleiter"]),
    ("Wann war die Französische Revolution?",     ["1789"]),
    ("Wer komponierte die Neunte Sinfonie?",      ["Beethoven"]),
    ("Wie hoch ist der Mount Everest?",           ["8848", "8.848", "8849"]),
    ("Welche Währung gilt in Deutschland?",       ["Euro"]),
    ("Wann wurde der Kölner Dom fertiggestellt?", ["1880"]),
    ("Wer erfand die Buchdruckerkunst?",          ["Gutenberg"]),
    ("Wie groß ist die Fläche Deutschlands?",     ["357"]),
    ("Wann wurde die DDR gegründet?",             ["1949"]),
]


def norm(s):
    return re.sub(r"[^\wäöüß.,]+", " ", s.lower()).strip()


ZAHL = re.compile(r"(\d[\d.]*(?:,\d+)?)\s*(Milliarden|Millionen|Mio\.?|Mrd\.?)?")


def zahlen(text):
    """'3,16 Millionen' -> 3160000.0, '1.232' -> 1232.0 (Punkt = Tausender)."""
    raus = []
    for z, gross in ZAHL.findall(text):
        try:
            w = float(z.replace(".", "").replace(",", "."))
        except ValueError:
            continue
        if gross:
            w *= 1e9 if gross.startswith(("Milliarden", "Mrd")) else 1e6
        raus.append(w)
    return raus


def enthaelt(text, kandidaten):
    # Nur ganze Woerter: vorher zaehlte "acht" in "Nacht", "Rom" in "Strom"
    # und "16" in "2016" als Treffer. Gefunden vom Gutachter am 25.09.; mit
    # Wortgrenze sinkt z. B. Suche@30 auf MKQA von 28,3 auf 23,6 %.
    t = norm(text)
    for k in kandidaten:
        n = norm(str(k))
        if n and re.search(r"(?<![\wäöüß])" + re.escape(n) + r"(?![\wäöüß])", t):
            return True
    # Grosse Zahlen mit 3 % Spielraum: MKQA erwartet "3161000", die
    # Faktenstufe sagt "3,16 Millionen". Jahreszahlen bleiben exakt, dort
    # waere 3 % ein halbes Jahrhundert.
    gesucht = []
    for k in kandidaten:
        try:
            w = float(str(k).replace(",", "."))
        except ValueError:
            continue
        if w >= 10000 or (w >= 100 and w != int(w)):
            gesucht.append(w)
    return bool(gesucht) and any(abs(g - w) <= 0.03 * g
                                 for g in gesucht for w in zahlen(text))


# Großgeschriebene Wörter, die keine Satzanfänge sind - die Kandidaten für
# erfundene Namen. Häufige Funktionswörter werden ausgenommen, sonst zählt
# jedes deutsche Substantiv mit.
GROSS = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-ZÄÖÜ][a-zäöüß]{3,}\b")
# Masseinheiten, die eine Antwort anhaengen darf, ohne dass sie im Absatz
# woertlich vorkommen muessen: "1,1 Quadratkilometer" ist keine Erfindung,
# wenn dort "1,1 km²" steht. Ohne diese Liste meldet der Pruefer 18,5 %
# Erfindungen bei einem Verfahren, das per Konstruktion nichts erfinden kann.
HARMLOS = {"Meter", "Kilometer", "Quadratkilometer", "Quadratmeter", "Hektar",
           "Jahre", "Jahren", "Jahr", "Millionen", "Milliarden", "Einwohner",
           "Einwohnern", "Stadt", "Land", "Nähe", "Abschnitt", "Fluss", "Berg",
           "Text", "Prozent", "Grad"}


def erfundene(antwort, absatz):
    """Namen in der Antwort, die im Absatz nirgends vorkommen."""
    return sorted({w for w in GROSS.findall(antwort)
                   if w not in HARMLOS and w.lower() not in absatz.lower()})


def balken(i, n, t0, text=""):
    b = 22
    v = int(b * i / max(n, 1))
    sys.stderr.write(f"\r  [{'█'*v}{'░'*(b-v)}] {i:>3}/{n}  "
                     f"{(time.time()-t0)/max(i,1):.1f} s/Frage  {text:<26}")
    sys.stderr.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--tokenizer", default="tokenizer/de_bpe.json")
    ap.add_argument("--wissen", default="wiki_gestuft")
    ap.add_argument("--absaetze", type=int, default=2)
    ap.add_argument("--zeichen", type=int, default=1000)
    ap.add_argument("--laenge", type=int, default=60)
    ap.add_argument("--strafe", type=float, action="append",
                    help="Wiederholungsstrafe, mehrfach moeglich")
    ap.add_argument("--zeigen", action="store_true", help="jede Antwort ausgeben")
    ap.add_argument("--fragen", help="JSON aus chat_fragen.py statt der festen Liste")
    ap.add_argument("--reihenfolge", default="wissen",
                    choices=["wissen", "frage"])
    ap.add_argument("--orakel", action="store_true",
                    help="den richtigen Absatz vorlegen - misst NUR das Modell")
    ap.add_argument("--kopieren", action="store_true",
                    help="Antwort muss woertlich aus dem Absatz stammen")
    ap.add_argument("--min-laenge", type=int, action="append",
                    help="kuerzeste Kopie, bevor EOS erlaubt ist")
    ap.add_argument("--regeln", action="store_true",
                    help="Codex' regelbasierte Extraktion statt des Modells")
    ap.add_argument("--regeln2", action="store_true",
                    help="verbreiterte Regeln + Satz-Rueckfall")
    ap.add_argument("--hybrid", action="store_true",
                    help="Regel zuerst, Modell nur wenn sie nicht zuendet")
    ap.add_argument("--spanne", action="store_true",
                    help="beste Spanne des Absatzes als Antwort")
    ap.add_argument("--alpha", type=float, action="append",
                    help="Laengenausgleich der Spannenbewertung")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kette", action="store_true",
                    help="Geraete-Kette: Fakten -> praezise Regel -> Modell -> Satz-Rueckfall")
    ap.add_argument("--fakten", action="store_true",
                    help="Wikidata-Tabellen zuerst (fakten.py), dann der Rest")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    from geraet import antworte, antworte_kopie, antworte_spanne, lade
    from regeln_breit import antwort as breit_antwort
    from esp32_sd.extractive_answer import (answer_from_text as regel_antwort,
                                            has_typed_answer as regel_greift)
    from train import pick_device
    from wissen_index import Wissensbasis

    device = pick_device(args.device)
    tok = Tokenizer.from_file(args.tokenizer)
    if args.fragen:
        roh = json.load(open(args.fragen, encoding="utf-8"))
        pruefung = [(e["frage"], e["antwort"]) for e in roh]
        # Nur bauen, wenn --orakel gesetzt ist: ein Fragensatz aus echten,
        # von Hand gestellten Fragen hat keinen hinterlegten Absatz.
        orakel = [e.get("absatz", "") for e in roh] if args.orakel else []
    else:
        pruefung, orakel = PRUEFUNG, [None] * len(PRUEFUNG)
    wb = None if args.orakel else Wissensbasis(args.wissen)
    fk = None
    if args.fakten:
        from fakten import Fakten
        fk = Fakten()
    quelle = "Orakel-Absatz" if args.orakel else f"{args.absaetze} Absätze je Frage"
    print(f"\n  {len(pruefung)} Fragen, {quelle}\n")

    kopf = (f"  {'Modell':<28}{'gefunden':>10}{'beantwortet':>13}"
            f"{'gelesen':>10}{'erfunden':>10}{'abgelehnt':>11}")
    print(FETT + kopf + AUS)

    for pfad in (args.ckpt or ["regeln"]):
      netz, cfg = (None, None) if (args.regeln or args.regeln2) else lade(pfad, device)
      n_regel = n_modell = 0
      for strafe in ((args.alpha if args.spanne else None)
                     or (args.min_laenge if args.kopieren else None)
                     or args.strafe or [1.15]):
          n_gef = n_ant = n_erf = n_abl = 0
          zeilen, t0 = [], time.time()
          for i, (frage, soll) in enumerate(pruefung, 1):
              # Orakel: der richtige Absatz wird vorgelegt. Dann misst der
              # Lauf NUR das Modell - die Suche kann nichts mehr verderben.
              # Die Suche liefert None, wenn kein Wort der Frage im Index
              # steht - das ist ein gueltiges Ergebnis, kein Fehler.
              absatz = (orakel[i - 1][:args.zeichen] if args.orakel else
                        wb.bester_text(frage, max_zeichen=args.zeichen,
                                       absaetze=args.absaetze)) or ""
              fakt = fk.antwort(frage) if fk else None
              if fakt:
                  # Tabellenwert: die Quelle ist die Tabelle, nicht der Absatz
                  aus = absatz = fakt
                  n_regel += 1
              elif args.kette:
                  # So soll das Geraet antworten (25.09.): die praezise Regel
                  # nur, wenn sie eine Antwort der gefragten Art findet; sonst
                  # liest das Modell. Erst wenn es ablehnt, der Satz-Rueckfall.
                  # Im alten Hybrid kam das Modell nie zum Zug, weil der
                  # Satz-Rueckfall der Regeln fast immer etwas lieferte.
                  aus = breit_antwort(frage, absatz)
                  if aus:
                      n_regel += 1
                  else:
                      aus = antworte(netz, tok, cfg, device, frage, absatz,
                                     args.laenge, strafe, args.reihenfolge)
                      n_modell += 1
                      if not aus.strip() or "nicht im text" in aus.lower() or \
                              "erwähnt das nicht" in aus.lower() or "nichts" in aus.lower():
                          aus = regel_antwort(frage, absatz) or aus
              elif args.regeln2:
                  # verbreiterte Auslöser zuerst, sonst Codex' Satz-Rückfall
                  aus = breit_antwort(frage, absatz) or regel_antwort(frage, absatz) or ""
              elif args.regeln:
                  # Regeln brauchen kein Modell - reine Textmuster ueber den Absatz
                  aus = regel_antwort(frage, absatz) or ""
              elif args.hybrid and (regel := regel_antwort(frage, absatz)):
                  # Erst die Regel FRAGEN, statt vorher zu raten, ob sie
                  # zuendet. Gemessen am 18.09.: der Waechter has_typed_answer
                  # liess sie nur bei 23 % der Fragen zu, waehrend sie im
                  # reinen Regelbetrieb 41,7 % beantwortete - der Hybrid
                  # verschenkte so jede fuenfte Frage ans Modell, das darauf
                  # erfand. Die Regel kann per Konstruktion nichts erfinden;
                  # liefert sie etwas, gibt es keinen Grund fuers Modell.
                  aus = regel
                  n_regel += 1
              else:
                aus = (antworte_spanne(netz, tok, cfg, device, frage, absatz,
                                       alpha=strafe) if args.spanne
                       else antworte_kopie(netz, tok, cfg, device, frage, absatz,
                                      min(args.laenge, 24), int(strafe))
                       if args.kopieren
                       else antworte(netz, tok, cfg, device, frage, absatz,
                                     args.laenge, strafe, args.reihenfolge))
              gef = enthaelt(absatz, soll)
              ant = enthaelt(aus, soll)
              erf = erfundene(aus, absatz)
              abl = "erwähnt das nicht" in aus.lower() or "weiß ich nicht" in aus.lower()
              n_gef += gef; n_ant += ant; n_erf += bool(erf); n_abl += abl
              zeilen.append((frage, soll, gef, ant, erf, abl, aus))
              balken(i, len(pruefung), t0, f"{n_ant}/{i} beantwortet")
          sys.stderr.write("\r" + " " * 90 + "\r")
          n = len(pruefung)
          gelesen = n_ant / n_gef if n_gef else 0.0
          # Ordner mitnehmen: sonst heissen zwoelf Modelle "sft.pt"
          name = "/".join(pfad.rstrip("/").split("/")[-2:])
          zusatz = f"  Regel {n_regel/n*100:.0f} %" if (args.hybrid or fk) else ""
          if args.kette:
              zusatz += f"  Modell {n_modell/n*100:.0f} %"
          print(f"  {(name[-21:] + ' ' + str(strafe)):<28}"
                f"{n_gef/n*100:>9.1f} %"
                f"{n_ant/n*100:>12.1f} %{gelesen*100:>9.1f} %"
                f"{n_erf/n*100:>9.1f} %{n_abl/n*100:>10.1f} %{zusatz}")

          if args.zeigen:
              print()
              for frage, soll, gef, ant, erf, abl, aus in zeilen:
                  marke = (GRUEN + "✓" + AUS if ant else
                           (GELB + "~" + AUS if gef else ROT + "×" + AUS))
                  print(f"  {marke} {frage}")
                  print(f"    {GRAU}soll {'/'.join(soll)}"
                        f"{'  im Absatz' if gef else '  NICHT im Absatz'}"
                        f"{'  erfunden: ' + ', '.join(erf[:3]) if erf else ''}{AUS}")
                  print(f"    {aus[:150]}")
              print()


if __name__ == "__main__":
    main()
