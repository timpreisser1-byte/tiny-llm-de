"""stufe_genau.py — Genauigkeit der Faktenstufe allein.

Die Kettenquote in chat_test.py zaehlt nur, wie oft die Antwort stimmt. Eine
falsche Tabellenantwort faellt dort nicht auf: "erfunden" prueft Namen gegen
den Absatz, und bei der Faktenstufe IST die Antwort der Absatz. Deshalb hier
getrennt: wie oft antwortet die Stufe, und wie oft stimmt es.

Eingefuehrt am 25.09.2026: auf Mintaka-dev stimmten anfangs nur 30 % der
Faktenantworten (Jahr auf "In welchem Staat ...?", Mehrschrittfragen).

    python stufe_genau.py echt hand neu60 mkqa_dev mintaka_dev
    python stufe_genau.py mkqa_dev --falsch 20       # die ersten 20 Fehler zeigen
"""
import argparse
import json

from chat_test import enthaelt
from fakten import Fakten


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("saetze", nargs="+", help="Namen wie mkqa_dev -> chat_fragen_mkqa_dev.json")
    ap.add_argument("--falsch", type=int, default=6, help="so viele Fehler je Satz zeigen")
    args = ap.parse_args()
    f = Fakten()
    for s in args.saetze:
        fragen = json.load(open(f"chat_fragen_{s}.json", encoding="utf-8"))
        feuer = ok = 0
        falsch = []
        for e in fragen:
            soll = e["antwort"] if isinstance(e["antwort"], list) else [e["antwort"]]
            a = f.antwort(e["frage"])
            if not a:
                continue
            feuer += 1
            if enthaelt(a, soll):
                ok += 1
            else:
                falsch.append((e["frage"][:60], a[:40], soll[0]))
        print(f"{s:12} antwortet {feuer:3}/{len(fragen)}  richtig {ok:3}  "
              f"Genauigkeit {100 * ok / max(feuer, 1):4.0f} %")
        for x in falsch[:args.falsch]:
            print("      falsch:", x)


if __name__ == "__main__":
    main()
