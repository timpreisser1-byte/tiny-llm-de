"""Ranglistenstufe einzeln pruefen - so, wie sie in der Kette antwortet
(mit Mehrschritt-Sperre), gezaehlt nur, wenn die Rangliste die Antwort gab."""
import json
import sys

from evaluation.run_chain import enthaelt
from knowledge.facts import Fakten


def main():
    f = Fakten()
    for s in sys.argv[1:] or ["superlatives"]:
        fragen = json.load(open(f"evaluation/questions/{s}.json", encoding="utf-8"))
        ok = feuer = 0
        for e in fragen:
            a = f.antwort(e["frage"])
            if not a or a != f.rangliste(e["frage"]):
                continue
            r = enthaelt(a, e["antwort"])
            feuer += 1
            ok += r
            if not r:
                print("  falsch:", e["frage"], "->", a, "| soll", e["antwort"][0])
        print(f"{s}: Rangliste antwortet {feuer}/{len(fragen)}, richtig {ok}")


if __name__ == "__main__":
    main()
