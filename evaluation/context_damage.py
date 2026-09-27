#!/usr/bin/env python3
"""evaluation/context_damage.py — nimmt der vorgelegte Absatz dem Modell Antworten weg?

Die Arbeit "Can Small Language Models Use What They Retrieve?" (arXiv
2603.11513, 360M-8B, drei Modellfamilien) berichtet, dass das Hinzufuegen
eines gefundenen Absatzes **42 bis 100 % der Antworten zerstoert, die das
Modell vorher wusste**. Ein Gegenmittel nennt sie nicht.

Fuer ein Geraet mit 8 MB Flash ist das eine bare Muenze: wenn der Absatz mehr
kaputt macht als er hilft, gehoert er bei manchen Fragearten weggelassen.

    python -m evaluation.context_damage --ckpt ckpt_gpu/bester_sft2.pt
"""
import argparse
import json
import re


def norm(s):
    return re.sub(r"[^\wäöüß.,]+", " ", s.lower()).strip()


def drin(text, kandidaten):
    if not text:
        return False
    t = norm(text)
    return any(norm(k) in t for k in
               (kandidaten if isinstance(kandidaten, list) else [kandidaten]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--fragen", default="evaluation/questions/hand.json")
    ap.add_argument("--wissen", default="wiki")
    ap.add_argument("--zeichen", type=int, default=1000)
    ap.add_argument("--absaetze", type=int, default=2)
    ap.add_argument("--strafe", type=float, default=1.05)
    ap.add_argument("--tokenizer", default="tokenizer/de_bpe.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from lm.inference import lade, antworte
    from tokenizers import Tokenizer
    from knowledge.search_index import Wissensbasis
    tok = Tokenizer.from_file(args.tokenizer)
    wb = Wissensbasis(args.wissen)
    fragen = json.load(open(args.fragen, encoding="utf-8"))

    F, G, R, D, A = "\033[1m", "\033[32m", "\033[31m", "\033[90m", "\033[0m"
    print(f"\n  {len(fragen)} Fragen · mit und ohne vorgelegten Absatz\n")
    print(F + f"  {'Modell':<26}{'ohne':>8}{'mit':>8}{'nur ohne':>11}"
          f"{'nur mit':>10}{'zerstört':>11}" + A)

    for pfad in args.ckpt:
        netz, cfg = lade(pfad, args.device)
        ohne = mit = nur_ohne = nur_mit = 0
        for e in fragen:
            a1 = antworte(netz, tok, cfg, args.device, e["frage"], None,
                          strafe=args.strafe)
            absatz = wb.bester_text(e["frage"], max_zeichen=args.zeichen,
                                    absaetze=args.absaetze)
            a2 = antworte(netz, tok, cfg, args.device, e["frage"], absatz,
                          strafe=args.strafe)
            r1, r2 = drin(a1, e["antwort"]), drin(a2, e["antwort"])
            ohne += r1; mit += r2
            nur_ohne += (r1 and not r2)
            nur_mit += (r2 and not r1)
        n = len(fragen)
        anteil = nur_ohne / ohne * 100 if ohne else 0.0
        farbe = R if anteil >= 42 else G
        print(f"  {pfad.split('/')[-1]:<26}{ohne/n*100:>7.1f}%{mit/n*100:>7.1f}%"
              f"{nur_ohne:>10}{nur_mit:>10}{farbe}{anteil:>10.0f}%{A}")

    print(f"\n  {D}'zerstört' = Anteil der ohne Absatz richtigen Antworten, die"
          f" mit Absatz falsch werden.{A}")
    print(f"  {D}Literatur meldet 42-100 % bei Modellen von 360M bis 8B.{A}\n")


if __name__ == "__main__":
    main()
