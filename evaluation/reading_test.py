"""Kann das Modell lesen? Die Zahl aus dem vorgelegten Absatz herausgreifen.

Das ist die Messung, auf die es beim Geraetekonzept ankommt. Wissen kommt von
der SD-Karte, nicht aus den Gewichten - also zaehlt nicht, was das Modell
auswendig weiss, sondern ob es im vorgelegten Text die richtige Stelle findet.

Aufbau: Aus der Wikipedia-Sammlung werden Absaetze gezogen, die mindestens
zwei verschiedene Zahlen enthalten. Aus einem Satz wird eine Zahl
herausgeschnitten, und das Modell soll sie ergaenzen. Bewertet wird nicht die
erzeugte Antwort - das waere unscharf -, sondern welcher von zwei Kandidaten
das Modell fuer wahrscheinlicher haelt: der richtigen Zahl oder einer anderen
Zahl aus demselben Absatz. Der Zufall liegt damit bei 50 %.

Drei Zahlen kommen heraus:

    mit Kontext    Absatz steht im Prompt - das ist der Ernstfall
    ohne Kontext   Absatz fehlt - misst, was reines Raten schafft
    erste Zahl     wie oft die erste Zahl des Absatzes bevorzugt wird,
                   unabhaengig davon ob sie stimmt. Das war die gemessene
                   Schwaeche der bisherigen Modelle.

    python -m evaluation.reading_test --ckpt ckpt_56m/sft_v5.pt --n 200
    python -m evaluation.reading_test --ckpt a.pt --ckpt b.pt --n 200
"""

import argparse
import random
import re
import sys
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from lm.chat_format import BOS, USER, ASSISTANT, EOS, mit_wissen
from lm.model import Config, TinyLM
from training.pretrain import pick_device

GRUEN, ROT, GRAU, FETT, AUS = "\033[92m", "\033[91m", "\033[90m", "\033[1m", "\033[0m"

# deutsche Zahlschreibweise: 1.200 / 24.000 / 38 / 12,5
ZAHL = re.compile(r"\b\d{1,3}(?:\.\d{3})+\b|\b\d+(?:,\d+)?\b")

# Eigennamen: grossgeschriebene Woerter, die nicht am Satzanfang stehen.
# Deutsch schreibt alle Substantive gross, deshalb greift die Regel auch
# "Haus" ab - das ist hier kein Schaden, gesucht wird ja nur, ob das Modell
# die richtige STELLE im Text findet, nicht ob es Namen erkennt.
NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-ZÄÖÜ][a-zäöüß]{3,}\b")


def balken(i, n, t0, text=""):
    breite = 28
    voll = int(breite * i / n)
    rest = (time.time() - t0) / max(i, 1) * (n - i)
    sys.stderr.write(f"\r  [{'█'*voll}{'░'*(breite-voll)}] {i:>4}/{n}"
                     f"  noch {rest/60:4.1f} min  {text:<24}")
    sys.stderr.flush()


def saetze(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def baue_aufgaben(pfad, n, rng, min_zeichen=250, max_zeichen=900, art="zahl"):
    """max_zeichen muss zum Kontextfenster passen - sonst schneidet der
    Test den Absatz vorne ab und misst Abschneiden statt Lesen."""
    """Absaetze in Lueckenaufgaben verwandeln."""
    aufgaben = []
    with open(pfad, encoding="utf-8") as f:
        zeilen = f.readlines()
    rng.shuffle(zeilen)
    for zeile in zeilen:
        if len(aufgaben) >= n:
            break
        absatz = zeile.split(": ", 1)[-1].strip()
        if not (min_zeichen <= len(absatz) <= max_zeichen):
            continue
        # Zahlen, die genau einmal vorkommen - sonst waere die Luecke mehrdeutig
        muster = ZAHL if art == "zahl" else NAME
        alle = muster.findall(absatz)
        # Listen und Tabellen aussortieren: dort ist "Satz" kein Satz
        if len(alle) > 8 or absatz.count(",") > 12:
            continue
        # Jahreschroniken ("15. November: ...") sind keine Prosa
        if re.match(r"^\d{1,2}\. \w+:", absatz):
            continue
        teile = saetze(absatz)
        if len(teile) < 2:
            continue
        einmalig = [z for z in dict.fromkeys(alle)
                    if absatz.count(z) == 1 and len(z) >= (2 if art == "zahl" else 4)]
        if len(einmalig) < 2:
            continue
        # Zielsatz: enthaelt eine der einmaligen Zahlen, aber nicht die erste
        # des Absatzes - sonst waere "immer die erste nehmen" schon richtig
        erste = einmalig[0]
        kandidaten = [(s, z) for s in teile for z in einmalig[1:]
                      if z in s and 40 <= len(s) <= 220 and s.count(",") <= 3]
        if not kandidaten:
            continue
        satz, richtig = kandidaten[0]
        # Ablenkung mit aehnlicher Stellenzahl - sonst ist die Wahl zu leicht.
        # Unter den passenden wird gewuerfelt: nimmt man immer die erste, steht
        # die Ablenkung fast immer VOR der Luecke, und die bekannte Vorliebe
        # der Modelle fuer die erste Zahl im Text druckt das Ergebnis unter
        # den Zufall - dann misst man die Testanlage statt das Modell.
        andere = [z for z in einmalig if z != richtig]
        if not andere:
            continue
        beste = min(abs(len(z) - len(richtig)) for z in andere)
        falsch = rng.choice([z for z in andere
                             if abs(len(z) - len(richtig)) == beste])
        luecke = satz.replace(richtig, "___", 1)
        aufgaben.append({"absatz": absatz, "frage": luecke,
                         "richtig": richtig, "falsch": falsch,
                         "erste": erste})
    return aufgaben


def lade(pfad, device):
    ck = torch.load(pfad, map_location="cpu", weights_only=False)
    cfg = Config(**ck["config"])
    netz = TinyLM(cfg)
    art = ck.get("art") or ("ternaer" if ck.get("ternaer") else None)
    if art:
        from lm.ternary import beschraenke
        beschraenke(netz, art)
    netz.load_state_dict(ck["model"])
    return netz.to(device).eval(), cfg


# Drei Einkleidungen derselben Aufgabe. Der Vergleich zwischen "roh" und
# "anders" trennt zwei Dinge, die sonst durcheinandergehen: hat ein Modell
# wirklich lesen gelernt, oder nur eine Vorlage wiedererkannt?
ERGAENZUNG = "Nenne nur die Zahl."

FORMEN = {
    # was ein feingetuntes Modell kennt
    "chat":   lambda a, f: (BOS + USER + f"Wissen: {a}\nFrage: {f} "
                            "Nenne nur die Zahl." + ASSISTANT),
    # genau die Vorlage aus training/reading_templates.py
    "roh":    lambda a, f: f"Wissen: {a}\nFrage: {f} {ERGAENZUNG}\nAntwort: ",
    # dieselbe Aufgabe, anders formuliert - niemand hat darauf trainiert
    "anders": lambda a, f: f"Text: {a}\nWelche Zahl fehlt hier? {f}\nEs ist: ",
    # Seit die Trainingsdaten achtzehn Formen enthalten, liegt "anders" zu
    # nah an einer davon. Diese hier steht in keiner Trainingsvorlage - weder
    # die Einleitung noch die Frage noch die Antwortmarke.
    "fremd":  lambda a, f: (f"Unten steht ein Abschnitt.\n{a}\n"
                            f"Welche Zahl gehört an die Stelle mit den Strichen? "
                            f"{f}\nDie Zahl: "),
}


@torch.no_grad()
def antwortverlust(netz, tok, cfg, device, vortext, antwort):
    """Mittlerer Verlust auf den Antwort-Token. Kleiner ist besser."""
    vor = tok.encode(vortext, add_special_tokens=False).ids
    ant = tok.encode(str(antwort).strip() + EOS, add_special_tokens=False).ids
    voll = vor + ant
    ids = voll[-cfg.block_size:]
    if len(voll) > cfg.block_size:
        antwortverlust.abgeschnitten += 1
    antwortverlust.gesamt += 1
    n_ant = min(len(ant), len(ids) - 1)
    if n_ant <= 0:
        return float("inf")
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    ziel = y.clone().fill_(-100)
    ziel[0, -n_ant:] = y[0, -n_ant:]
    _, v = netz(x, ziel)
    return v.item()


antwortverlust.abgeschnitten = 0
antwortverlust.gesamt = 0


def pruefe(netz, tok, cfg, device, aufgaben, name, form="chat"):
    bau = FORMEN[form]
    treffer = ohne = ersterang = 0
    t0 = time.time()
    for i, a in enumerate(aufgaben, 1):
        mit = bau(a["absatz"], a["frage"])
        # derselbe Aufbau, aber der Absatz fehlt - misst reines Raten
        blank = bau("", a["frage"])
        vr = antwortverlust(netz, tok, cfg, device, mit, a["richtig"])
        vf = antwortverlust(netz, tok, cfg, device, mit, a["falsch"])
        if vr < vf:
            treffer += 1
        # dieselbe Wahl ohne den Absatz - misst reines Raten
        orr = antwortverlust(netz, tok, cfg, device, blank, a["richtig"])
        orf = antwortverlust(netz, tok, cfg, device, blank, a["falsch"])
        if orr < orf:
            ohne += 1
        # zieht es die erste Zahl des Absatzes vor?
        if a["erste"] not in (a["richtig"], a["falsch"]):
            ve = antwortverlust(netz, tok, cfg, device, mit, a["erste"])
            if ve < vr:
                ersterang += 1
        if i % 5 == 0 or i == len(aufgaben):
            balken(i, len(aufgaben), t0, name)
    sys.stderr.write("\r" + " " * 90 + "\r")
    n = len(aufgaben)
    quote = antwortverlust.abgeschnitten / max(antwortverlust.gesamt, 1)
    antwortverlust.abgeschnitten = antwortverlust.gesamt = 0
    return treffer / n, ohne / n, ersterang / n, quote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--chunks", default="wiki/chunks.txt")
    ap.add_argument("--tokenizer", default="tokenizer/de_bpe.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--art", default="zahl", choices=["zahl", "name"],
                    help="wonach gefragt wird: eine Zahl oder ein Eigenname")
    ap.add_argument("--max-zeichen", type=int, default=900,
                    help="laengster Absatz; bei 256 Token Kontext hoechstens 500")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--form", default="chat", choices=list(FORMEN),
                    help="chat = feingetunt, roh = Vorlage aus training/reading_templates.py, "
                         "anders = umformuliert (prueft echtes Lesen)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = Tokenizer.from_file(args.tokenizer)
    rng = random.Random(args.seed)

    print(f"\n  Baue {args.n} Leseaufgaben aus {args.chunks} ...")
    aufgaben = baue_aufgaben(args.chunks, args.n, rng,
                             max_zeichen=args.max_zeichen, art=args.art)
    if len(aufgaben) < args.n:
        print(f"  nur {len(aufgaben)} Aufgaben gefunden")
    globals()["ERGAENZUNG"] = ("Nenne nur die Zahl." if args.art == "zahl"
                               else "Nenne nur das Wort.")
    print(f"  {len(aufgaben)} Aufgaben, Form '{args.form}', Art '{args.art}', "
          f"Zufall liegt bei 50 %\n")
    b = aufgaben[0]
    print(f"  {GRAU}Beispiel{AUS}")
    print(f"  {GRAU}Absatz : {b['absatz'][:110]}...{AUS}")
    print(f"  {GRAU}Frage  : {b['frage']}{AUS}")
    print(f"  {GRAU}richtig: {b['richtig']}   Ablenkung: {b['falsch']}{AUS}\n")

    print(f"  {'Modell':30}{'mit Kontext':>13}{'ohne':>8}{'Gewinn':>9}"
          f"{'erste Zahl':>13}{'gekürzt':>10}")
    print("  " + "-" * 84)
    for pfad in args.ckpt:
        netz, cfg = lade(pfad, device)
        mit, ohne, erste, quote = pruefe(netz, tok, cfg, device, aufgaben,
                                        pfad.split("/")[-1], args.form)
        farbe = GRUEN if mit - ohne > 0.15 else (ROT if mit - ohne < 0.05 else "")
        warn = ROT if quote > 0.1 else ""
        print(f"  {pfad:30}{farbe}{mit*100:12.1f}%{AUS}{ohne*100:7.1f}%"
              f"{(mit-ohne)*100:+8.1f}{erste*100:12.1f}%"
              f"{warn}{quote*100:9.1f}%{AUS}")
    print()


if __name__ == "__main__":
    main()
