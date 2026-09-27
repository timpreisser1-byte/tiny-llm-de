"""Kurze Kontexte zum Nachschlagen - das Regime, das bisher fehlt.

Die Messung, aus der dieses Skript entstand: Das 64M-Modell lehnt bei kurzen
vorgelegten Texten in zwölf von fuenfzehn Faellen ab, obwohl die Antwort
woertlich dasteht. Zwei Ursachen, beide in den Daten:

  1. In `hochwertig.jsonl` steht 4.144-mal derselbe Ablehnungssatz. Ueber zwei
     Epochen ist das die haeufigste Zeichenkette im ganzen Feintuning - viermal
     haeufiger als der naechste Satz. Wer unsicher ist, sagt ihn.

  2. `kontext_qa.jsonl` hat im Mittel 505 Zeichen Kontext, die Pruefungen
     benutzen 65. Das kurze Regime wird praktisch nicht geuebt.

Dieses Skript erzeugt den fehlenden Datensatz aus der Wikipedia-Sammlung:
ein bis zwei Saetze Kontext, eine Frage, eine kurze Antwort. Die Fragen
entstehen durch Umkehrung typischer deutscher Satzmuster - kein fremdes
Modell noetig, und dadurch nachvollziehbar, was drinsteht.

Ein kleiner Teil ist bewusst nicht beantwortbar, damit die Faehigkeit zur
Ablehnung nicht verloren geht - aber mit wechselnden Formulierungen, damit
kein einzelner Satz wieder zum Reflex wird.

    python kurzkontext.py --anzahl 15000
"""

import argparse
import json
import os
import random
import re
import sys
import time

# (Muster, Fragevorlage, welche Gruppe die Antwort ist)
# (Muster, Fragevorlage, Antwortvorlage)
#
# Nur Zahlmuster. Die ersten Versuche mit "besteht aus X" und "liegt in X"
# lieferten Bruchstuecke wie "Aus negativ." - eine Nominalphrase sauber aus
# einem Satz zu schneiden braucht mehr als einen regulaeren Ausdruck. Zahlen
# und Jahre dagegen sind eindeutig, und sie sind genau das, was in den
# Pruefungen gefragt wird.
MUSTER = [
    (r"(?P<x>[A-ZÄÖÜ][\wäöüß\- ]{2,38}?) (?:ist|misst) (?P<a>[\d.,]+) ?(?:m|Meter) hoch",
     "Wie hoch ist {x}?", "{a} Meter"),
    (r"(?P<x>[A-ZÄÖÜ][\wäöüß\- ]{2,38}?) (?:ist|misst) (?P<a>[\d.,]+) ?(?:m|Meter) (?:tief|lang|breit)",
     "Wie tief ist {x}?", "{a} Meter"),
    (r"(?P<x>[A-ZÄÖÜ][\wäöüß\- ]{2,38}?) (?:ist|misst) (?P<a>[\d.,]+) ?(?:km|Kilometer) lang",
     "Wie lang ist {x}?", "{a} Kilometer"),
    (r"(?P<x>[A-ZÄÖÜ][\wäöüß\- ]{2,38}?) (?:hat|zählt|besitzt) (?P<a>[\d.]+) Einwohner",
     "Wie viele Einwohner hat {x}?", "{a} Einwohner"),
    (r"(?P<x>[A-ZÄÖÜ][\wäöüß\- ]{2,38}?) wurde (?:im Jahr(?:e)? )?(?P<a>1[0-9]{3}|20[0-2][0-9]) (?:gegründet|erbaut|errichtet|eröffnet|fertiggestellt|geweiht)",
     "In welchem Jahr wurde {x} gebaut?", "Im Jahr {a}"),
    (r"(?P<x>[A-ZÄÖÜ][\wäöüß\- ]{2,38}?) umfasst (?:eine Fläche von )?(?P<a>[\d.,]+) ?(?:km²|Quadratkilometer)",
     "Wie groß ist {x}?", "{a} Quadratkilometer"),
    (r"(?P<x>[A-ZÄÖÜ][\wäöüß\- ]{2,38}?) (?:hat|zählt) (?P<a>[\d.]+) (?:Mitglieder|Beschäftigte|Schüler|Studenten|Sitzplätze)",
     "Wie viele Mitglieder hat {x}?", "{a}"),
    (r"(?P<x>[A-ZÄÖÜ][\wäöüß\- ]{2,38}?) wiegt (?P<a>[\d.,]+) ?(?:kg|Kilogramm|Tonnen)",
     "Wie schwer ist {x}?", "{a} Kilogramm"),
    # Eigennamen lassen sich ebenso sauber schneiden wie Zahlen, solange man
    # eine GROSSGESCHRIEBENE Wortfolge nimmt statt eines einzelnen Wortes.
    # Der erste Versuch scheiterte an "Aus negativ." - ein Wort reicht nicht.
    (r"(?P<x>[A-ZÄÖÜ][\wäöüß\- ]{2,38}?) liegt (?:im|in) (?:der |dem )?"
     r"(?P<a>[A-ZÄÖÜ][\wäöüß]+(?:[ -][A-ZÄÖÜ][\wäöüß]+){0,2})",
     "Wo liegt {x}?", "In {a}"),
    (r"(?P<x>[A-ZÄÖÜ][\wäöüß\- ]{2,38}?) ist (?:die|der|das) Hauptstadt "
     r"(?:von|des|der) (?P<a>[A-ZÄÖÜ][\wäöüß]+(?:[ -][A-ZÄÖÜ][\wäöüß]+){0,2})",
     "Wovon ist {x} die Hauptstadt?", "Von {a}"),
    (r"(?P<x>[A-ZÄÖÜ][\wäöüß\- ]{2,38}?) (?:wurde|wird) von "
     r"(?P<a>[A-ZÄÖÜ][\wäöüß]+(?:[ -][A-ZÄÖÜ][\wäöüß]+){0,2}) "
     r"(?:gegründet|entwickelt|erfunden|entworfen|gebaut)",
     "Von wem wurde {x} gebaut?", "Von {a}"),
    (r"(?P<x>[A-ZÄÖÜ][\wäöüß\- ]{2,38}?) (?:mündet|fließt) (?:in|durch) "
     r"(?:die |den |das )?(?P<a>[A-ZÄÖÜ][\wäöüß]+)",
     "Wo mündet {x}?", "In {a}"),
    (r"(?P<x>[A-ZÄÖÜ][\wäöüß\- ]{2,38}?) gehört zu(?:m|r)? "
     r"(?P<a>[A-ZÄÖÜ][\wäöüß]+(?:[ -][A-ZÄÖÜ][\wäöüß]+){0,2})",
     "Wozu gehört {x}?", "Zu {a}"),
    (r"(?P<x>[A-ZÄÖÜ][\wäöüß\- ]{2,38}?) besteht aus "
     r"(?P<a>[A-ZÄÖÜ][\wäöüß]+(?:[ -]und[ -][A-ZÄÖÜ][\wäöüß]+)?)",
     "Woraus besteht {x}?", "Aus {a}"),
]


def saeubere(x):
    """Betreff aus dem Satz zurechtstutzen, sonst steht "Woraus besteht Die
    Huelle?" in der Frage. Artikel klein, Bindewoerter vorne weg."""
    x = x.strip(" ,;")
    for weg in ("und ", "oder ", "aber ", "denn ", "sowie ", "auch "):
        if x.lower().startswith(weg):
            x = x[len(weg):]
    teile = x.split()
    if teile and teile[0] in ("Der", "Die", "Das", "Ein", "Eine", "Den", "Dem"):
        teile[0] = teile[0].lower()
    x = " ".join(teile)
    # endet der Betreff auf einem Fuellwort, taugt er nicht
    if not x or x.split()[-1].lower() in ("in", "an", "auf", "von", "mit",
                                          "bei", "für", "im", "am", "zum"):
        return None
    return x if 3 <= len(x) <= 40 else None


# Wechselnde Formulierungen fuer den Fall, dass es wirklich nicht dasteht.
# Genau das war der Fehler im alten Gemisch: EIN Satz, viertausendmal.
ABLEHNUNGEN = [
    "Das steht nicht in dem Text.",
    "Dazu steht dort nichts.",
    "Der Text sagt dazu nichts.",
    "Im Text finde ich dazu keine Angabe.",
    "Das geht aus dem Text nicht hervor.",
    "Dazu gibt der Text nichts her.",
    "Steht da nicht drin.",
    "Der Abschnitt erwähnt das nicht.",
    "Dazu schweigt der Text.",
    "Das lässt sich hier nicht ablesen.",
    "Diese Angabe fehlt im Text.",
    "Zu dieser Frage steht dort nichts.",
]

# Fragen, deren Antwort im Text sicher NICHT steht
NICHT_DA = [
    "Wie viele Fenster hat es?", "Wer ist der heutige Besitzer?",
    "Wie hoch waren die Kosten?", "Wie viele Menschen arbeiten dort?",
    "Welche Farbe hat es?", "Wie lautet die Telefonnummer?",
    "Wann sind die Öffnungszeiten?", "Wie schwer ist es?",
    "Wie viele Stockwerke sind es?", "Welches Material wurde verwendet?",
]



def aus_kontext_qa(pfad, rng, nachbarn=1):
    """Vorhandene Nachschlage-Dialoge auf das kurze Regime eindampfen.

    `kontext_qa.jsonl` hat im Mittel 505 Zeichen Kontext, die Pruefungen
    benutzen 65. Die Fragen darin sind aber gut und natuerlich formuliert -
    besser, als ein regulaerer Ausdruck sie je erzeugt. Statt neue zu bauen
    wird hier der Kontext gekuerzt: der Satz, der die Antwort traegt, plus
    hoechstens ein Nachbarsatz als Ablenkung.

    Welcher Satz die Antwort traegt, entscheidet die Ueberschneidung mit der
    Antwort - Zahlen zaehlen dabei doppelt, weil sie eindeutig sind.
    """
    raus = []
    for zeile in open(pfad, encoding="utf-8"):
        try:
            d = json.loads(zeile)
        except Exception:
            continue
        u = [m["content"] for m in d["messages"] if m["role"] == "user"]
        a = [m["content"] for m in d["messages"] if m["role"] == "assistant"]
        if not u or not a or "Frage:" not in u[0]:
            continue
        wissen, _, frage = u[0].partition("\nFrage:")
        wissen = wissen.replace("Wissen:", "", 1).strip()
        frage, antwort = frage.strip(), a[-1].strip()
        teile = saetze(wissen)
        if len(teile) < 2:
            continue
        # Zahlen aus der Antwort sind der verlaesslichste Anker
        zahlen = set(re.findall(r"[\d.,]{2,}", antwort))
        woerter = {w.lower() for w in re.findall(r"[\wäöüß]{5,}", antwort)}
        beste, punkte = None, -1
        for satz in teile:
            p = 3 * len(zahlen & set(re.findall(r"[\d.,]{2,}", satz)))
            p += len(woerter & {w.lower() for w in re.findall(r"[\wäöüß]{5,}", satz)})
            if p > punkte:
                beste, punkte = satz, p
        if beste is None or punkte <= 0:
            continue
        i = teile.index(beste)
        kurz = [beste]
        if nachbarn and len(teile) > 1:
            nachbar = teile[i + 1] if i + 1 < len(teile) else teile[i - 1]
            if len(nachbar) < 160:
                kurz = [beste, nachbar] if i + 1 < len(teile) else [nachbar, beste]
        rng.shuffle(kurz) if False else None
        raus.append({"messages": [
            {"role": "user",
             "content": f"Wissen: {' '.join(kurz)}\nFrage: {frage}"},
            {"role": "assistant", "content": antwort}]})
    return raus


def balken(i, n, t0):
    breite = 28
    voll = int(breite * i / max(n, 1))
    rest = (time.time() - t0) / max(i, 1) * (n - i)
    sys.stderr.write(f"\r  [{'█'*voll}{'░'*(breite-voll)}] {i:>6,}/{n:,}"
                     f"  noch {rest/60:4.1f} min")
    sys.stderr.flush()


def saetze(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="wiki/chunks.txt")
    ap.add_argument("--out", default="data/sft/kurzkontext.jsonl")
    ap.add_argument("--anzahl", type=int, default=15000)
    ap.add_argument("--ablehnung", type=float, default=0.04,
                    help="Anteil nicht beantwortbarer Fragen")
    ap.add_argument("--kontext-qa", default="data/sft/kontext_qa.jsonl",
                    help="vorhandene Nachschlage-Dialoge kuerzen; '' schaltet aus")
    ap.add_argument("--je-frage", type=int, default=600,
                    help="wie oft dieselbe Fragevorlage vorkommen darf - "
                         "verschiedene Kontexte, verschiedene Antworten")
    ap.add_argument("--lang", action="store_true",
                    help="die langen, fast woertlichen Antworten aus "
                         "kontext_qa mitnehmen (lehrt Abschreiben)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gebaut = [re.compile(m) for m, _, _ in MUSTER]
    # Entdoppelt wird ueber (Frage, Antwort), nicht ueber die Frage allein.
    # "Wie lang ist der Bach?" mit fuenf verschiedenen Baechen und fuenf
    # verschiedenen Antworten ist kein Duplikat, sondern der Kern der Uebung:
    # das Modell kann die Antwort nicht auswendig kennen, es muss nachsehen.
    from collections import Counter
    raus, gesehen, je_frage = [], set(), Counter()
    t0 = time.time()

    with open(args.chunks, encoding="utf-8") as f:
        for zeile in f:
            if len(raus) >= args.anzahl:
                break
            absatz = zeile.split(": ", 1)[-1].strip()
            teile = saetze(absatz)
            for i, satz in enumerate(teile):
                if len(raus) >= args.anzahl or len(satz) > 200:
                    continue
                for r, (_, vorlage, antwort) in zip(gebaut, MUSTER):
                    t = r.search(satz)
                    if not t:
                        continue
                    x = saeubere(t.group("x"))
                    if not x:
                        continue
                    frage = vorlage.format(x=x)
                    if je_frage[frage] >= args.je_frage:
                        continue
                    # Kontext: der Zielsatz plus hoechstens ein Nachbarsatz als
                    # Ablenkung - so bleibt es kurz wie in den Pruefungen
                    kontext = satz
                    if i + 1 < len(teile) and len(teile[i + 1]) < 130 and rng.random() < 0.6:
                        kontext = satz + " " + teile[i + 1]
                    elif i > 0 and len(teile[i - 1]) < 130 and rng.random() < 0.4:
                        kontext = teile[i - 1] + " " + satz
                    if rng.random() < args.ablehnung:
                        frage = rng.choice(NICHT_DA)
                        ant = rng.choice(ABLEHNUNGEN)
                    else:
                        ant = antwort.format(**t.groupdict()).strip()
                        if len(ant) > 40 or len(ant) < 4:
                            continue
                        # Deutsche Substantive sind grossgeschrieben, deshalb
                        # faengt das Eigennamen-Muster auch Allerweltswoerter
                        # ein - "In Gebiet." ist keine Antwort.
                        if any(w in ant for w in GENERISCH):
                            continue
                        if not ant.endswith("."):
                            ant += "."
                        if (frage, ant) in gesehen:
                            continue
                        gesehen.add((frage, ant))
                        je_frage[frage] += 1
                    from leseform import VORLAGEN
                    v = rng.choice(VORLAGEN)
                    kopf, _, rest = v.partition("\n")
                    mitte = rest.rpartition("\n")[0]
                    raus.append({"messages": [
                        {"role": "user", "content":
                            (kopf.format(a=kontext, f="", z="").rstrip() + "\n"
                             + mitte.format(a="", f=frage, z="").strip())},
                        {"role": "assistant", "content": ant}]})
                    break
            if len(raus) % 500 < 2:
                balken(len(raus), args.anzahl, t0)
    sys.stderr.write("\r" + " " * 70 + "\r")

    if args.lang and args.kontext_qa and os.path.exists(args.kontext_qa):
        gekuerzt = aus_kontext_qa(args.kontext_qa, rng)
        print(f"  {len(gekuerzt):,} vorhandene Dialoge gekürzt")
        raus.extend(gekuerzt)
        rng.shuffle(raus)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for e in raus:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    laengen = [len(e["messages"][0]["content"]) for e in raus]
    abl = sum(1 for e in raus
              if e["messages"][1]["content"] in ABLEHNUNGEN)
    print(f"  {len(raus):,} Beispiele -> {args.out}")
    print(f"  Prompt-Länge im Mittel {sum(laengen)/len(laengen):.0f} Zeichen")
    print(f"  davon nicht beantwortbar: {abl:,} ({abl/len(raus)*100:.1f} %), "
          f"in {len(ABLEHNUNGEN)} Formulierungen")
    print("\n  Beispiele:")
    for e in raus[:3]:
        print(f"    {e['messages'][0]['content'][:95]}")
        print(f"    -> {e['messages'][1]['content']}\n")


if __name__ == "__main__":
    main()
