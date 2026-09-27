"""lesen_daten.py — Lesetraining mit ECHTEN Fragen in der Form der Kette.

Gemessen am 25.09.2026 (strenger Bewerter, gestufter Index): auf MKQA-dev
steht die Antwort in 21 % der Faelle in den drei geholten Absaetzen, die
Kette beantwortet aber nur 10,8 %. Mit dem jetzigen Modell als Leser sinkt
die Quote sogar auf 8,2 %, bei 11 % erfundenen Antworten. Das Modell wurde
auf Schablonenfragen trainiert ("Wie hoch ist X?"), nicht auf echten.

Dieses Skript baut Lesebeispiele genau in der Form, die das Modell in der
Kette sieht - "Wissen: <drei Absaetze aus der Suche>\\nFrage: ..." -> kurze
Antwort oder "Das steht nicht im Text.":

  GermanQuAD      11.500 deutsche Fragen von Menschen zu Wikipedia-Absaetzen
                  (CC BY 4.0). Der richtige Absatz + 2 Ablenker aus unserer
                  Suche, gemischt.
  Mintaka-train   14.000 Fragen (getrennt von dev/test)       } Absaetze genau
  MKQA-Rest       ohne jede Dev-/Pruefsatzfrage               } wie in der Kette

Pruefsatz-Sperre: keine Frage aus irgendeinem Mess- oder Entwicklungssatz
(chat_fragen_*.json) - siehe pruefsatz-nicht-mittrainieren (82 % Leck).

    python lesen_daten.py            # -> data/sft/lesen_echt.jsonl
"""
import glob
import json
import random
import re
import sys
import time

from chat_format import mit_wissen
from chat_test import enthaelt, norm
from pruefsatz_extern import mkqa, mintaka
from wissen_index import Wissensbasis

JE_ABSATZ = 500               # wie bester_text(max_zeichen=1500, absaetze=3)
ABLEHNUNG = "Das steht nicht im Text."
ANTEIL_ABLEHNUNG = 0.35       # von den Fragen ohne Antwort im Text so viele behalten


def kuerze(text, je=JE_ABSATZ):
    """Genau wie wissen_index.bester_text: an einer Satzgrenze kuerzen."""
    if len(text) > je:
        schnitt = text.rfind(". ", 0, je)
        text = text[:schnitt + 1] if schnitt > 120 else text[:je]
    return text


def fenster(titel, absatz, antwort, je=JE_ABSATZ):
    """Ausschnitt des richtigen Absatzes, der die Antwort sicher enthaelt."""
    i = absatz.find(antwort)
    kopf = f"{titel}: "
    platz = je - len(kopf)
    if i + len(antwort) <= platz - 20:
        return kopf + kuerze(absatz, platz)
    start = max(0, i - platz // 2)
    satz = absatz.rfind(". ", 0, start)        # an einem Satzanfang beginnen
    if satz > 0 and i - satz < platz - len(antwort):
        start = satz + 2
    return kopf + absatz[start:start + platz]


def gesperrte_fragen():
    raus = set()
    for p in glob.glob("chat_fragen_*.json"):
        for e in json.load(open(p, encoding="utf-8")):
            raus.add(norm(e["frage"]))
    return raus


def germanquad(pfad="pruefsaetze/germanquad_prompted.parquet"):
    """Frage, Antwort, Titel und Absatz aus der Schablonen-Fassung zurueckgewinnen.

    Die Kopie auf Hugging Face enthaelt jedes Beispiel in ~14 Schablonen. Die
    Schablone "Question: X\\n\\nAnswer:" liefert die reine Frage; eine andere
    Zeile derselben Frage enthaelt Titel ("Recht_der_Vereinigten_Staaten")
    und Absatz mit der Antwort.
    """
    import pyarrow.parquet as pq
    zeilen = pq.read_table(pfad).to_pylist()
    fragen = {}
    for z in zeilen:
        m = re.fullmatch(r"Question: (.+?)\s*\n\nAnswer:", z["inputs"].strip(), re.S)
        if m:
            fragen[m.group(1).strip()] = z["targets"].strip()
    beste = {}
    for z in zeilen:
        text = z["inputs"]
        # Jede Frage gegen jede Zeile waere teuer - stattdessen die Frage aus
        # der Zeile ziehen: sie steht woertlich darin
        m = re.search(r"\n\n([^\n]+)\n\n=+ .+? =+\n(.+)", text, re.S)
        if not m:
            continue
        titel = m.group(1).strip().replace("_", " ")
        absatz = max(m.group(2).split("\n"), key=len).strip()
        antwort = z["targets"].strip()
        if antwort not in absatz:
            continue
        for frage in re.findall(r"[^\n\"„“:]{8,300}\?", text):
            f = frage.strip()
            if f in fragen and fragen[f] == antwort:
                beste[f] = (antwort, titel, absatz)
    return [(f, a, t, p) for f, (a, t, p) in beste.items()]


def main():
    rng = random.Random(25)
    gesperrt = gesperrte_fragen()
    print(f"  {len(gesperrt):,} Mess-/Entwicklungsfragen gesperrt", flush=True)
    wb = Wissensbasis("wiki_gestuft")
    beispiele, zaehler, t0 = [], {}, time.time()

    def zaehle(k):
        zaehler[k] = zaehler.get(k, 0) + 1

    # ---- GermanQuAD: richtiger Absatz + 2 Ablenker
    gq = germanquad()
    print(f"  GermanQuAD: {len(gq):,} Fragen zurueckgewonnen  {time.time()-t0:.0f}s", flush=True)
    for n, (frage, antwort, titel, absatz) in enumerate(gq):
        if norm(frage) in gesperrt or len(antwort) > 60 or len(antwort.split()) > 6:
            zaehle("gq_zu_lang_oder_gesperrt")
            continue
        ablenker = [kuerze(t) for _, t in wb.suche(frage, 6)
                    if antwort.lower() not in t.lower()][:2]
        if len(ablenker) < 2:
            continue
        if rng.random() < 0.12:            # ohne den richtigen Absatz: ablehnen
            teile, ziel = ablenker + [kuerze(t) for _, t in wb.suche(frage, 8)
                                      if antwort.lower() not in t.lower()][2:3], ABLEHNUNG
            zaehle("gq_ablehnung")
        else:
            teile = ablenker + [fenster(titel, absatz, antwort)]
            rng.shuffle(teile)
            ziel = antwort
            zaehle("gq_antwort")
        beispiele.append({"messages": [{"role": "user", "content": mit_wissen(frage, " ".join(teile))},
                                       {"role": "assistant", "content": ziel}]})
        if n % 2000 == 0:
            print(f"    {n:,}  {time.time()-t0:.0f}s", flush=True)

    # ---- Mintaka-train und MKQA-Rest: Absaetze genau wie in der Kette
    mk = mkqa()
    rng_mk = random.Random(7)            # dieselbe Mischung wie pruefsatz_extern
    rng_mk.shuffle(mk)
    quellen = [("mintaka", mintaka(("pruefsaetze/mintaka_train.json",))),
               ("mkqa", mk[1000:])]      # die ersten 1000 sind Pruefung und Entwicklung
    for name, daten in quellen:
        for n, e in enumerate(daten):
            frage = e["frage"]
            if norm(frage) in gesperrt:
                zaehle(f"{name}_gesperrt")
                continue
            absatz = wb.bester_text(frage, max_zeichen=1500, absaetze=3) or ""
            if not absatz:
                continue
            # Antwort in der Schreibung des Textes (ganzes Wort)
            ziel = None
            for k in e["antwort"]:
                m = re.search(r"(?<![\wäöüß])" + re.escape(str(k)) + r"(?![\wäöüß])",
                              absatz, re.IGNORECASE)
                if m:
                    ziel = m.group(0)
                    break
            if ziel is None:
                if rng.random() > ANTEIL_ABLEHNUNG:
                    continue
                ziel = ABLEHNUNG
                zaehle(f"{name}_ablehnung")
            else:
                zaehle(f"{name}_antwort")
            beispiele.append({"messages": [{"role": "user", "content": mit_wissen(frage, absatz)},
                                           {"role": "assistant", "content": ziel}]})
            if n % 2000 == 0:
                print(f"    {name} {n:,}  {time.time()-t0:.0f}s", flush=True)

    rng.shuffle(beispiele)
    with open("data/sft/lesen_echt.jsonl", "w", encoding="utf-8") as f:
        for b in beispiele:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"\n  {len(beispiele):,} Beispiele -> data/sft/lesen_echt.jsonl  {time.time()-t0:.0f}s")
    for k in sorted(zaehler):
        print(f"    {k:<28}{zaehler[k]:>7,}")


if __name__ == "__main__":
    sys.exit(main())
