"""pruefsatz_extern.py — unabhaengige deutsche Pruefsaetze aus MKQA und Mintaka.

Warum: Unsere 195 echten Fragen (chat_fragen_echt.json) und die 60 der
Gegenprobe haben wir selbst geschrieben, und die Faktenstufe wurde an ihnen
nachgebessert. Wer an seinem Pruefsatz schraubt, misst irgendwann nur noch
den Pruefsatz. Diese beiden Saetze hat niemand von uns gesehen:

  MKQA (Apple, 2020): 10.000 echte Suchanfragen aus Natural Questions, von
      Menschen in 26 Sprachen uebersetzt, Antworten an Wikidata gebunden.
      https://github.com/apple/ml-mkqa  (CC BY-SA 3.0)
  Mintaka (Amazon, 2022): 20.000 komplexe Fragen (Superlative, Zaehlen,
      Vergleiche, Mehrschritt), in 8 Sprachen inkl. Deutsch.
      https://github.com/amazon-science/mintaka  (CC BY 4.0)

Genommen werden nur Fragen mit kurzer Antwort (Entitaet, Zahl, Datum), keine
Ja/Nein- und keine unbeantwortbaren. Feste Zufallsauswahl, damit Laeufe
vergleichbar bleiben.

    python pruefsatz_extern.py            # -> chat_fragen_mkqa.json, chat_fragen_mintaka.json
                                          #    + *_dev.json zum Entwickeln

Entwicklung und Pruefung sind getrennt (Rat des Gutachters vom 25.09.): an
den *_dev-Saetzen wird gebaut und gestellt, die 500er-Pruefsaetze werden je
Schritt nur einmal gemessen. MKQA-dev kommt aus dem Rest von MKQA (ohne jede
Pruefsatzfrage), Mintaka-dev aus mintaka_dev.json.
"""
import gzip
import json
import random
import re

N = 500


def zahl_schluessel(text):
    """'11.0 Jahr' -> ['11'], '1232.7' -> ['1232,7', '1232.7', '1232']"""
    m = re.match(r"-?\d+(?:\.\d+)?", text.strip())
    if not m:
        return [text]
    z = m.group(0)
    if z.endswith(".0"):
        z = z[:-2]
    raus = [z]
    if "." in z:
        raus += [z.replace(".", ","), z.split(".")[0]]
    return raus


def mkqa(pfad="pruefsaetze/mkqa.jsonl.gz"):
    raus = []
    for zeile in gzip.open(pfad, "rt", encoding="utf-8"):
        d = json.loads(zeile)
        frage = d["queries"].get("de", "").strip()
        antworten = d["answers"].get("de", [])
        if not frage or not antworten:
            continue
        a = antworten[0]
        typ = a.get("type")
        if typ in ("unanswerable", "long_answer", "binary") or not a.get("text"):
            continue
        if typ in ("number", "number_with_unit"):
            schl = zahl_schluessel(a["text"])
        elif typ == "date":
            schl = [a["text"][:4]] if re.match(r"\d{4}", a["text"]) else [a["text"]]
        else:
            schl = [a["text"]] + list(a.get("aliases") or [])
        schl = [s for s in schl if s and len(s) >= 2]
        if schl:
            raus.append({"frage": frage, "antwort": schl, "typ": typ, "quelle": "mkqa"})
    return raus


def mintaka(pfade=("pruefsaetze/mintaka_test.json",)):
    raus = []
    for p in pfade:
        for e in json.load(open(p, encoding="utf-8")):
            frage = e.get("translations", {}).get("de", "").strip()
            a = e.get("answer", {})
            if not frage or a.get("answerType") not in ("entity", "numerical", "date"):
                continue
            if a["answerType"] == "entity":
                schl = [x.get("label", {}).get("de") or x.get("label", {}).get("en")
                        for x in (a.get("answer") or [])]
            elif a["answerType"] == "numerical":
                schl = zahl_schluessel(str((a.get("answer") or [""])[0]))
            else:
                schl = [str((a.get("answer") or [""])[0])[:4]]
            schl = [s for s in schl if s and len(s) >= 2]
            if schl:
                raus.append({"frage": frage, "antwort": schl, "typ": e.get("complexityType"),
                             "quelle": "mintaka"})
    return raus


def schreibe(name, auswahl, brauchbar):
    json.dump(auswahl, open(f"chat_fragen_{name}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    typen = {}
    for e in auswahl:
        typen[e["typ"]] = typen.get(e["typ"], 0) + 1
    print(f"  {name}: {brauchbar} brauchbar, {len(auswahl)} gezogen  {typen}")


if __name__ == "__main__":
    for name, daten in (("mkqa", mkqa()), ("mintaka", mintaka())):
        rng = random.Random(7)
        rng.shuffle(daten)
        schreibe(name, daten[:N], len(daten))
        if name == "mkqa":
            # Entwicklung: dieselbe Mischung, aber keine einzige Pruefsatzfrage
            schreibe("mkqa_dev", daten[N:2 * N], len(daten) - N)
    dev = mintaka(("pruefsaetze/mintaka_dev.json",))
    random.Random(7).shuffle(dev)
    schreibe("mintaka_dev", dev[:N], len(dev))
