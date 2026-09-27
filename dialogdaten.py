"""Deutsche Dialogdatensaetze einlesen - eine Stelle fuer alle Formate.

Wird von build_mix.py (Pretraining) und get_sft_data.py (Finetuning) benutzt,
damit beide exakt dieselben Daten und Filter sehen.

Stolperstein bei den HF-Datensaetzen: das Feld "instruction" enthaelt je nach
Datensatz zwei voellig verschiedene Dinge.

    dolly:      instruction = die Frage,           input = Zusatztext
    dolphin:    instruction = "Du bist ein KI ..." input = die Frage
    airoboros:  instruction = "Du bist ein ..."    input = die Frage

Wuerde man beides stumpf aneinanderhaengen, stuende in jeder zweiten
Nutzerfrage ein Systemprompt. Ein 5M-Modell lernt daraus, Rollenbeschreibungen
zu wiederholen statt zu antworten.
"""

import gzip
import json
import re

from chat_format import ASSISTANT, BOS, EOS, USER

# Anfaenge, die eine Rollenbeschreibung markieren statt einer Frage
SYSTEM_ANFANG = re.compile(
    r"^(du bist|deine rolle|sie sind|als (ki|assistent)|you are|dein ziel ist)", re.I)

# Inhalte, die ein kleines deutsches Konversationsmodell nur verwirren
UNERWUENSCHT = re.compile(r"```|\$\$|\\\(|\\begin\{|<html|def |SELECT |#include")

# Fremde Schriftsysteme (kyrillisch, griechisch, CJK, arabisch, hebraeisch)
FREMDSCHRIFT = re.compile(r"[Ͱ-ϿЀ-ӿ֐-ۿ"
                          r"　-鿿가-힯]")
# Buchstaben, die es im Deutschen nicht gibt (tuerkisch, polnisch, ...)
FREMDBUCHSTABEN = re.compile(r"[ıİğĞşŞąćęłńśźżŁÐþıçÇåøæ]")
# haeufige deutsche Funktionswoerter (keines davon ist ein englisches Wort)
DEUTSCH = re.compile(
    r"\b(der|die|das|des|dem|den|und|oder|aber|ist|sind|war|waren|wird|werden|"
    r"ich|du|er|sie|es|wir|ihr|nicht|kein|keine|ein|eine|einen|einem|einer|"
    r"zu|zum|zur|mit|auf|für|von|vom|sich|man|im|ins|als|auch|wenn|hat|haben|"
    r"bei|nach|aus|über|unter|zwischen|durch|gegen|ohne|um|noch|schon|nur|"
    r"sehr|mehr|dann|denn|damit|dass|weil|kann|können|muss|müssen|soll|"
    r"wie|was|wer|wo|warum|welche|diese|dieser|dieses|seine|ihre|zwei|drei)\b",
    re.I)

MAX_FRAGE = 400      # Zeichen
MAX_ANTWORT = 700


def ist_deutsch(text: str) -> bool:
    """Grobe, aber wirksame Sprachpruefung.

    Noetig, weil in dolphin_de und aehnlichen Datensaetzen Uebersetzungsaufgaben
    stecken: deutsche Frage, tuerkische oder russische Antwort. Ein Modell mit
    5-8M Parametern hat fuer so etwas keine Kapazitaet - es faengt dann an,
    mitten im Satz die Sprache zu wechseln.
    """
    if FREMDSCHRIFT.search(text) or FREMDBUCHSTABEN.search(text):
        return False
    woerter = text.split()
    if len(woerter) < 6:                  # zu kurz zum Messen, durchlassen
        return True
    # in normalem Deutsch ist rund jedes dritte Wort ein Funktionswort;
    # eins pro zehn Woerter ist eine grosszuegige Untergrenze
    return len(DEUTSCH.findall(text)) >= max(1, len(woerter) // 10)


def _norm(msgs, max_frage=MAX_FRAGE, max_antwort=MAX_ANTWORT):
    """Vereinheitlicht und filtert einen Dialog. Gibt None zurueck, wenn er raus soll."""
    if len(msgs) < 2:
        return None
    sauber = []
    for m in msgs:
        inhalt = (m.get("content") or "").strip()
        grenze = max_frage if m["role"] == "user" else max_antwort
        if not inhalt or UNERWUENSCHT.search(inhalt) or not ist_deutsch(inhalt) \
                or len(inhalt) > grenze:
            # Nicht den ganzen Dialog wegwerfen, sondern hier abschneiden.
            # Gerade mehrstufige Gespraeche haben oft eine lange Nachricht am
            # Ende - der Anfang davor ist trotzdem gutes Trainingsmaterial.
            break
        sauber.append({"role": m["role"], "content": inhalt})

    while sauber and sauber[-1]["role"] != "assistant":
        sauber.pop()                      # muss mit einer Antwort enden
    if len(sauber) < 2 or sauber[0]["role"] != "user":
        return None
    return sauber


def _aus_alpaca(r):
    frage = (r.get("instruction") or r.get("prompt") or r.get("question") or "").strip()
    extra = (r.get("input") or "").strip()
    antwort = (r.get("output") or r.get("response") or r.get("completion")
               or r.get("answer") or "").strip()
    if not antwort:
        return None

    if not frage and extra:          # Datensaetze mit nur input/output
        frage, extra = extra, ""
    if extra and SYSTEM_ANFANG.match(frage):
        frage = extra            # instruction war nur die Rollenbeschreibung
    elif extra:
        frage = f"{frage}\n{extra}"
    if not frage:
        return None

    msgs = []
    for paar in (r.get("history") or []):   # evol-instruct: frueherer Gespraechsverlauf
        if isinstance(paar, (list, tuple)) and len(paar) == 2:
            msgs.append({"role": "user", "content": str(paar[0])})
            msgs.append({"role": "assistant", "content": str(paar[1])})
    msgs.append({"role": "user", "content": frage})
    msgs.append({"role": "assistant", "content": antwort})
    return msgs


def _rolle(wert):
    return "user" if str(wert).lower() in ("human", "user", "prompter", "mensch") \
        else "assistant"


def _aus_zeile(r):
    """Eine Datensatzzeile -> Nachrichtenliste, egal in welchem Format sie kommt."""
    if "conversations" in r and r["conversations"] is not None:      # ShareGPT
        return [{"role": _rolle(c.get("from")), "content": (c.get("value") or "").strip()}
                for c in r["conversations"]]
    if "messages" in r and r["messages"] is not None:                 # OASST-Ketten
        return [{"role": _rolle(m.get("role")), "content": (m.get("content") or "").strip()}
                for m in r["messages"]]
    if "speaker_a" in r and "speaker_b" in r:                         # Alltagsgespraeche
        a, b = r.get("speaker_a") or {}, r.get("speaker_b") or {}
        # die Felder emotion/intent/body_language sind Regieanweisungen,
        # nur der gesprochene Text gehoert ins Training
        return [{"role": "user", "content": (a.get("text") or "").strip()},
                {"role": "assistant", "content": (b.get("text") or "").strip()}]
    return _aus_alpaca(r)


def _oasst_baum(knoten, sprache="de"):
    """Bester Pfad durch einen OpenAssistant-Gespraechsbaum.

    Die Baeume enthalten mehrere Antwortvarianten pro Frage, von Menschen
    bewertet (rank 0 = beste). Wir folgen immer der bestbewerteten Antwort,
    damit kein Muell ins Training kommt.
    """
    msgs = []
    aktuell = knoten
    while aktuell is not None:
        if aktuell.get("lang") != sprache:
            return None
        rolle = "user" if aktuell.get("role") == "prompter" else "assistant"
        msgs.append({"role": rolle, "content": (aktuell.get("text") or "").strip()})
        antworten = [a for a in (aktuell.get("replies") or [])
                     if a.get("lang") == sprache and not a.get("deleted")]
        if not antworten:
            break
        antworten.sort(key=lambda a: (a.get("rank") if a.get("rank") is not None else 99))
        aktuell = antworten[0]
    return msgs if len(msgs) >= 2 else None


def lade(path, max_frage=MAX_FRAGE, max_antwort=MAX_ANTWORT):
    """Liefert Dialoge als [{'role':..., 'content':...}, ...] - schon gefiltert.

    Fuer das Pretraining darf man grosszuegiger sein (laengere Antworten), fuer
    das Finetuning lieber streng - deshalb sind die Grenzen einstellbar.
    """
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2000):
            for r in batch.to_pylist():
                msgs = _aus_zeile(r)
                if msgs:
                    d = _norm(msgs, max_frage, max_antwort)
                    if d:
                        yield d
        return

    if path.endswith((".jsonl", ".jsonl.gz")):
        oeffnen = gzip.open if path.endswith(".gz") else open
        with oeffnen(path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                msgs = _oasst_baum(r["prompt"]) if "prompt" in r and \
                    isinstance(r.get("prompt"), dict) else _aus_zeile(r)
                if msgs:
                    d = _norm(msgs, max_frage, max_antwort)
                    if d:
                        yield d
        return

    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return
    if isinstance(data, dict):
        data = data.get("data", [])

    for r in data:
        if not isinstance(r, dict):
            continue
        msgs = _aus_zeile(r)
        if msgs:
            d = _norm(msgs, max_frage, max_antwort)
            if d:
                yield d


def als_text(msgs):
    """Dialog -> Trainingstext im Chat-Format (fuer das Pretraining)."""
    teile = [BOS]
    for m in msgs:
        teile.append((USER if m["role"] == "user" else ASSISTANT) + m["content"])
    return "".join(teile) + EOS
