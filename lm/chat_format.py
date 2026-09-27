"""Gemeinsames Chat-Format fuer Finetuning und Inferenz.

Ein Dialog sieht als Token-Strom so aus:

    <s> \n<user>\n Wie geht es dir? \n<assistant>\n Gut, danke! </s>

Wichtig: Finetuning und Chat muessen exakt dasselbe Format benutzen,
sonst antwortet das Modell Unsinn.
"""

USER = "\n<user>\n"
ASSISTANT = "\n<assistant>\n"
BOS = "<s>"
EOS = "</s>"


def mit_wissen(frage: str, wissen: str) -> str:
    """Frage mit nachgeschlagenem Wissen kombinieren.

    Exakt das Format aus wissen_daten.py - Training und Inferenz muessen
    zeichengenau uebereinstimmen, sonst nutzt das Modell den Kontext nicht.
    Genau hier haengt spaeter die Wissens-Engine von der SD-Karte ein.
    """
    return f"Wissen: {wissen.strip()}\nFrage: {frage.strip()}"


def frage_zuerst(frage: str, wissen: str) -> str:
    """Dieselben Bestandteile, umgekehrte Reihenfolge.

    Der Grund ist die kausale Aufmerksamkeit: Steht die Frage hinten, hat
    das Modell den ganzen Absatz gelesen, OHNE zu wissen, wonach es sucht.
    Jedes Absatz-Token wurde verarbeitet, bevor die Frage bekannt war - die
    Suche kann erst im allerletzten Moment beginnen.

    Steht die Frage vorn, sieht jedes Absatz-Token sie bereits. Bei einem
    Fenster von 384 Token und Absaetzen um 250 Token ist das kein kleiner
    Unterschied. Dazu kommt der Abfall der Rotationskodierung ueber die
    Entfernung, der Anfang und Ende bevorzugt.

    Getestet werden kann das nur mit passend trainierten Daten - ein
    Formatwechsel allein waere ausserhalb dessen, was das Modell kennt.
    """
    return f"Frage: {frage.strip()}\nWissen: {wissen.strip()}"


def build_prompt(messages, add_generation_prompt=True):
    """messages: [{'role': 'user'|'assistant', 'content': str}, ...]"""
    parts = [BOS]
    for m in messages:
        tag = USER if m["role"] == "user" else ASSISTANT
        parts.append(tag + m["content"].strip())
    if add_generation_prompt:
        parts.append(ASSISTANT)
    return "".join(parts)


def encode_example(tok, messages, block_size):
    """Erzeugt (input_ids, labels). Nur die Assistenz-Antworten zaehlen fuer den
    Loss - der Rest bekommt -100, damit das Modell nicht lernt, die Frage
    des Nutzers zu erfinden."""
    ids, labels = [], []

    def add(text, learn):
        e = tok.encode(text, add_special_tokens=False).ids
        ids.extend(e)
        labels.extend(e if learn else [-100] * len(e))

    add(BOS, False)
    for m in messages:
        if m["role"] == "user":
            add(USER + m["content"].strip(), False)
        else:
            add(ASSISTANT, False)
            add(m["content"].strip() + EOS, True)

    ids, labels = ids[:block_size], labels[:block_size]
    return ids, labels
