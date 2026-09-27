"""knowledge/works.py — Fragen zu Werken: Lieder, Filme, Buecher, Spiele.

MKQA fragt vieles ueber englischsprachige Werke, klein geschrieben und ohne
deutschen Wikipedia-Artikel: "wer singt i can't fight this feeling",
"wann kam das erste spyro-spiel heraus". Die Tabellen in fakten/ sind nach
deutschen Artikeltiteln geordnet und finden das nicht. Diese Stufe schlaegt
ueber englische UND deutsche Wikidata-Titel nach (scripts/fetch_works.sh).

Probe am 25.09. auf MKQA-dev, 51 Fragen "Wer singt ...?":
    Teilstueck des Titels genuegt       47 Antworten, 19 richtig (40 %)
    Rest der Frage = genau ein Titel      17 Antworten, 11 richtig (65 %)
    + Coverversionen: bekanntestes Werk   16 Antworten, 13 richtig (81 %)
Genauigkeit vor Reichweite: eine falsche Antwort mit Fakten-Anschein ist
schlimmer als keine. Deshalb muss der Rest der Frage GENAU ein Titel sein.

Mehrdeutigkeit in zwei Stufen, beide ueber die Zahl der Sprachversionen:
  - welches Werk: "The House of the Rising Sun" gibt es als Original und als
    Dutzende Cover - das Original hat die meisten Sprachversionen
  - welcher Wert: dasselbe Werk nennt mehrere Interpreten (Original und
    Cover) - gefragt ist fast immer der bekannteste
"""
import csv
import os
import re
from collections import defaultdict

GATTUNG = r"(?:das lied|den song|der song|das original|den film|der film|die serie|" \
          r"das buch|den roman|der roman|das gedicht|das album|das spiel|das videospiel|" \
          r"die oper|das musical|die sinfonie|das stück)"

# (Relationen in Vorrangfolge, Muster mit dem Titel als Gruppe 1)
FRAGEN = [
    (("interpret",), [
        rf"^wer (?:hat |haben )?(?:sang|singt|singen)(?: {GATTUNG})?(?: von)? (.+?)(?: gesungen)?$",
        rf"^wer hat (?:{GATTUNG} )?(.+) gesungen$",
        r"^wer ist (?:der sänger|die sängerin|der interpret) von (.+)$",
    ]),
    (("regie",), [
        rf"^wer (?:führte|hat) regie bei (?:{GATTUNG} )?(.+?)(?: geführt)?$",
        rf"^wer (?:führte|hat) (?:bei )?(?:{GATTUNG} )?(.+?) regie(?: geführt)?$",
        rf"^wer (?:ist|war) (?:der )?regisseur (?:von|des films) (?:{GATTUNG} )?(.+)$",
        rf"^wer drehte (?:{GATTUNG} )?(.+)$",
    ]),
    (("liedtexter", "komponist", "autor"), [
        r"^wer (?:schrieb|hat) (?:das lied|den song|den text zu|den text von) (.+?)(?: geschrieben)?$",
    ]),
    (("komponist",), [
        rf"^wer (?:komponierte|hat) (?:{GATTUNG} )?(.+?)(?: komponiert)?$",
    ]),
    (("autor", "komponist", "liedtexter"), [
        rf"^wer (?:schrieb|verfasste) (?:{GATTUNG} )?(.+)$",
        rf"^wer hat (?:{GATTUNG} )?(.+) geschrieben$",
    ]),
    (("erschienen",), [
        rf"^wann (?:kam|kamen|erschien|erschienen)(?: {GATTUNG})? (.+?)(?: heraus| raus| auf den markt| in die kinos)?$",
        rf"^wann wurde (?:{GATTUNG} )?(.+?) (?:veröffentlicht|herausgebracht|uraufgeführt)$",
        rf"^wann (?:ist|sind) (?:{GATTUNG} )?(.+?) (?:erschienen|herausgekommen|rausgekommen)$",
    ]),
]
DATEN = {"erschienen"}


def norm(text):
    """Kleinschreibung, Satzzeichen weg (Apostroph bleibt: "can't")."""
    text = text.lower().replace("’", "'").replace("`", "'")
    return re.sub(r"\s+", " ", re.sub(r"[^\w' ]+", " ", text)).strip()


def varianten(titel):
    """"the house of ..." <-> "house of ...", ebenso deutsche Artikel."""
    raus = [titel]
    for art in ("the ", "der ", "die ", "das "):
        if titel.startswith(art):
            raus.append(titel[len(art):])
    raus += ["the " + titel]
    return raus


class Werke:
    def __init__(self, ordner="fakten_werk"):
        # relation -> titel -> [(wert, sprachversionen_werk, sprachversionen_wert)]
        self.tab = defaultdict(lambda: defaultdict(list))
        if not os.path.isdir(ordner):
            return
        for datei in os.listdir(ordner):
            if not datei.endswith(".csv"):
                continue
            rel = datei[:-4]
            with open(os.path.join(ordner, datei), encoding="utf-8") as f:
                for z in csv.reader(f):
                    # nur Werke mit Wikipedia-Artikel irgendwo (siehe scripts/fetch_works.sh)
                    if len(z) < 3 or z[0] == "t" or not z[2].isdigit() or z[2] == "0":
                        continue
                    t = norm(z[0])
                    if len(t) < 3:
                        continue
                    m = int(z[3]) if len(z) > 3 and z[3].isdigit() else 0
                    self.tab[rel][t].append((z[1], int(z[2]), m))

    def werk(self, rel, titel):
        """(Sprachversionen, Eintraege) des bekanntesten Werks mit diesem Titel."""
        eintraege = [e for v in varianten(titel) for e in self.tab[rel].get(v, [])]
        if not eintraege:
            return 0, []
        # Welches Werk: das mit den meisten Sprachversionen (Original statt Cover)
        n_max = max(e[1] for e in eintraege)
        return n_max, [e for e in eintraege if e[1] == n_max]

    def wert(self, rel, werk):
        if rel in DATEN:
            jahre = [e[0][:4] for e in werk if re.match(r"\d{4}", e[0])]
            return min(jahre) if jahre else None
        # Welcher Wert: der bekannteste (Original-Interpret statt Cover)
        return max(werk, key=lambda e: e[2])[0]

    def antwort(self, frage):
        if not self.tab:
            return None
        q = norm(frage)
        for relationen, muster in FRAGEN:
            for m in muster:
                treffer = re.match(m, q)
                if not treffer:
                    continue
                titel = treffer.group(1).strip()
                if len(titel) < 3:
                    continue
                # "Wer schrieb die Zauberfloete?": die Oper hat einen Komponisten,
                # keinen Autor - unter "autor" steht nur ein unbekanntes Buch
                # gleichen Namens. Das bekannteste Werk ueber alle passenden
                # Beziehungen entscheidet; bei Gleichstand die Vorrangfolge.
                # Die erste Beziehung ist die zur Frage passende und zaehlt
                # doppelt: "Wer schrieb Romeo und Julia?" meint den Autor, auch
                # wenn die Verfilmung (Komponist Nino Rota) fast so bekannt ist.
                # Die Zauberfloete hat gar keinen bekannten Autor-Eintrag, dort
                # gewinnt die Oper trotzdem.
                beste = None
                for rang, rel in enumerate(relationen):
                    n, werk = self.werk(rel, titel)
                    note = n * (2 if rang == 0 else 1)
                    if werk and (beste is None or note > beste[0]):
                        beste = (note, rel, werk)
                if beste:
                    a = self.wert(beste[1], beste[2])
                    if a:
                        return a
        return None


if __name__ == "__main__":
    import sys
    print(Werke().antwort(" ".join(sys.argv[1:])))
