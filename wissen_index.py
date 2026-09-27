"""Wissens-Engine: Wikipedia durchsuchbar machen.

Punkt 9 aus dem Projektplan. Aus dem Rohtext wird ein Suchindex, mit dem sich
zu jeder Frage der passende Absatz finden laesst. Der wandert dann als
"Wissen: ..." in den Prompt - und das Modell formuliert die Antwort daraus.

Aufbau des Index (bewusst einfach, damit die C-Engine ihn spaeter 1:1 lesen kann):

    wiki/chunks.txt     alle Absaetze hintereinander, UTF-8
    wiki/offsets.npy    Startposition jedes Absatzes (uint64)
    wiki/vokabular.json Wort -> (Startzeile in postings, Anzahl)
    wiki/postings.npy   fuer jedes Wort die Absatznummern (uint32)

Das ist eine klassische invertierte Liste. Suchen heisst: Frage in Woerter
zerlegen, fuer jedes Wort die Absatznummern holen, zaehlen welcher Absatz am
haeufigsten vorkommt - gewichtet danach, wie selten das Wort ist.

    python wissen_index.py bauen --max-artikel 500000
    python wissen_index.py suchen "Wie hoch ist der Eiffelturm?"
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time
import itertools
from collections import defaultdict

import numpy as np

WORT = re.compile(r"[a-zäöüß0-9]{3,20}")
# Woerter, die in fast jedem Absatz stehen und beim Suchen nur stoeren
STOPP = set("""der die das den dem des ein eine einen einem einer eines und oder
aber ist sind war waren wird werden wurde wurden hat habe haben hatte hatten
sich nicht auch noch schon nur sehr mehr als wie was wer wie wo bei mit von vom
für fuer auf aus dass weil wenn dann durch über ueber unter zwischen nach vor
seit bis zum zur ins die man ihm ihn ihr ihre sein seine sowie etwa rund
jahr jahre jahren teil zwei drei erste ersten heute
wann warum wieso weshalb welche welcher welches welchem welchen
viel viele vieles lang lange laenger länger hoch hohe hohen tief tiefe
gross groß grosse große alt alte weit weite schwer schwere breit breite
nenne sage sag gib zeige heisst heißt bedeutet""".split())

STANDARD = "wiki"

# Wendungen, bei denen Nachschlagen nur schadet. Wikipedia hat zu fast jedem
# Alltagswort einen Artikel - "hallo" fuehrt zum Hallo-Welt-Programm, "danke"
# zu "Macht's gut, und danke fuer den Fisch". Solche Treffer machen aus einer
# Begruessung einen Vortrag ueber Programmiersprachen.
SMALLTALK = set("""hi hallo hey moin servus na ciao tschau tschüss tschüs
danke dankeschön bitte ok okay klar cool super toll interessant verstanden
ja nein vielleicht hm hallöchen grüße grüß guten tag morgen abend nacht
wer bist du was kannst du wie heißt du bis später bis bald mach's gut""".split())

# Persoenliche Aussagen und Aufforderungen wollen Rat, kein Lexikon
KEIN_NACHSCHLAGEN = re.compile(
    r"^(ich |mir |mich |wir |kannst du |kann ich |erzähl|erzaehl|sag mir |"
    r"schreib|hilf|danke|bitte|hallo|hi|hey|moin|guten |gute |wie geht|"
    r"was machst|wer bist|was kannst|wie heißt)", re.I)


def braucht_wissen(frage: str) -> bool:
    """Soll fuer diese Eingabe ueberhaupt nachgeschlagen werden?

    Nur bei echten Wissensfragen. Begruessungen, Dank und persoenliche
    Aussagen beantwortet das Modell aus sich heraus - dort richtet ein
    Wikipedia-Absatz mehr Schaden an als Nutzen.
    """
    text = frage.strip().lower().rstrip("?!.")
    if not text or text in SMALLTALK:
        return False
    if KEIN_NACHSCHLAGEN.match(text):
        return False
    woerter = [w for w in text.split() if w not in SMALLTALK]
    if len(woerter) < 2:                      # ein einzelnes Wort sagt zu wenig
        return False
    return bool(zerlege(text))                # mindestens ein Suchwort uebrig


def zerlege(text: str) -> list:
    """Text -> Suchwoerter. Muss beim Bauen und beim Suchen identisch sein."""
    return [w for w in WORT.findall(text.lower()) if w not in STOPP]


# ---------------------------------------------------------------- bauen

def stuecke(titel, text, chunk_zeichen, max_pro_artikel):
    """Artikel in suchbare Haeppchen schneiden.

    Jedes Stueck beginnt mit dem Artikeltitel. Das ist entscheidend: sucht
    jemand nach "Eiffelturm", sollen alle Absaetze dieses Artikels treffen,
    nicht nur die, in denen das Wort zufaellig nochmal vorkommt. Ausserdem
    weiss das Sprachmodell dadurch immer, worum es im Kontext geht.
    """
    absaetze = [a.strip() for a in text.split("\n") if len(a.strip()) > 60]
    raus, puffer = [], ""
    for a in absaetze:
        if len(puffer) + len(a) < chunk_zeichen:
            puffer = f"{puffer} {a}".strip()
        else:
            if puffer:
                raus.append(f"{titel}: {puffer}"[:chunk_zeichen])
            puffer = a[:chunk_zeichen]
        if len(raus) >= max_pro_artikel:
            return raus
    if puffer:
        raus.append(f"{titel}: {puffer}"[:chunk_zeichen])
    return raus[:max_pro_artikel]


def bauen(quelle, ziel, max_artikel, chunk_zeichen, max_pro_artikel=3):
    import pyarrow.parquet as pq

    os.makedirs(ziel, exist_ok=True)
    chunks_pfad = os.path.join(ziel, "chunks.txt")

    # Mehrere Pakete moeglich: die deutsche Wikipedia liegt in zwanzig Teilen,
    # und ein einzelnes deckt nur einen Bruchteil der Artikel ab.
    quellen = sorted(glob.glob(quelle)) if any(z in quelle for z in "*?[") else [quelle]
    print(f"Lese {len(quellen)} Paket(e) ...")
    from array import array
    # array statt list: 4 Byte je Eintrag statt ~36 - sonst passt die ganze
    # Wikipedia nicht in den Arbeitsspeicher
    offsets, postings, titel_postings = array("Q"), defaultdict(lambda: array("I")), defaultdict(lambda: array("I"))
    t0, n, artikel = time.time(), 0, 0

    with open(chunks_pfad, "w", encoding="utf-8") as out:
        position = 0
        for pfad in quellen:
          pf = pq.ParquetFile(pfad)
          for batch in pf.iter_batches(batch_size=1000, columns=["title", "text"]):
            for titel, text in zip(batch.column("title").to_pylist(),
                                   batch.column("text").to_pylist()):
                artikel += 1
                titel_woerter = set(zerlege(titel))
                for stueck in stuecke(titel, text, chunk_zeichen, max_pro_artikel):
                    zeile_aus = stueck.replace("\n", " ") + "\n"
                    out.write(zeile_aus)
                    offsets.append(position)
                    position += len(zeile_aus.encode("utf-8"))
                    for w in set(zerlege(stueck)):
                        postings[w].append(n)
                    # Titelwoerter getrennt fuehren: wer nach "Eiffelturm" fragt,
                    # will den Eiffelturm-Artikel, nicht jeden Absatz, in dem das
                    # Wort nebenbei vorkommt.
                    for w in titel_woerter:
                        titel_postings[w].append(n)
                    n += 1
                if artikel % 20000 == 0:
                    print(f"  {artikel:>7,} Artikel | {n:>8,} Absaetze | "
                          f"{len(postings):>7,} Woerter | {time.time()-t0:5.0f}s",
                          flush=True)
                if max_artikel and artikel >= max_artikel:
                    break
            if max_artikel and artikel >= max_artikel:
                break
          if max_artikel and artikel >= max_artikel:
              break
    print(f"\n{artikel:,} Artikel aus {len(quellen)} Paket(en) verarbeitet")

    print(f"{n:,} Absaetze, {len(postings):,} verschiedene Woerter")

    # Sehr haeufige Woerter bringen beim Suchen nichts und blaehen den Index auf
    grenze = max(1000, int(n * 0.12))
    raus = [w for w, liste in postings.items() if len(liste) > grenze]
    for w in raus:
        del postings[w]
    print(f"{len(raus):,} zu haeufige Woerter entfernt (ueber {grenze:,} Treffer)")

    vokabular, alle = {}, array("I")
    for wort in sorted(postings):
        liste = postings[wort]
        vokabular[wort] = [len(alle), len(liste)]
        alle.extend(liste)

    titel_vok, titel_alle = {}, array("I")
    for wort in sorted(titel_postings):
        liste = titel_postings[wort]
        titel_vok[wort] = [len(titel_alle), len(liste)]
        titel_alle.extend(liste)

    np.frombuffer(offsets, dtype=np.uint64).tofile(os.path.join(ziel, "offsets.npy"))
    np.frombuffer(alle, dtype=np.uint32).tofile(os.path.join(ziel, "postings.npy"))
    np.frombuffer(titel_alle, dtype=np.uint32).tofile(os.path.join(ziel, "titel_postings.npy"))
    with open(os.path.join(ziel, "vokabular.json"), "w", encoding="utf-8") as f:
        json.dump({"n_chunks": n, "woerter": vokabular, "titel": titel_vok},
                  f, ensure_ascii=False)

    groesse = sum(os.path.getsize(os.path.join(ziel, d))
                  for d in os.listdir(ziel))
    print(f"\nIndex fertig in {ziel}/ ({groesse/1e6:.0f} MB, "
          f"{len(alle)/1e6:.1f} Mio. Eintraege)")
    print(f"Dauer: {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------- suchen

LEERWOERTER = {
    "was", "wer", "wie", "wo", "wann", "warum", "welche", "welcher", "welches",
    "ist", "sind", "war", "waren", "hat", "haben", "wird", "werden", "kann",
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
    "und", "oder", "von", "vom", "mit", "auf", "in", "im", "an", "am", "zu",
    "fuer", "für", "es", "man", "sich", "viel", "viele", "gibt", "heisst",
    "heißt", "lang", "hoch", "gross", "groß", "tief", "alt",
}


def inhaltswoerter(frage):
    """Die Woerter der Frage, die ueberhaupt etwas ueber das Thema aussagen."""
    return [w for w in zerlege(frage) if w not in LEERWOERTER and len(w) > 2]


class Wissensbasis:
    """Geladener Index. Beantwortet: welcher Absatz passt zu dieser Frage?"""

    def __init__(self, ordner=STANDARD):
        with open(os.path.join(ordner, "vokabular.json"), encoding="utf-8") as f:
            daten = json.load(f)
        self.woerter = daten["woerter"]
        self.titel = daten.get("titel", {})
        self.n = daten["n_chunks"]
        self.postings = np.fromfile(os.path.join(ordner, "postings.npy"),
                                    dtype=np.uint32)
        pfad = os.path.join(ordner, "titel_postings.npy")
        self.titel_postings = (np.fromfile(pfad, dtype=np.uint32)
                               if os.path.exists(pfad) else np.array([], np.uint32))
        self.offsets = np.fromfile(os.path.join(ordner, "offsets.npy"),
                                   dtype=np.uint64)
        self.datei = open(os.path.join(ordner, "chunks.txt"), "rb")

    def text(self, nummer: int) -> str:
        self.datei.seek(int(self.offsets[nummer]))
        return self.datei.readline().decode("utf-8", "ignore").strip()

    # Wie stark zaehlt es, dass zwei Inhaltswoerter der Frage im Absatz dicht
    # beieinander stehen? Gemessen ueber 1000 Kandidaten (60 handgeschriebene
    # und 200 umformulierte Fragen):
    #
    #   Gewicht   handgeschrieben k=2/k=3   umformuliert k=2/k=3
    #      0,0          46,7 / 46,7              84,5 / 96,0
    #      0,3          50,0 / 53,3              84,0 / 96,0
    #      0,7          51,7 / 55,0              83,0 / 95,5
    #      1,5          50,0 / 58,3              80,5 / 93,5
    #      3,0          51,7 / 56,7              71,0 / 86,0
    PAAR_GEWICHT = 0.7
    PAAR_FENSTER = 5       # Woerter Abstand, bis zu dem ein Paar zaehlt
    # So viele Kandidaten werden nach Paaren umsortiert. 200 statt 1000
    # (25.09.): jede Kandidatenlesung ist auf der SD ein zufaelliger Zugriff
    # (~1 ms); 1000 kosten auf dem ESP32 ~1,3 s, 200 ~0,3 s. Suche@3 auf dem
    # gestuften Index: MKQA-dev 23,0 -> 22,2, Mintaka-dev 26,0 -> 25,8,
    # eigene 195 49,7 -> 50,8 - im Rauschen.
    PAAR_TIEFE = 200

    TITEL_BONUS = 4.0      # ein Titeltreffer zaehlt so viel wie vier im Text
    # ... aber nur fuer seltene Woerter. Gemessen an zehn Fragen: "Wie viele
    # Einwohner hat Hamburg?" fand die Viele-Welten-Interpretation, "Wie lang
    # ist der Rhein?" einen Schriftsteller namens Thomas Lang. Ein einziger
    # zufaelliger Titeltreffer mit einem mittelhaeufigen Wort schlug jedes Mal
    # mehrere Treffer im Text zum richtigen Thema.
    TITEL_MIN_SELTENHEIT = 6.0

    def paar_punkte(self, text, inhaltswoerter, idf):
        """Stehen zwei Inhaltswoerter der Frage dicht beieinander?

        Der Fall, der das noetig machte: "Welcher Berg ist der hoechste
        Europas?" Der Elbrus-Absatz enthaelt alle drei Fragewoerter und stand
        trotzdem auf Rang 334 - "berg", "hoechste" und "europas" sind einzeln
        zu haeufig, um jemanden nach oben zu tragen. Als FOLGE sind sie
        selten. Die Reihenfolge darf dabei nicht zaehlen: im Text steht
        "der hoechste Berg Europas", in der Frage "Berg ... hoechste Europas".
        """
        ziel = set(inhaltswoerter)
        if len(ziel) < 2:
            return 0.0
        stellen = defaultdict(list)
        for i, w in enumerate(zerlege(text)):
            if w in ziel:
                stellen[w].append(i)
        s = 0.0
        for a, b in itertools.combinations(sorted(stellen), 2):
            abstand = min((abs(i - j) for i in stellen[a] for j in stellen[b]),
                          default=999)
            if abstand <= self.PAAR_FENSTER:
                s += (idf.get(a, 3.0) + idf.get(b, 3.0)) / (1 + abstand)
        return s

    def suche(self, frage: str, anzahl=3):
        """Die besten Absaetze zur Frage, als (Punkte, Text)."""
        punkte = defaultdict(float)
        idf = {}
        for wort in set(zerlege(frage)):
            # Seltene Woerter wiegen schwerer - das ist der Kern von IDF
            eintrag = self.woerter.get(wort)
            if eintrag:
                start, laenge = eintrag
                gewicht = math.log(self.n / laenge)
                idf[wort] = gewicht
                for nummer in self.postings[start:start + laenge]:
                    punkte[int(nummer)] += gewicht
            eintrag = self.titel.get(wort)
            if eintrag:
                start, laenge = eintrag
                # Das Gewicht eines Titeltreffers richtet sich nach der
                # Seltenheit des Wortes IM TEXT, nicht im Titel. Eine harte
                # Schranke war ein Abgrund: "rhein" fiel knapp darunter und
                # verlor seinen Bonus komplett, worauf der Bodensee gewann.
                im_text = self.woerter.get(wort)
                seltenheit = (math.log(self.n / im_text[1]) if im_text
                              else math.log(self.n / laenge))
                gewicht = seltenheit * self.TITEL_BONUS
                for nummer in self.titel_postings[start:start + laenge]:
                    punkte[int(nummer)] += gewicht

        # Nachsortierung. Zwei Signale stehen zur Verfuegung:
        #
        #   Titelanteil  - wie viele Woerter des Titels kommen in der Frage vor.
        #                  Ohne das gewinnt "Ludwigshafen am Rhein" gegen
        #                  "Rhein" und "Geboren am 4. Juli" gegen "Goethe".
        #   Paarnaehe    - stehen zwei Inhaltswoerter der Frage dicht
        #                  beieinander. Faengt die Faelle, in denen jedes
        #                  Einzelwort haeufig, die Folge aber selten ist.
        #
        # Die Tiefe ist bewusst gross: die Antwort steht bei handgeschriebenen
        # Fragen zu 85 % unter den besten 30, aber zu 98,3 % unter den besten
        # 1000. Der Titelanteil allein kann sie dort nicht heben.
        vor = sorted(punkte.items(), key=lambda x: -x[1])[:self.PAAR_TIEFE]
        woerter = set(zerlege(frage))
        iw = inhaltswoerter(frage)
        bewertet = []
        for nummer, p in vor:
            text = self.text(nummer)
            titel = text.split(":", 1)[0] if ":" in text[:80] else ""
            tw = set(zerlege(titel))
            t_anteil = len(tw & woerter) / len(tw) if tw else 0.0
            paare = self.paar_punkte(text, iw, idf)
            # Beides multiplikativ, nicht additiv: ein Absatz ohne jede
            # Wortuebereinstimmung soll auch mit perfekter Paarnaehe nicht
            # nach oben kommen. Die Signale verstaerken, was BM25 schon
            # gefunden hat - sie ersetzen es nicht.
            note = p * (1.0 + 2.0 * t_anteil) * (1.0 + self.PAAR_GEWICHT * paare)
            bewertet.append((note, nummer, text))
        bewertet.sort(key=lambda x: -x[0])
        return [(p, t) for p, _, t in bewertet[:anzahl]]

    def bester_text(self, frage: str, max_zeichen=450, absaetze=1):
        """Die besten Absaetze, gekuerzt - so wie sie in den Prompt passen.

        Mit `absaetze=2` kommen zwei statt einem. Gemessen an zehn Fragen:
        mit einem Absatz steht die Antwort in sieben Faellen drin, mit zweien
        in neun - die drei Fehlschlaege waren durchweg "richtiger Artikel,
        falscher Absatz". Das Modell ist auf mehrere Absaetze trainiert und
        muss ohnehin den passenden heraussuchen; ihm zwei zu geben ist
        deshalb billiger als die Suche weiter zu verfeinern.
        """
        treffer = self.suche(frage, anzahl=absaetze)
        if not treffer:
            return None
        # Absaetze, in denen eine Angabe der GESUCHTEN ART steht, nach vorn.
        # Gemessen am 23.09. auf 383 Fragen: das hebt die Antwort von 51,4 auf
        # 80,4 % auf Platz eins und die Trefferquote von 49,6 auf 52,2 %.
        # Die Rangfolge der Suche bleibt innerhalb beider Gruppen erhalten -
        # umsortiert wird nur zwischen "kann antworten" und "kann nicht".
        if len(treffer) > 1:
            try:
                from regeln_breit import antwort as _art
                mit = [t for t in treffer if _art(frage, t[1])]
                if mit and len(mit) < len(treffer):
                    ohne = [t for t in treffer if not _art(frage, t[1])]
                    treffer = mit + ohne
            except Exception:
                pass          # ohne Regeln lieber unsortiert als gar nichts
        je = max_zeichen // max(len(treffer), 1)
        teile = []
        for _, text in treffer:
            if len(text) > je:
                schnitt = text.rfind(". ", 0, je)
                text = text[:schnitt + 1] if schnitt > 120 else text[:je]
            teile.append(text)
        return " ".join(teile)


def main():
    ap = argparse.ArgumentParser()
    unter = ap.add_subparsers(dest="befehl", required=True)

    b = unter.add_parser("bauen")
    b.add_argument("--quelle", default="data/parquet/train-00000-of-00020.parquet")
    b.add_argument("--ziel", default=STANDARD)
    b.add_argument("--max-artikel", type=int, default=500000, help="0 = alle")
    b.add_argument("--chunk-zeichen", type=int, default=600)
    b.add_argument("--max-pro-artikel", type=int, default=3,
                   help="wie viele Absaetze pro Artikel in den Index")

    s = unter.add_parser("suchen")
    s.add_argument("frage", nargs="+")
    s.add_argument("--ordner", default=STANDARD)
    s.add_argument("--anzahl", type=int, default=3)

    args = ap.parse_args()
    if args.befehl == "bauen":
        bauen(args.quelle, args.ziel, args.max_artikel, args.chunk_zeichen,
              args.max_pro_artikel)
    else:
        wb = Wissensbasis(args.ordner)
        frage = " ".join(args.frage)
        t0 = time.time()
        treffer = wb.suche(frage, args.anzahl)
        dauer = (time.time() - t0) * 1000
        print(f'\n"{frage}"   ({dauer:.0f} ms, {wb.n:,} Absaetze durchsucht)\n')
        for i, (punkte, text) in enumerate(treffer, 1):
            print(f"{i}. [{punkte:.1f}] {text[:300]}\n")


if __name__ == "__main__":
    main()
