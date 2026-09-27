"""fakten.py — Fragen direkt aus Wikidata-Tabellen beantworten.

Gemessen am 24.09.2026: Die Wikipedia-Kopie (20231101.de) hat die Vorlagen
entfernt, darin standen Hoehen, Tiefen und Laengen ("Die Zugspitze ist mit  der
hoechste Gipfel"). Auch ein Index ueber die ganze Wikipedia (8 GB) findet die
Zahl deshalb nicht. Die Zahlen liegen aber sauber in Wikidata.
`fakten_holen.sh` laedt sie als CSV, dieses Modul schlaegt nach:
Gegenstand + Merkmal -> Antwort.

Stufen, in dieser Reihenfolge (die ersten vier sind Sonderformen, die die
Tabellen-Stufe falsch beantworten wuerde):

    einheiten    "Wie viele Sekunden hat eine Stunde?"   -> Rechnung
    nennt_man    "Welche Stadt nennt man die Ewige Stadt?" -> Kurzname-Ziel
    fluss_rueck  "Welche Stadt liegt an der Elbe?"       -> Fluss rueckwaerts
    wer_war      "Wer war Marie Curie?"                  -> erster Satz (leitsatz.tsv)
    werke        "Wer singt i can't fight this feeling"  -> fakten_werk.py (en+de Titel)
    tabelle      Gegenstand + Merkmal                     -> Wert

Gemessen am 25.09. (strenger Bewerter, gestufter Index, --regeln2 --fakten):
die Erweiterungen gegenueber der ersten Fassung heben eigene 195 45,6 -> 53,8,
neu60 68,3 -> 71,7, hand 55,0 -> 60,0; MKQA-dev/Mintaka-dev bleiben gleich
(+-1 Frage). Sie helfen deutschen Alltagsfragen, nicht den US-lastigen Saetzen.

Auf dem ESP32 ist jede Tabelle eine sortierte Datei auf der SD (Titel ->
Werte), gesucht per Binaersuche; kein Sprachmodell noetig.

    python fakten.py "Wie hoch ist die Zugspitze?"
    python fakten.py --leitsaetze wiki_gestuft      # baut fakten/leitsatz.tsv
"""
import csv
import os
import re
from collections import defaultdict

import fakten_wahl
from fakten_werk import Werke

# Merkmal: (Datei, Fragewoerter, Einheit, Auswahl)
MERKMALE = [
    ("tiefe", r"\bwie tief\b|\btiefe\b", "Meter", max),
    # Bauhoehe (P2048) vor Hoehe ueber dem Meer (P2044): der Schiefe Turm
    # ist 56 m hoch, steht aber auf 210 m
    ("bauhoehe", r"\bwie hoch\b|\bhöhe\b", "Meter", max),
    ("hoehe", r"\bwie hoch\b|\bhöhe\b", "Meter", max),
    ("laenge", r"\bwie lang\b|\blänge\b", "", max),
    ("flaeche", r"\bfläche\b|\bwie groß ist die fläche\b", "Quadratkilometer", max),
    ("einwohner", r"\beinwohner\b|\bwie viele menschen\b", "Einwohner", max),
    ("hauptstadt", r"\bhauptstadt\b", "", None),
    ("waehrung", r"\bwährung\b|\bzahlt man\b", "", None),
    ("sprache", r"\bsprache\b|\bspricht man\b", "", None),
    ("komponist", r"\bkomponi", "", None),
    ("erfinder", r"\berfunden\b|\berfand\b", "", None),
    ("gruendung", r"\bgegründet\b|\bgründung\b", "", min),
    ("gruender", r"\bwer gründete\b|\bgründer\b", "", None),
    ("geburt", r"\bgeboren\b", "", min),
    ("autor", r"\bwer schrieb\b|\bwer hat .* geschrieben\b|\bautor\b", "", None),
    ("fluss", r"\bfluss\b.*\bdurch\b|\ban welchem fluss\b", "", None),
    ("durchmesser", r"\bdurchmesser\b", "", max),
    # "Wann wurde der Reichstag gebaut?" - Bauwerke haben ein Gruendungsdatum (P571)
    ("gruendung", r"\bwann wurde\b.*\b(gebaut|erbaut|errichtet|erfunden|eröffnet|verabschiedet)\b",
     "", min),
    # Ereignisse und Lebensdaten. Wikidata fuehrt "sinken" und "fallen" als
    # Ende der Existenz (P576): Titanic (Schiff) 15.04.1912, Berliner Mauer
    # 09.11.1989. "Ende" (P582) steht vor der Aufloesung, damit "Wann endete
    # der Erste Weltkrieg?" das Kriegsende nimmt.
    ("tod", r"\bwann (?:starb|ist .+ gestorben)\b|\btodesjahr\b", "", min),
    ("beginn", r"\bwann (?:begann|startete|fing)\b", "", min),
    ("ende", r"\bwann (?:endete|war .+ zu ende|hörte)\b", "", min),
    ("aufloesung", r"\bwann (?:sank|versank|fiel|endete|wurde .+ (?:aufgelöst|zerstört|"
                   r"abgerissen|versenkt|abgeschafft|stillgelegt))\b", "", min),
    ("zeitpunkt", r"\bwann (?:fand|geschah|passierte|ereignete)\b", "", min),
    # zuletzt: "Wie groß ist Frankreich?" meint die Flaeche
    ("flaeche", r"\bwie groß ist\b", "Quadratkilometer", max),
]
JAHRE = {"gruendung", "geburt", "tod", "beginn", "ende", "aufloesung", "zeitpunkt"}
NAMENSWERTE = {"hauptstadt", "waehrung", "sprache", "komponist", "erfinder",
               "gruender", "autor", "fluss"}
ARTIKEL = {"der", "die", "das", "den", "dem", "des"}
# Jahresmerkmale antworten nur, wenn nach einer Zeit gefragt ist. Gemessen am
# 25.09. auf Mintaka-dev: "In welchem Staat war der Cheftrainer ... geboren?"
# bekam ein Jahr, Genauigkeit der Faktenstufe dort nur 30 %.
ZEITFRAGE = re.compile(r"^(wann|in welchem jahr|seit wann|um welches jahr)\b")
# "Wann wurde X geboren?" - X muss GENAU der Gegenstand sein. Sonst beantwortet
# "Wann wurde der Autor der Hungerspiele geboren?" das Geburtsjahr irgendeines
# anderen Namens aus der Frage.
DATUMSFRAGE = re.compile(
    r"^(?:wann|in welchem jahr) (?:wurde|ist|war|wurden|sind) (.+?) "
    r"(?:geboren|gegründet|gebaut|erbaut|errichtet|erfunden|eröffnet|verabschiedet|entstanden|"
    r"gestorben|aufgelöst|zerstört|abgerissen|versenkt|abgeschafft|stillgelegt|zu ende)\b"
    r"|^wann (?:sank|versank|fiel|begann|endete|starb|startete) (.+?)(?: an)?$"
    r"|^wann (?:fand|geschah|passierte|ereignete sich) (.+?)(?: statt)?$")
# Auch "drittgroesste", "zweitlaengste": kein \b vor dem Wortstamm
SUPERLATIV = re.compile(r"(meisten|größte|längste|höchste|kleinste|tiefste|älteste|"
                        r"jüngste|bevölkerungsreichste)[nmrs]?\b")
# Weitere Faelle, in denen der Rest der Frage GENAU der Gegenstand sein muss
# (Mehrschrittfragen aus Mintaka: "Wer schrieb die Buecher, auf denen die
# HBO-Serie ... basiert?", "Wie viele Einwohner hat Karatschi, Pakistan, ...")
WERFRAGE = re.compile(r"^wer (?:schrieb|komponierte|erfand|gründete|hat) (.+?)"
                      r"(?: geschrieben| komponiert| erfunden| gegründet)?$")
MENGENFRAGE = re.compile(r"^wie viele (?:einwohner|menschen) (?:hat|leben in|wohnen in) (.+?)"
                         r"(?:,| gemäß| laut| im jahr| heute|$)")
SPANNE = {"autor": WERFRAGE, "komponist": WERFRAGE, "erfinder": WERFRAGE,
          "gruender": WERFRAGE, "einwohner": MENGENFRAGE}

# Nie selbst Gegenstand: "die" ist auch ein Artikeltitel
KEIN_GEGENSTAND = {"die", "der", "das", "den", "dem", "des", "ein", "eine", "wann",
    "wer", "wie", "was", "wo", "welche", "welcher", "welches", "wurde", "ist",
    "hat", "hoch", "tief", "lang", "groß", "viele", "gibt", "es", "von", "in",
    "am", "im", "an", "auf", "man", "wir", "sind", "war", "heißt"}

ERLAUBT = {"hauptstadt", "fläche", "einwohner", "sprache", "amtssprache",
           "währung", "tiefe", "höhe", "länge", "menschen", "stadt", "land",
           "fluss", "durchmesser"}

ZEIT = {"sekunde": 1, "minute": 60, "stunde": 3600, "tag": 86400,
        "woche": 604800, "monat": 2629746, "jahr": 31556952}
PLURAL = {"sekunden": "sekunde", "minuten": "minute", "stunden": "stunde",
          "tage": "tag", "wochen": "woche", "monate": "monat", "jahre": "jahr"}
ABK = {"v", "chr", "ca", "geb", "st", "dr", "u", "z", "bzw", "n", "jh", "gest"}


def jahr(werte):
    j = min(int(w.split("-")[0] or -int(w.split("-")[1])) for w in werte)
    return f"{1 - j} v. Chr." if j <= 0 else str(j)   # astronomisch: -752 = 753 v. Chr.


def erster_satz(text, mit_titel=True):
    """Erster Satz ohne Klammern; "v. Chr." und "15. Mai" beenden ihn nicht."""
    t = text.split(": ", 1)[1] if mit_titel and ": " in text[:120] else text
    for _ in range(3):
        t = re.sub(r"\s*\([^()]*\)", "", t)
    for m in re.finditer(r"\.\s+(?=[A-ZÄÖÜ])", t):
        vor = re.findall(r"[\wäöüß]+$", t[:m.start()])
        if vor and (vor[0].lower() in ABK or vor[0].isdigit()):
            continue
        if m.start() > 30:
            return t[:m.start() + 1]
    return t[:200]


def zahl_text(zahl, einheit):
    if zahl >= 1e9:
        s = f"{zahl/1e9:.2f} Milliarden".replace(".", ",")
    elif zahl >= 1e6:
        s = f"{zahl/1e6:.2f} Millionen".replace(".", ",")
    elif zahl >= 10000:
        s = f"{zahl:,.0f}".replace(",", ".")          # 357.588 statt 357587,8
    elif zahl != int(zahl):
        s = f"{zahl:.1f}".replace(".", ",")
    else:
        s = f"{zahl:.0f}"
    return f"{s} {einheit}".strip()


def lies_csv(pfad):
    if not os.path.exists(pfad):
        return
    with open(pfad, encoding="utf-8") as f:
        for zeile in csv.reader(f):
            if len(zeile) == 2 and zeile[0] != "t":
                yield zeile


class Fakten:
    def __init__(self, ordner="fakten"):
        self.tab = defaultdict(lambda: defaultdict(list))
        self.schreibweise = {}
        for name in {m[0] for m in MERKMALE}:
            for t, v in lies_csv(os.path.join(ordner, f"{name}.csv")):
                self.tab[t.lower()][name].append(v)
                self.schreibweise.setdefault(t.lower(), t)
        # Kurznamen (UNO, DDR) -> Titel; nur Ziele, zu denen es Fakten gibt
        self.alias = defaultdict(list)
        for kurz, t in lies_csv(os.path.join(ordner, "alias.csv")):
            if t.lower() in self.tab:
                self.alias[kurz.lower()].append(t.lower())
        # Rueckwaerts: Elementsymbol -> Titel ("Welches Element hat das Zeichen Fe?")
        self.symbol = {v: t for t, v in lies_csv(os.path.join(ordner, "symbol.csv"))}
        # Bekanntheit = Zahl der Sprachversionen. Entscheidet bei mehrdeutigen
        # Namen ("Bach" meint Johann Sebastian) UND bei mehreren Namenswerten
        # (Erfinder der Gluehbirne: Edison vor Swan).
        namen = set(self.tab)
        for merkmale in self.tab.values():
            for name in NAMENSWERTE & set(merkmale):
                namen.update(w.lower() for w in merkmale[name])
        self.bekannt = {}
        for t, v in lies_csv(os.path.join(ordner, "bekanntheit.csv")):
            if v.isdigit() and t.lower() in namen:
                self.bekannt[t.lower()] = max(self.bekannt.get(t.lower(), 0), int(v))
        for t, merkmale in self.tab.items():
            if "geburt" in merkmale and " " in t and "(" not in t:
                self.alias[t.rsplit(" ", 1)[1]].append(t)
        self.titel = set(self.tab) | set(self.alias)
        # Zahlen mit Rang, Einheit und Stichtag (fakten_rang/, optional)
        self.rang = fakten_wahl.lade(os.path.join(os.path.dirname(ordner) or ".", "fakten_rang"))
        # "Wer war X?" -> erster Satz des Artikels (optional, python fakten.py --leitsaetze)
        self.leitsatz = {}
        pfad = os.path.join(ordner, "leitsatz.tsv")
        if os.path.exists(pfad):
            with open(pfad, encoding="utf-8") as f:
                for z in f:
                    t, _, satz = z.rstrip("\n").partition("\t")
                    self.leitsatz[t] = satz
        self._flussrueck = None
        # Klassen und Lage fuer Ranglisten (fakten_rangliste/, optional)
        self.klasse = defaultdict(set)          # klasse -> {titel}
        self.lage = defaultdict(set)            # titel -> {kontinent, staat, gebirge, gewaesser}
        self.staaten = defaultdict(set)         # titel -> {staat} (Fluesse: nur ganz im Land)
        rl = os.path.join(os.path.dirname(ordner) or ".", "fakten_rangliste")
        for datei in ("klasse.csv", "klasse2.csv"):
            for t, k in lies_csv(os.path.join(rl, datei)):
                self.klasse[k].add(t.lower())
        if os.path.exists(os.path.join(rl, "planet.csv")):
            with open(os.path.join(rl, "planet.csv"), encoding="utf-8") as f:
                self.klasse["planet"] = {z.strip().lower() for z in f if z.strip() and z.strip() != "t"}
        kontinent_von = defaultdict(set)
        for t, v in lies_csv(os.path.join(rl, "kontinent_von.csv")):
            kontinent_von[t.lower()].add(v.lower())
            self.lage[t.lower()].add(v.lower())
        gebirge = defaultdict(set)
        for t, v in lies_csv(os.path.join(rl, "gebirge_von.csv")):
            gebirge[t.lower()].add(v.lower())
        # Gebirge sind verschachtelt: Mont Blanc -> Mont-Blanc-Gruppe ->
        # Grajische Alpen -> Westalpen -> Alpen. Ohne die Kette nach oben
        # waere der Mont Blanc kein Berg "der Alpen".
        for t in list(gebirge):
            offen, gesehen = list(gebirge[t]), set()
            while offen and len(gesehen) < 20:
                g = offen.pop()
                if g not in gesehen:
                    gesehen.add(g)
                    offen.extend(gebirge.get(g, ()))
            self.lage[t] |= gesehen
        for t, v in lies_csv(os.path.join(rl, "staat_von.csv")):
            self.lage[t.lower()].add(v.lower())
            self.staaten[t.lower()].add(v.lower())
        # Kontinent ueber den Staat: Mount Everest, Denali, Aconcagua und der Nil
        # haben in Wikidata keinen eigenen Kontinent (P30), ihre Staaten schon.
        # Nur Staaten auf GENAU einem Kontinent vererben: ueber Russland oder die
        # Sowjetunion wurden sonst die Lena und der Pik Ibn Sina "europaeisch".
        for t, staaten in self.staaten.items():
            for st in staaten:
                if len(kontinent_von.get(st, ())) == 1:
                    self.lage[t] |= kontinent_von[st]
        for t, merkmale in self.tab.items():       # P206: Inseln im Mittelmeer
            for v in merkmale.get("fluss", []):
                self.lage[t].add(v.lower())
        self.orte = set().union(*self.lage.values()) if self.lage else set()
        # Werke mit englischen und deutschen Titeln (fakten_werk_holen.sh)
        self.werke = Werke(os.path.join(os.path.dirname(ordner) or ".", "fakten_werk"))

    # ---------------------------------------------------------- Gegenstand

    def kandidaten(self, frage):
        """Alle Titel/Kurznamen in der Frage, mit Zahl ihrer Inhaltswoerter.

        Artikelvarianten: "die Weser" findet auch "Weser", "der Schiefe Turm
        von Pisa" auch "Schiefer Turm von Pisa" nicht - dafuer gibt es Aliase.
        """
        text = frage.lower().rstrip("?!. ")
        w = re.findall(r"[\wäöüß\-\.]+", text)
        out = []
        for i in range(len(w)):
            for j in range(min(len(w), i + 6), i, -1):
                kand = " ".join(w[i:j])
                var = [kand]
                if kand.endswith("s"):
                    var.append(kand[:-1])
                if w[i] in ARTIKEL and j - i > 1:
                    rest = " ".join(w[i + 1:j])
                    var += [a + " " + rest for a in ("der", "die", "das")] + [rest]
                    # "der Erste Weltkrieg" -> Titel "Erster Weltkrieg",
                    # "das Deutsche Reich" -> "Deutsches Reich": nach dem Artikel
                    # steht das Adjektiv schwach gebeugt, im Titel stark.
                    if j - i > 2 and w[i + 1].endswith("e"):
                        for endung in ("r", "s"):
                            var.append(" ".join([w[i + 1] + endung] + w[i + 2:j]))
                inhalt = sum(1 for x in w[i:j] if x not in ARTIKEL)
                for k in var:
                    if (k in self.titel and k not in KEIN_GEGENSTAND
                            and (j - i > 1 or w[i] not in KEIN_GEGENSTAND)):
                        out.append((k, inhalt, i, j))
        # Teilstuecke eines laengeren direkten Titels sind Namensteile
        # ("julia" in "Romeo und Julia") und keine eigenen Gegenstaende
        spannen = [(i, j) for k, _, i, j in out if k in self.tab]
        return [(k, inh) for k, inh, i, j in out
                if not any(a <= i and j <= b and (b - a) > (j - i) for a, b in spannen)]

    @staticmethod
    def davor_sperrt(frage, g):
        """"Wie hoch ist der Fernsehturm in Berlin?" fragt nach dem Turm, nicht
        nach Berlin: steht vor "in/von <Gegenstand>" ein weiteres Hauptwort,
        ist der gefundene Titel nur die Ortsangabe."""
        m = re.search(r"\b(?:in|von|der|des|für|zu|zum|zur)\s+(?:der\s+|dem\s+|den\s+)?" + re.escape(g),
                      frage.lower())
        if m:
            davor = re.findall(r"\b[A-ZÄÖÜ][\wäöüß]+", frage[len(frage.split()[0]):m.start()])
            return any(x.lower() not in ERLAUBT for x in davor)
        return False

    @staticmethod
    def deckt(k, rest):
        """Ist der Kandidat der ganze Rest (bis auf Artikel, Genitiv-s und
        Adjektivendung: "der erste weltkrieg" deckt "erster weltkrieg")?"""
        def grund(p):
            p = re.sub(r"^(der|die|das|den|dem|des) ", "", p.strip())
            return " ".join(re.sub(r"e[rsnm]$", "e", w) for w in p.split())
        rest, k = grund(rest), grund(k)
        return k == rest or k == rest.rstrip("s") or k + "s" == rest

    # ---------------------------------------------------------- Werte

    def formatiere(self, titel, name, einheit, wahl):
        werte = self.tab[titel][name]
        if name in JAHRE:
            return jahr(werte)
        if wahl:
            # Wikidata sagt selbst, welcher Wert gilt: bevorzugter Rang, dann
            # juengster Stichtag (Einwohner), mit Einheit (km vs. m).
            eintraege = self.rang.get(titel, {}).get(name)
            if eintraege:
                wert, e = fakten_wahl.waehle(eintraege)[:2]
                return zahl_text(wert, e or einheit)
            return zahl_text(wahl(float(x) for x in werte), einheit)
        if name in NAMENSWERTE:
            if name == "fluss":   # Meere und Seen raus: Fluesse haben eine Laenge
                fl = [x for x in werte if self.tab.get(x.lower(), {}).get("laenge")
                      and not self.tab.get(x.lower(), {}).get("flaeche")]
                werte = fl or werte
            return max(werte, key=lambda x: self.bekannt.get(x.lower(), 0))
        return werte[0]

    def tabelle(self, frage):
        text = frage.lower()
        # fragt nach dem Gegenstand, nicht nach dem Wert - ausser "an der
        # tiefsten Stelle" (Merkmal EINES Gegenstands)
        if SUPERLATIV.search(text) and not re.search(
                r"\ban (?:der|seiner|ihrer) \w+sten stelle\b", text):
            return None
        if re.search(r"\b(zeichen|symbol)\b", text):
            for x in re.findall(r"\b[A-Z][a-z]?\b", frage):
                if x in self.symbol:
                    return self.symbol[x]
        zeitfrage = bool(ZEITFRAGE.match(text))
        passend = [(r, m) for r, m in enumerate(MERKMALE) if re.search(m[1], text)
                   and not (m[0] == "erfinder" and not text.startswith("wer"))
                   and (m[0] not in JAHRE or zeitfrage)]
        d = DATUMSFRAGE.match(text.rstrip("?!. "))
        datums_rest = next((g for g in d.groups() if g), None) if d else None
        best = None
        for k, inhalt in self.kandidaten(frage):
            if self.davor_sperrt(frage, k):
                continue
            for r, (name, _, einheit, wahl) in passend:
                if name in JAHRE and datums_rest is not None and not self.deckt(k, datums_rest):
                    continue
                if name in SPANNE:
                    sp = SPANNE[name].match(text.rstrip("?!. "))
                    if sp and not self.deckt(k, sp.group(1)):
                        continue
                direkt = k in self.tab and bool(self.tab[k].get(name))
                ziele = [k] if direkt else [z for z in self.alias.get(k, []) if self.tab[z].get(name)]
                # Kurzname nur, wenn sein Ziel bekannter ist als der gleichnamige
                # direkte Titel: "Leipzig" ohne Gruendungsdatum darf nicht zum
                # Datum eines unbekannten Namensvetters werden.
                if not direkt and k in self.tab:
                    ziele = [z for z in ziele if self.bekannt.get(z, 0) > self.bekannt.get(k, 0)]
                if not ziele:
                    continue
                z = max(ziele, key=lambda z: self.bekannt.get(z, 0))
                # Mehr Inhaltswoerter schlagen weniger, direkter Titel schlaegt
                # Kurznamen, dann Bekanntheit, dann Reihenfolge der Merkmale
                note = (inhalt, direkt, self.bekannt.get(z, 0), -r)
                if best is None or note > best[0]:
                    best = (note, z, name, einheit, wahl)
                break          # erstes passendes Merkmal dieses Kandidaten
        if best:
            _, z, name, einheit, wahl = best
            return self.formatiere(z, name, einheit, wahl)
        return None

    # ---------------------------------------------------------- Sonderformen

    @staticmethod
    def einheiten(frage):
        m = re.search(r"wie viele (\w+) hat (?:ein|eine|einen) (\w+)", frage.lower())
        if not m:
            return None
        klein, gross = PLURAL.get(m.group(1), m.group(1)), m.group(2)
        if klein in ZEIT and gross in ZEIT and ZEIT[gross] > ZEIT[klein]:
            return str(int(ZEIT[gross] / ZEIT[klein]))
        return None

    def nennt_man(self, frage):
        text = frage.lower().rstrip("?!. ")
        if not re.search(r"nennt man|genannt|bezeichnet man|spitzname", text):
            return None
        w = re.findall(r"[\wäöüß\-\.]+", text)
        best = None
        for i in range(len(w)):
            for j in range(i + 2, min(len(w), i + 6) + 1):
                k = " ".join(w[i:j])
                if k in self.alias and (best is None or len(k) > len(best)):
                    best = k
        if best:
            return max(self.alias[best], key=lambda z: self.bekannt.get(z, 0)).title()
        return None

    def fluss_rueck(self, frage):
        m = re.search(r"welche stadt liegt an (?:der |dem |den )?([\wäöüß\-]+)", frage.lower())
        if not m:
            return None
        if self._flussrueck is None:
            self._flussrueck = defaultdict(list)
            for t, mm in self.tab.items():
                for v in mm.get("fluss", []):
                    if mm.get("einwohner") and not mm.get("hauptstadt") and not mm.get("waehrung"):
                        self._flussrueck[v.lower()].append(t)
        staedte = self._flussrueck.get(m.group(1))
        if not staedte:
            return None
        return max(staedte, key=lambda t: max(float(x) for x in self.tab[t]["einwohner"])).title()

    def wer_war(self, frage):
        m = re.match(r"^\s*wer (?:war|ist) (.+?)\s*\??$", frage.lower())
        if not m or not self.leitsatz:
            return None
        x = m.group(1)
        for k in (x, re.sub(r"^(der|die|das) ", "", x)):
            if k in self.leitsatz:
                return self.leitsatz[k]
        return None

    # ---------------------------------------------------------- Vergleiche

    # Steigerungswort -> (Merkmal, True = groesserer Wert gewinnt). Nur was die
    # Tabellen wirklich tragen; Oscars, Titel, Tore stehen nicht darin
    # (Mintaka-dev: gut ein Viertel der 44 Vergleichsfragen ist so loesbar).
    VERGLEICH = [
        (r"\bälter\b|\blebte zuerst\b|\bzuerst geboren\b|\bfrüher geboren\b", "geburt", False),
        (r"\bjünger\b|\bspäter geboren\b", "geburt", True),
        (r"\bhöher(?:e[rsnm]?)?\b", "hoehe", True),
        (r"\btiefer(?:e[rsnm]?)?\b", "tiefe", True),
        (r"\blänger(?:e[rsnm]?)?\b", "laenge", True),
        (r"\bbevölkerung|\bmehr einwohner\b|\bbevölkerungsreicher", "einwohner", True),
        (r"\bgrößer(?:e[rsnm]?)?\b|\bflächenmäßig\b", "flaeche", True),
        (r"\bzuerst gegründet\b|\bfrüher gegründet\b|\bälter als firma\b", "gruendung", False),
    ]
    OPTION_STOP = re.compile(r"\s+(?:ein|eine|einen|zuerst|früher|später|länger|höher|größer|"
                             r"älter|jünger|mehr|weniger|flächenmäßig)\b.*$")

    def optionen(self, frage):
        """"Welcher Kontinent ist größer, Afrika oder Europa?" -> ("afrika", "europa")."""
        text = frage.lower().rstrip("?!. ")
        if " oder " not in text:
            return None
        links, rechts = text.rsplit(" oder ", 1)
        # Listenform: nach dem letzten Trenner steht die erste Option
        for trenner in (", ", " - ", ": ", " – "):
            if trenner in links:
                links = links.rsplit(trenner, 1)[1]
                break
        else:
            # "Ist der Kangchenjunga oder der Lhotse ein höherer Berg?"
            links = re.sub(r"^(?:ist|war|sind|waren|hat|hatte|lebte)\s+", "", links)
        rechts = self.OPTION_STOP.sub("", rechts.split(", ")[0])
        art = r"^(?:der|die|das|den|dem|des|mit den|mit dem|im|in der|in dem)\s+"
        a, b = re.sub(art, "", links.strip()), re.sub(art, "", rechts.strip())
        return (a, b) if a and b and a != b else None

    def wert_von(self, option, name):
        """Tabellenwert einer Option: direkter Titel, sonst bekanntester Kurzname."""
        for k in (option, option.rstrip("s")):
            if self.tab.get(k, {}).get(name):
                return k
            ziele = [z for z in self.alias.get(k, []) if self.tab[z].get(name)]
            if ziele:
                return max(ziele, key=lambda z: self.bekannt.get(z, 0))
        return None

    def vergleich(self, frage):
        text = frage.lower()
        merkmal = next(((n, groesser) for m, n, groesser in self.VERGLEICH
                        if re.search(m, text)), None)
        paar = self.optionen(frage) if merkmal else None
        if not paar:
            return None
        name, groesser = merkmal
        titel = [self.wert_von(o, name) for o in paar]
        if not all(titel):
            return None                  # nie raten: beide Werte muessen da sein
        if name in JAHRE:
            werte = [int(jahr(self.tab[t][name]).split()[0]) * (-1 if "v. Chr." in jahr(self.tab[t][name]) else 1)
                     for t in titel]
        else:
            werte = [max(float(x) for x in self.tab[t][name]) for t in titel]
        if werte[0] == werte[1]:
            return None
        sieger = paar[0] if (werte[0] > werte[1]) == groesser else paar[1]
        # Die Antwort ist eine der beiden Optionen - so, wie sie in der Frage
        # steht. Der aufgeloeste Titel kann ein Namensvetter sein ("Nick
        # Carter" -> "Murs (Rapper)", buergerlich Nick Carter).
        i = frage.lower().find(sieger)
        return frage[i:i + len(sieger)] if i >= 0 else sieger

    # ---------------------------------------------------------- Ranglisten

    KLASSE_WORT = {"berg": "berg", "gipfel": "berg", "fluss": "fluss", "strom": "fluss",
                   "see": "see", "land": "land", "staat": "land", "stadt": "stadt",
                   "insel": "insel", "ozean": "ozean", "kontinent": "kontinent",
                   "erdteil": "kontinent", "planet": "planet"}
    WELT = re.compile(r"\b(?:der welt|der erde|weltweit|im sonnensystem|unseres sonnensystems|"
                      r"auf der erde|aller zeiten)\b")

    def steigerung(self, text, klasse):
        """Superlativ -> (Merkmal, True = groesster Wert)."""
        if re.search(r"\b(?:meisten einwohner|bevölkerungsreichste)", text):
            return "einwohner", True
        if re.search(r"\bgrößte fläche\b", text):
            return "flaeche", True
        for wort, merkmal, groesser in (("höchste", "hoehe", True), ("längste", "laenge", True),
                                        ("tiefste", "tiefe", True), ("größte", None, True),
                                        ("kleinste", None, False)):
            if re.search(wort + r"[nmrs]?\b", text):
                if merkmal:
                    return merkmal, groesser
                if klasse == "stadt":
                    return "einwohner", groesser
                if klasse == "planet":
                    return "durchmesser", groesser
                return "flaeche", groesser
        return None

    GRUNDEINHEIT = {"Kilometer": 1000.0}     # Laengen vergleichen in Metern
    # Welche Einheit zu welchem Merkmal passt. Der Kompienga-Stausee hat in
    # Wikidata eine Flaeche "in Metern" (175 Mio.) und schlug so den Victoriasee.
    PASSEND = {"flaeche": {"Quadratkilometer"}, "laenge": {"Meter", "Kilometer"},
               "hoehe": {"Meter", "Kilometer"}, "bauhoehe": {"Meter", "Kilometer"},
               "tiefe": {"Meter", "Kilometer"}, "durchmesser": {"Meter", "Kilometer"}}
    # Zusammengesetzte Grossraeume, die Wikidata als Kontinent oder Ozean fuehrt
    AUSSCHLUSS = {"afrika-eurasien", "amerika", "australien und ozeanien", "eurasien",
                  "weltmeere", "weltozean"}

    def zahl(self, t, name):
        """Wert in Grundeinheit - nur mit bekannter Einheit (fakten_rang/).

        Rohwerte ohne Einheit taugen nicht fuer Ranglisten: ein Fluss mit
        Laenge in Metern schlug den Jangtse in Kilometern, ein Stausee mit
        Flaeche in Quadratmetern den Victoriasee (25.09.).
        """
        if name == "einwohner":          # Einwohner haben keine Einheit
            eintraege = self.rang.get(t, {}).get(name)
            return fakten_wahl.waehle(eintraege)[0] if eintraege else None
        eintraege = [e for e in self.rang.get(t, {}).get(name, [])
                     if e[1] in self.PASSEND.get(name, {e[1]})]
        if not eintraege:
            return None
        wert, einheit = fakten_wahl.waehle(eintraege)[:2]
        return wert * self.GRUNDEINHEIT.get(einheit, 1.0)

    ORDINAL = {"zweit": 2, "dritt": 3, "viert": 4, "fünft": 5}
    # Woerter, die eine einfache Ranglistenfrage enthalten darf. Alles andere
    # ("einer der fuenf", "nicht", "in Arizona" ohne erkannten Ort, "auf dem
    # afrikanischen Kontinent") heisst: schweigen. Mintaka-dev 25.09.: ohne diese
    # Pruefung 18 Antworten, 2 richtig.
    FUELLWORT = {"welche", "welcher", "welches", "welchen", "was", "wie", "heißt", "heisst",
                 "ist", "sind", "war", "der", "die", "das", "den", "dem", "des", "in", "im",
                 "auf", "von", "hat", "hatte", "es", "gibt", "am", "meisten", "einwohner",
                 "fläche", "welt", "erde", "weltweit", "sonnensystem", "unseres", "aller",
                 "zeiten", "nach", "flächenmäßig", "gemessen", "an"}

    def rangliste(self, frage):
        """"Welcher Berg ist der höchste Europas?" -> vorberechenbar: Klasse,
        Merkmal, Bereich. Auf dem ESP32 eine kleine Tabelle der Top-3 je
        (Klasse, Merkmal, Bereich); hier wird zur Laufzeit sortiert."""
        text = frage.lower().rstrip("?!. ")
        # "zweit größte" -> "zweitgrößte"; Ordnungszahl merken
        text = re.sub(r"\b(zweit|dritt|viert|fünft) (?=\w+ste)", r"\1", text)
        platz = 1
        m = re.search(r"\b(zweit|dritt|viert|fünft)(?=\w+ste)", text)
        if m:
            platz = self.ORDINAL[m.group(1)]
            text = text[:m.start()] + text[m.end():]
        woerter = re.findall(r"[\wäöüß\-]+", text)
        # Die Klasse ist der Kopf der Frage (in den ersten sechs Woertern):
        # "Welche Wueste ... auf dem afrikanischen Kontinent" ist keine
        # Frage nach einem Kontinent.
        klasse = next((self.KLASSE_WORT[w] for w in woerter[:6] if w in self.KLASSE_WORT), None)
        if not klasse or not self.klasse.get(klasse):
            return None
        stufe = self.steigerung(text, klasse)
        if not stufe:
            return None
        name, groesser = stufe
        # Bereich: Welt, sonst ein Ort aus der Frage ("Europas", "der Schweiz", "im Mittelmeer")
        ort = None
        if not self.WELT.search(text):
            for i in range(len(woerter)):
                for j in range(min(len(woerter), i + 3), i, -1):
                    k = " ".join(woerter[i:j])
                    for v in (k, k[:-1] if k.endswith("s") else None):
                        if v and v in self.orte and v not in self.KLASSE_WORT:
                            ort = v
            # (letzter Treffer gewinnt: "Welches Land in Afrika ist das größte?")
        # Jedes Wort muss erklaert sein, sonst ist es keine einfache Rangliste
        erklaert = self.FUELLWORT | set(self.KLASSE_WORT)
        if ort:
            erklaert |= set(ort.split()) | {w + "s" for w in ort.split()}
        for w in woerter:
            if w not in erklaert and not re.fullmatch(
                    r"(?:höchst|längst|tiefst|größt|kleinst|bevölkerungsreichst)e[nmrs]?", w):
                return None
        kandidaten = []
        for t in self.klasse[klasse]:
            # Nur Irdisches und Bestehendes: sonst ist Olympus Mons (Mars) der
            # hoechste Berg der Welt und das Russische Kaiserreich das groesste Land
            if klasse in ("berg", "fluss", "see", "insel", "stadt") and not self.lage.get(t):
                continue
            if self.tab.get(t, {}).get("aufloesung") or t in self.AUSSCHLUSS:
                continue
            if ort:
                if klasse == "fluss" and ort in {s for s in self.staaten.get(t, ())}:
                    # Ein Fluss "Frankreichs" liegt ganz in Frankreich: sonst
                    # gewinnt der Rhein, der Frankreich nur streift.
                    if len(self.staaten[t]) > 1:
                        continue
                elif ort not in self.lage.get(t, ()):
                    continue
            w = self.zahl(t, name)
            if w is not None and w > 0:
                kandidaten.append((w, t))
        if len(kandidaten) < platz + 1:
            return None                  # eine Rangliste aus einem Eintrag ist keine
        kandidaten.sort(reverse=groesser)
        w, t = kandidaten[platz - 1]
        return re.sub(r"\s*\([^)]*\)$", "", self.schreibweise.get(t, t))

    # Relativsatz = Mehrschrittfrage: "Wann starb der Autor, der ... schrieb?"
    # Der Gegenstand steckt hinter einer zweiten Frage; ein Tabellentreffer auf
    # irgendeinen Namen darin ist fast immer falsch (Mintaka-Pruefsatz 25.09.:
    # 5 Antworten der Faktenstufe, 0 richtig).
    MEHRSCHRITT = re.compile(r",\s*(?:der|die|das|dessen|deren|dem|den|welche[rsnm]?|wo|als)\b"
                             r"|\b(?:in dem|in der|auf dem|auf der) .+ (?:spielt|lebte|stattfand)\b")

    def antwort(self, frage):
        if self.MEHRSCHRITT.search(frage.lower()):
            return self.wer_war(frage) if frage.lower().startswith("wer war") else None
        for stufe in (self.vergleich, self.rangliste, self.einheiten, self.nennt_man,
                      self.fluss_rueck, self.wer_war, self.werke.antwort, self.tabelle):
            a = stufe(frage)
            if a:
                return a
        return None


def baue_leitsaetze(index_ordner, ziel="fakten/leitsatz.tsv",
                    bekanntheit="fakten/bekanntheit.csv"):
    """Titel -> erster Satz des ersten Absatzes, fuer "Wer war X?".

    Begriffsklaerungen ("X steht fuer:") bleiben draussen - deren erster Satz
    beantwortet nichts. Der Titel ist das laengste bekannte Praefix vor ": ",
    sonst wird aus "Napoleon: Total War: ..." der Leitsatz von "Napoleon".
    """
    from index_bauen import lies_bekanntheit, titel_der_zeile
    bekannt = set(lies_bekanntheit(bekanntheit))
    gesehen = set()
    n = 0
    with open(os.path.join(index_ordner, "chunks.txt"), encoding="utf-8", errors="ignore") as f, \
            open(ziel, "w", encoding="utf-8") as out:
        for z in f:
            titel = titel_der_zeile(z, bekannt)
            if not titel:
                continue
            t = titel.lower()
            if t in gesehen:
                continue
            gesehen.add(t)
            if "steht für" in z[:200] or "bezeichnet:" in z[:200]:
                continue
            satz = erster_satz(z[len(titel) + 2:].strip(), mit_titel=False).replace("\t", " ")
            out.write(f"{t}\t{satz}\n")
            n += 1
    print(f"  {n:,} Leitsaetze -> {ziel}")


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--leitsaetze"]:
        baue_leitsaetze(sys.argv[2])
    else:
        print(Fakten().antwort(" ".join(sys.argv[1:])))
