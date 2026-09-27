"""Deutscher Text zu Lautschrift - der erste Baustein jeder Sprachausgabe.

Ganz gleich, welchen Weg die Ausgabe am Ende nimmt - neuronales Modell,
zusammengesetzte Lautbausteine von der SD-Karte oder Regelsynthese -, alle
brauchen dasselbe zuerst: aus Buchstaben Laute machen. "Sprich" hat fuenf
Buchstaben und vier Laute, "Chemie" faengt nicht mit einem K an, und das V in
"Vogel" klingt anders als das in "Vase".

Deutsch ist dafuer ein dankbares Feld. Anders als im Englischen laesst sich
die Aussprache zu ueber neunzig Prozent aus der Schreibung ableiten - der
Rest sind Fremdwoerter und eine Handvoll Ausnahmen. Deshalb steht hier ein
Regelwerk und kein neuronales Netz: es passt in wenige Kilobyte, laeuft auf
dem ESP32 in reinem C und macht keine Fehler, die man nicht nachvollziehen
kann.

Die Lautschrift ist SAMPA-nah und bewusst reines ASCII, damit die C-Fassung
mit Bytes statt mit Unicode arbeiten kann.

    python tts_laute.py "Guten Morgen, wie geht es dir?"
    python tts_laute.py --pruefen
"""

import argparse
import json
import os
import re
import sys

# ------------------------------------------------------------------ Inventar

VOKALE = set("aeiouäöüy")

# Woerter, die sich der Regel entziehen. Kurz gehalten - jede Zeile hier ist
# eine Regel, die nicht gefunden wurde.
AUSNAHMEN = {
    "ist": "Ist", "das": "das", "des": "dEs", "was": "vas", "es": "Es",
    "sie": "zi:", "wie": "vi:", "die": "di:", "nie": "ni:",
    "chemie": "Cemi:", "chor": "ko:6", "charakter": "karaktA",
    "china": "Ci:na", "christ": "krIst",
    "vase": "va:z@", "vogel": "fo:g@l", "vater": "fa:tA",
    "video": "vi:deo", "vitamin": "vitami:n", "klavier": "klavi:6",
    "sechs": "zEks", "wachs": "vaks", "fuchs": "fUks",
    "englisch": "ENlIS", "hunger": "hUNA",
    "theater": "tea:tA", "thema": "te:ma",
    "computer": "kOmpju:tA", "handy": "hEndi",
    "der": "de:6", "wer": "ve:6", "her": "he:6", "mehr": "me:6",
    "sehr": "ze:6", "meer": "me:6", "aber": "a:bA", "oder": "o:dA",
    "ihr": "i:6", "wir": "vi:6", "mir": "mi:6", "vier": "fi:6",
    "dreißig": "dRaIsIC", "zwanzig": "tsvantsIC", "vierzig": "fIRtsIC",
    "bin": "bIn", "in": "In", "im": "Im", "mit": "mIt", "bis": "bIs",
    "um": "Um", "zum": "tsUm", "vom": "fOm", "an": "an", "am": "am",
    "man": "man", "hat": "hat", "ab": "ap", "ob": "Op", "von": "fOn",
    "eins": "aIns", "zwei": "tsvaI", "drei": "draI", "vier": "fi:6",
    "fünf": "fYnf", "sechs ": "zEks", "sieben": "zi:b@n", "acht": "axt",
    "neun": "nOYn", "zehn": "tse:n", "null": "nUl",
}

# Mehrbuchstabige Zeichenfolgen, laengste zuerst. Die Reihenfolge ist die
# Regel: "tsch" muss vor "sch" stehen, "sch" vor "ch", "ch" vor "c".
FOLGEN = [
    ("tsch", "tS"), ("dsch", "dZ"), ("sch", "S"),
    ("chs", "ks"),
    ("ck", "k"), ("ph", "f"), ("th", "t"), ("dt", "t"),
    ("qu", "kv"), ("pf", "pf"), ("tz", "ts"),
    ("ie", "i:"), ("ei", "aI"), ("ai", "aI"), ("ey", "aI"), ("ay", "aI"),
    ("eu", "OY"), ("äu", "OY"), ("au", "aU"),
    ("aa", "a:"), ("ee", "e:"), ("oo", "o:"),
    ("ss", "s"), ("ß", "s"), ("ng", "N"),
]

EINZELN = {
    "b": "b", "c": "k", "d": "d", "f": "f", "g": "g", "h": "h", "j": "j",
    "k": "k", "l": "l", "m": "m", "n": "n", "p": "p", "r": "R", "t": "t",
    "w": "v", "x": "ks", "z": "ts", "v": "f",
    "a": "a", "e": "E", "i": "I", "o": "O", "u": "U",
    "ä": "E", "ö": "9", "ü": "Y", "y": "Y",
}

LANG = {"a": "a:", "e": "e:", "i": "i:", "o": "o:", "u": "u:",
        "ä": "E:", "ö": "2:", "ü": "y:", "y": "y:"}

# stimmhafte Laute, die am Silbenende hart werden (Auslautverhaertung)
HART = {"b": "p", "d": "t", "g": "k", "z": "s", "v": "f"}


def _ch_laut(wort, i):
    """ch ist der eine Fall, in dem der Vorlaut entscheidet.

    ich-Laut nach hellen Vokalen und Konsonanten, ach-Laut nach dunklen,
    und am Wortanfang vor a/o/u/l/r wie ein k ("Charakter", "Chlor").
    """
    if i == 0:
        return "k" if i + 2 < len(wort) and wort[i + 2] in "aoulr" else "C"
    vorher = wort[i - 1]
    if vorher in "aou" or wort[max(0, i - 2):i] in ("au",):
        return "x"
    return "C"


def _s_laut(wort, i):
    """s vor Vokal ist stimmhaft, am Wortanfang vor p/t wird es zum sch."""
    if i + 1 < len(wort) and wort[i + 1] in "pt" and i == 0:
        return "S"
    if i + 1 < len(wort) and wort[i + 1] in VOKALE:
        return "z"
    return "s"


# Buchstabenfolgen, die EINEN Konsonanten bezeichnen. Fuer die Vokallaenge
# ist das der entscheidende Unterschied: "Buch" hat nach dem u nur einen
# Laut, obwohl dort zwei Buchstaben stehen - deshalb ist das u lang.
EIN_LAUT = ("sch", "ch", "ph", "th", "ß")
# diese signalisieren umgekehrt ausdruecklich einen kurzen Vokal
KURZ_ZEICHEN = ("ck", "ss", "tz", "ng", "nk")


def _ist_lang(wort, i):
    """Vokallaenge nach der Schulregel, aber auf Lauten statt Buchstaben.

    lang   vor Dehnungs-h, vor genau einem Konsonantenlaut, am Wortende
    kurz   vor zwei Lauten, vor verdoppelten Konsonanten, vor ck/ss/tz
    """
    rest = wort[i + 1:]
    if rest.startswith("h"):
        return True                    # Dehnungs-h, auch zwischen Vokalen
    # ich-Laut nach hellem Vokal kuerzt: "ich", "sprich", "Blech" -
    # anders als der ach-Laut in "Buch", "hoch", "Sprache"
    if wort[i] in "ie" and rest.startswith("ch"):
        return False
    j = k = 0
    while j < len(rest) and rest[j] not in VOKALE:
        if any(rest.startswith(z, j) for z in KURZ_ZEICHEN):
            return False
        for folge in EIN_LAUT:
            if rest.startswith(folge, j):
                j += len(folge); k += 1; break
        else:
            if j + 1 < len(rest) and rest[j] == rest[j + 1]:
                return False           # verdoppelter Konsonant
            j += 1; k += 1
        if k > 1:
            return False
    return k <= 1


def wort_zu_lauten(wort: str, mit_spannen=False):
    """Wort -> Lautkette. Mit `mit_spannen` zusaetzlich, welcher
    Buchstabenbereich welchen Laut erzeugt hat.

    Das braucht die Sprachausgabe: die erzwungene Ausrichtung liefert
    Zeitmarken je BUCHSTABE, geschnitten wird aber nach LAUTEN. "sch" sind
    drei Buchstaben und ein Laut, "x" ein Buchstabe und zwei. Ohne diese
    Zuordnung liegen die Schnitte daneben.
    """
    wort = wort.lower()
    if wort in AUSNAHMEN:
        aus = AUSNAHMEN[wort]
        if mit_spannen:      # bei Ausnahmen gleichmaessig verteilen
            n = max(len(aus), 1)
            return aus, [(k * len(wort) // n, (k + 1) * len(wort) // n)
                         for k in range(n)]
        return aus
    laute, spannen, i = [], [], 0
    while i < len(wort):
        # Endungen zuerst - sie klingen anders als ihre Buchstaben
        if wort[i:] == "er" and i > 0:
            laute.append("A"); spannen.append((i, len(wort))); break
        if wort[i:] == "en" and i > 0:
            laute.append("@n"); spannen.append((i, len(wort))); break
        if wort[i:] == "el" and i > 0:
            laute.append("@l"); spannen.append((i, len(wort))); break
        if wort[i:] == "e" and i > 0:
            laute.append("@"); spannen.append((i, len(wort))); break

        if wort[i:i + 2] == "ch":
            laute.append(_ch_laut(wort, i)); spannen.append((i, i + 2)); i += 2; continue

        for folge, laut in FOLGEN:
            if wort.startswith(folge, i):
                laute.append(laut); spannen.append((i, i + len(folge)))
                i += len(folge); break
        else:
            z = wort[i]
            if z == "s":
                laute.append(_s_laut(wort, i)); spannen.append((i, i + 1)); i += 1; continue
            if z == "h" and i > 0 and wort[i - 1] in VOKALE:
                i += 1; continue                    # Dehnungs-h ist stumm
            if z == "r" and i > 0 and wort[i - 1] in VOKALE | {"h"} \
                    and i + 1 == len(wort):
                laute.append("6"); spannen.append((i, i + 1)); i += 1; continue      # vokalisiertes r
            if z in VOKALE:
                laute.append(LANG[z] if _ist_lang(wort, i) else EINZELN[z])
                spannen.append((i, i + 1))
                i += 1; continue
            # verdoppelter Konsonant ist ein Laut - die Doppelung sagt nur,
            # dass der Vokal davor kurz ist, und das ist oben schon erledigt
            doppelt = i + 1 < len(wort) and wort[i + 1] == z
            if z in HART and (i + 1 == len(wort) or wort[i + 1] not in VOKALE) \
                    and not doppelt:
                laute.append(HART[z]); spannen.append((i, i + 1)); i += 1; continue
            laute.append(EINZELN.get(z, ""))
            spannen.append((i, i + (2 if doppelt else 1)))
            i += 2 if doppelt else 1
    return ("".join(laute), spannen) if mit_spannen else "".join(laute)


def woerter(text: str):
    """Text -> Liste von Woertern, Satzzeichen weg.

    DIE gemeinsame Zerlegung fuer Schneiden und Sprechen. Vorher hatte jede
    Seite ihre eigene: das Schneiden trennte an Leerzeichen und scheiterte
    deshalb an "sagen," - das Wort fiel ganz weg -, das Sprechen trennte mit
    findall und warf Satzzeichen als eigene Marken aus. Ergebnis: die
    Wortgrenzen standen an verschiedenen Stellen, und nur 70 Prozent der
    gesuchten Uebergaenge existierten ueberhaupt. Selbst aus EINER Aufnahme
    liess sich der eigene Satz nicht mehr zusammensetzen.
    """
    return re.findall(r"[A-Za-zÄÖÜäöüß]+", text)


def zu_lauten(text: str) -> str:
    """Text -> Lautkette, Woerter durch Leerzeichen getrennt."""
    return " ".join(wort_zu_lauten(w) for w in woerter(text))


def lautfolge(text: str):
    """Text -> flache Lautliste mit Wortgrenzen als '#'.

    Genau diese Folge wird beim Schneiden und beim Sprechen gebraucht, und
    beide holen sie ab jetzt hier.
    """
    from tts_bausteine import zerlege
    raus = ["#"]
    for w in woerter(text):
        raus.extend(zerlege(wort_zu_lauten(w)))
        raus.append("#")
    return raus


# ------------------------------------------------------------------ Pruefung

PROBEN = [
    ("Hallo",     "halo:"),
    ("sprich",    "SpRIC"),
    ("Sprache",   "SpRa:x@"),
    ("Buch",      "bu:x"),
    ("ich",       "IC"),
    ("Milch",     "mIlC"),
    ("Nacht",     "naxt"),
    ("Tag",       "ta:k"),
    ("Hand",      "hant"),
    ("Weg",       "ve:k"),
    ("Kind",      "kInt"),
    ("Haus",      "haUs"),
    ("Leute",     "lOYt@"),
    ("Bäume",     "bOYm@"),
    ("heiß",      "haIs"),
    ("Straße",    "StRa:s@"),
    ("Wasser",    "vasA"),
    ("Sonne",     "zOn@"),
    ("gehen",     "ge:@n"),
    ("Mutter",    "mUtA"),
    ("Vogel",     "fo:g@l"),
    ("zwei",      "tsvaI"),
    ("Pflanze",   "pflants@"),
    ("Quelle",    "kvEl@"),
    ("Zunge",     "tsUN@"),
    ("Apfel",     "apf@l"),
    ("dir",       "di:6"),
    ("Uhr",       "u:6"),
]


def pruefen():
    treffer = 0
    print(f"\n  {'Wort':14}{'erwartet':16}{'bekommen':16}")
    print("  " + "-" * 48)
    for wort, soll in PROBEN:
        ist = wort_zu_lauten(wort)
        ok = ist == soll
        treffer += ok
        zeichen = "\033[92m✓\033[0m" if ok else "\033[91m✗\033[0m"
        print(f"  {wort:14}{soll:16}{ist:16}{zeichen}")
    print(f"\n  {treffer}/{len(PROBEN)} richtig "
          f"({treffer/len(PROBEN)*100:.0f} %)\n")
    return treffer / len(PROBEN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--pruefen", action="store_true")
    args = ap.parse_args()
    if args.pruefen or not args.text:
        pruefen()
    else:
        text = " ".join(args.text)
        print(f"\n  {text}\n  {zu_lauten(text)}\n")


if __name__ == "__main__":
    main()
