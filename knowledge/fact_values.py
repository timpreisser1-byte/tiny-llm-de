"""knowledge/fact_values.py — aus mehreren Wikidata-Werten den richtigen waehlen.

Bisher nahm knowledge/facts.py bei mehreren Werten einfach den groessten. Wikidata sagt
aber selbst, welcher gilt:
  - Rang: PreferredRank schlaegt NormalRank; DeprecatedRank ist veraltet/falsch
  - Stichtag (P585): bei Einwohnern der juengste Stand
  - Einheit: 21195 bei der Chinesischen Mauer sind km, beim Burj Khalifa m,
    Flaechen stehen teils in Hektar oder Quadratmeilen

`scripts/fetch_wikidata.sh` (Abschnitt "mit Rang") laedt dazu fakten_rang/*.csv mit den
Spalten t, v, einheit, rang, zeit.
"""
import csv
import os
from collections import defaultdict

# Wikidata-Einheit -> (Faktor zur Zieleinheit, Zieleinheit)
EINHEIT = {
    "Q11573": (1, "Meter"), "Q828224": (1, "Kilometer"),
    "Q174728": (0.01, "Meter"), "Q174789": (0.001, "Meter"),
    "Q3710": (0.3048, "Meter"), "Q218593": (0.0254, "Meter"),
    "Q253276": (1.609344, "Kilometer"), "Q93318": (1.852, "Kilometer"),
    "Q712226": (1, "Quadratkilometer"), "Q35852": (0.01, "Quadratkilometer"),
    "Q25343": (1e-6, "Quadratkilometer"), "Q232291": (2.589988, "Quadratkilometer"),
    "Q81292": (0.00404686, "Quadratkilometer"),
    "Q199": (1, ""),
}
RANG = {"PreferredRank": 2, "NormalRank": 1, "DeprecatedRank": 0}


def lade(ordner="fakten_rang"):
    """-> {titel_klein: {merkmal: [(wert, einheit, rang, zeit), ...]}}"""
    tab = defaultdict(lambda: defaultdict(list))
    if not os.path.isdir(ordner):
        return tab
    for datei in os.listdir(ordner):
        if not datei.endswith(".csv"):
            continue
        name = datei[:-4]
        with open(os.path.join(ordner, datei), encoding="utf-8") as f:
            for z in csv.reader(f):
                if len(z) < 4 or z[0] == "t":
                    continue
                q = z[2].rsplit("/", 1)[-1]
                if q not in EINHEIT:
                    continue
                faktor, einheit = EINHEIT[q]
                try:
                    wert = float(z[1]) * faktor
                except ValueError:
                    continue
                rang = RANG.get(z[3].rsplit("#", 1)[-1], 1)
                zeit = z[4] if len(z) > 4 else ""
                tab[z[0].lower()][name].append((wert, einheit, rang, zeit))
    return tab


def waehle(eintraege):
    """Bester Wert: hoechster Rang, dann juengster Stichtag, dann groesster.

    Groesster als letzter Ausweg, weil mehrere Normal-Werte ohne Stichtag
    meist Messvarianten sind (Gipfel vs. Vorgipfel) - der Hauptwert ist
    dann in der Regel der groessere.
    """
    gueltig = [e for e in eintraege if e[2] > 0] or eintraege
    top = max(e[2] for e in gueltig)
    gueltig = [e for e in gueltig if e[2] == top]
    mit_zeit = [e for e in gueltig if e[3]]
    if mit_zeit:
        neu = max(e[3] for e in mit_zeit)
        gueltig = [e for e in mit_zeit if e[3] == neu]
    return max(gueltig, key=lambda e: e[0])


def zahl_text(wert, einheit):
    if wert >= 1e9:
        s = f"{wert/1e9:.2f} Milliarden".replace(".", ",")
    elif wert >= 1e6:
        s = f"{wert/1e6:.2f} Millionen".replace(".", ",")
    elif wert != int(wert):
        s = f"{wert:.1f}".replace(".", ",")
    else:
        s = f"{wert:.0f}"
    return f"{s} {einheit}".strip()
