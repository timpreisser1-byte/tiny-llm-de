#!/bin/bash
# scripts/fetch_works.sh — Werke (Lieder, Filme, Buecher, Spiele) mit englischem
# UND deutschem Titel. MKQA fragt "wer singt i can't fight this feeling":
# englische Titel, klein geschrieben, oft ohne deutschen Wikipedia-Artikel.
#
# Spalten: t (Titel en/de), v (Wert: Name oder Datum), n (Sprachversionen des
# Werks), m (Sprachversionen des Werts). Nur Werke mit mindestens einem
# Wikipedia-Artikel irgendwo (n >= 1): das wirft Coverversionen und die
# Millionen wissenschaftlicher Artikel mit Erscheinungsdatum hinaus.
# Gefiltert wird beim Laden (knowledge/works.py), NICHT in der Abfrage: QLever
# wertet FILTER(?n >= 1) je nach Ausfuehrungsplan falsch aus und verwirft
# still Zeilen (Pulp Fiction fehlte, COUNT ergab 0; 25.09.).
# scripts/ liegt eine Ebene unter dem Projekt
cd "$(dirname "$0")/.."; mkdir -p fakten_werk
P='PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX wikibase: <http://wikiba.se/ontology#> '
# QLever liefert unter Last manchmal nur die Kopfzeile (HTTP 200, leer):
# dann nach einer Pause bis zu dreimal neu fragen.
frage() {   # datei abfrage
  # Auch ein Abbruch durch das Zeitlimit hinterlaesst eine halbe, gueltig
  # aussehende Datei (autor.csv endete bei "R..." - Romeo und Julia fehlte).
  # Deshalb zaehlt nur ein curl ohne Fehler.
  for versuch in 1 2 3; do
    if curl -sfL -m 3600 -G https://qlever.dev/api/wikidata --data-urlencode "query=$2" \
         -H "Accept: text/csv" -o "$1" && [ "$(wc -l < "$1")" -gt 1 ]; then
      return
    fi
    echo "    Versuch $versuch fehlgeschlagen" >&2; sleep 20
  done
}
hole_name() {   # name property
  Q="$P SELECT ?t ?v ?n ?m WHERE { ?i wdt:$2 ?x . MINUS { ?i wdt:P31 wd:Q13442814 } ?i wikibase:sitelinks ?n . ?i rdfs:label ?t . FILTER(LANG(?t)=\"en\" || LANG(?t)=\"de\") ?x rdfs:label ?v . FILTER(LANG(?v)=\"de\" || LANG(?v)=\"en\") ?x wikibase:sitelinks ?m . }"
  frage "fakten_werk/$1.csv" "$Q"
  nur_mit_artikel "fakten_werk/$1.csv"
  echo "  $1: $(($(wc -l < fakten_werk/$1.csv)-1))"
}
hole_lied() {   # wie hole_name, aber nur Lieder/Singles/Musikwerke: sonst
                # gewinnt bei "When a Man Loves a Woman" die Filmmusik (25.09.)
  Q="$P SELECT ?t ?v ?n ?m WHERE { VALUES ?typ { wd:Q7366 wd:Q134556 wd:Q105543609 wd:Q2188189 } ?i wdt:P31 ?typ . ?i wdt:$2 ?x . ?i wikibase:sitelinks ?n . ?i rdfs:label ?t . FILTER(LANG(?t)=\"en\" || LANG(?t)=\"de\") ?x rdfs:label ?v . FILTER(LANG(?v)=\"de\" || LANG(?v)=\"en\") ?x wikibase:sitelinks ?m . }"
  frage "fakten_werk/$1.csv" "$Q"
  nur_mit_artikel "fakten_werk/$1.csv"
  echo "  $1: $(($(wc -l < fakten_werk/$1.csv)-1))"
}
hole_datum() {
  Q="$P SELECT ?t ?v ?n WHERE { ?i wdt:$2 ?v . MINUS { ?i wdt:P31 wd:Q13442814 } ?i wikibase:sitelinks ?n . ?i rdfs:label ?t . FILTER(LANG(?t)=\"en\" || LANG(?t)=\"de\") }"
  frage "fakten_werk/$1.csv" "$Q"
  nur_mit_artikel "fakten_werk/$1.csv"
  echo "  $1: $(($(wc -l < fakten_werk/$1.csv)-1))"
}
# Werke ohne jeden Wikipedia-Artikel (n = 0) sofort verwerfen: sonst 2,3 GB,
# fast nur wissenschaftliche Aufsaetze mit Autor und Erscheinungsdatum.
nur_mit_artikel() {
  python3 - "$1" <<'PY2'
import csv, sys, os
p = sys.argv[1]; tmp = p + ".neu"
with open(p, encoding="utf-8", newline="") as f, open(tmp, "w", encoding="utf-8", newline="") as g:
    w = csv.writer(g)
    for z in csv.reader(f):
        if z and (z[0] == "t" or (len(z) > 2 and z[2] not in ("", "0"))):
            w.writerow(z)
os.replace(tmp, p)
PY2
}
hole_lied interpret P175; hole_name regie P57; hole_name autor P50
hole_name komponist P86; hole_name liedtexter P676
hole_datum erschienen P577
