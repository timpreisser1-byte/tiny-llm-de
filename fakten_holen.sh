#!/bin/bash
# fakten_holen.sh — Zahlen und Kernfakten aus Wikidata als Tabellen fuer die SD.
# Grund: die Wikipedia-Kopie (20231101.de) wirft Vorlagen weg, darin stehen
# Hoehen, Tiefen, Laengen. "Die Zugspitze ist mit  der hoechste Gipfel".
cd "$(dirname "$0")"; mkdir -p fakten
P='PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX schema: <http://schema.org/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> '
hole() {  # name property item(0/1)
  if [ "$3" = 1 ]; then
    Q="$P SELECT ?t ?v WHERE { ?i wdt:$2 ?x . ?x rdfs:label ?v . FILTER(LANG(?v)=\"de\") ?a schema:about ?i ; schema:isPartOf <https://de.wikipedia.org/> ; schema:name ?t . }"
  else
    Q="$P SELECT ?t ?v WHERE { ?i wdt:$2 ?v . ?a schema:about ?i ; schema:isPartOf <https://de.wikipedia.org/> ; schema:name ?t . }"
  fi
  curl -sL -m 600 -G https://qlever.dev/api/wikidata --data-urlencode "query=$Q" -H "Accept: text/csv" -o "fakten/$1.csv"
  echo "  $1: $(($(wc -l < fakten/$1.csv)-1)) Zeilen"
}
hole hoehe P2044 0; hole tiefe P4511 0; hole laenge P2043 0; hole flaeche P2046 0
hole einwohner P1082 0; hole gruendung P571 0; hole hauptstadt P36 1
hole waehrung P38 1; hole sprache P37 1; hole komponist P86 1; hole erfinder P61 1
hole symbol P246 0
# Zweite Runde (24.09.): Arten, an denen echte Fragen noch scheiterten
hole geburt P569 0; hole autor P50 1; hole fluss P206 1; hole gruender P112 1
hole durchmesser P2386 0
# Kurznamen (UNO, DDR, BRD) -> Artikeltitel
Q="$P PREFIX skos: <http://www.w3.org/2004/02/skos/core#> SELECT ?v ?t WHERE { ?i skos:altLabel ?v . FILTER(LANG(?v)=\"de\") ?a schema:about ?i ; schema:isPartOf <https://de.wikipedia.org/> ; schema:name ?t . }"
curl -sL -m 900 -G https://qlever.dev/api/wikidata --data-urlencode "query=$Q" -H "Accept: text/csv" -o fakten/alias.csv
echo "  alias: $(($(wc -l < fakten/alias.csv)-1)) Zeilen"
# Bekanntheit: Zahl der Sprachversionen, entscheidet bei mehrdeutigen Namen
Q="$P PREFIX wikibase: <http://wikiba.se/ontology#> SELECT ?t ?v WHERE { ?a schema:about ?i ; schema:isPartOf <https://de.wikipedia.org/> ; schema:name ?t . ?i wikibase:sitelinks ?v . }"
curl -sL -m 900 -G https://qlever.dev/api/wikidata --data-urlencode "query=$Q" -H "Accept: text/csv" -o fakten/bekanntheit.csv
echo "  bekanntheit: $(($(wc -l < fakten/bekanntheit.csv)-1)) Zeilen"
hole bauhoehe P2048 0
# Ereignisse und Lebensdaten (25.09.): "Wann starb Goethe?", "Wann endete der
# Erste Weltkrieg?", "Wann sank die Titanic?" (-> "Untergang der Titanic", P585)
hole tod P570 0; hole beginn P580 0; hole ende P582 0; hole zeitpunkt P585 0
hole aufloesung P576 0

# ---------------------------------------------------------------------------
# Zahlen mit Einheit, Rang und Stichtag (fakten_rang/, gelesen von fakten_wahl.py):
# welcher von mehreren Werten gilt - bevorzugter Rang, juengster Stichtag.
mkdir -p fakten_rang fakten_rangliste
RP='PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX p: <http://www.wikidata.org/prop/> PREFIX ps: <http://www.wikidata.org/prop/statement/> PREFIX psv: <http://www.wikidata.org/prop/statement/value/> PREFIX pq: <http://www.wikidata.org/prop/qualifier/> PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX schema: <http://schema.org/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> '
DE='?a schema:about ?i ; schema:isPartOf <https://de.wikipedia.org/> ; schema:name ?t .'
frage() {   # datei abfrage - curl -f und Zeilenpruefung: abgeschnittene Antworten erkennen
  for versuch in 1 2 3; do
    curl -sfL -m 1800 -G https://qlever.dev/api/wikidata --data-urlencode "query=$RP $2" \
      -H "Accept: text/csv" -o "$1" && [ "$(wc -l < "$1")" -gt 1 ] && return
    sleep 20
  done
  echo "  FEHLER: $1" >&2
}
mit_rang() {   # name property
  frage "fakten_rang/$1.csv" "SELECT ?t ?v ?einheit ?rang ?zeit WHERE { ?i p:$2 ?st . ?st ps:$2 ?v ; wikibase:rank ?rang ; psv:$2 ?vn . ?vn wikibase:quantityUnit ?einheit . OPTIONAL { ?st pq:P585 ?zeit } $DE }"
  echo "  rang/$1: $(($(wc -l < fakten_rang/$1.csv)-1))"
}
mit_rang hoehe P2044; mit_rang bauhoehe P2048; mit_rang tiefe P4511; mit_rang laenge P2043
mit_rang flaeche P2046; mit_rang durchmesser P2386; mit_rang einwohner P1082

# Ranglisten (fakten_rangliste/): Klasse, Kontinent, Staat, Gebirge je Artikel
frage fakten_rangliste/klasse.csv "SELECT DISTINCT ?t ?k WHERE { VALUES (?typ ?k) { (wd:Q8502 \"berg\") (wd:Q207326 \"berg\") (wd:Q4022 \"fluss\") (wd:Q23397 \"see\") (wd:Q23442 \"insel\") (wd:Q515 \"stadt\") } ?i wdt:P31/wdt:P279* ?typ . $DE }"
frage fakten_rangliste/klasse2.csv "SELECT DISTINCT ?t ?k WHERE { VALUES (?typ ?k) { (wd:Q3624078 \"land\") (wd:Q9430 \"ozean\") (wd:Q5107 \"kontinent\") } ?i wdt:P31 ?typ . $DE }"
frage fakten_rangliste/planet.csv "SELECT DISTINCT ?t WHERE { ?i wdt:P31/wdt:P279* wd:Q634 . ?i wdt:P397 wd:Q525 . $DE }"
frage fakten_rangliste/kontinent_von.csv "SELECT DISTINCT ?t ?v WHERE { ?i wdt:P30 ?x . ?x rdfs:label ?v . FILTER(LANG(?v)=\"de\") $DE }"
frage fakten_rangliste/staat_von.csv "SELECT DISTINCT ?t ?v WHERE { ?i wdt:P17 ?x . ?x rdfs:label ?v . FILTER(LANG(?v)=\"de\") $DE }"
frage fakten_rangliste/gebirge_von.csv "SELECT DISTINCT ?t ?v WHERE { ?i wdt:P4552 ?x . ?x rdfs:label ?v . FILTER(LANG(?v)=\"de\") $DE }"
echo "  rangliste: $(ls fakten_rangliste | wc -l | tr -d ' ') Dateien"
