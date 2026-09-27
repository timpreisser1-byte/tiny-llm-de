# Tiny-LLM DE — eine Offline-Wissens-KI auf Deutsch für den ESP32-S3

Ziel: ein Gerät mit **ESP32-S3** (8 MiB Flash, 8 MiB PSRAM) und einer
**32-GB-SD-Karte**, das deutsche Wissensfragen ohne Internet beantwortet und
aus den Gesprächen mit seinem Besitzer mitlernt.

Die Grundidee: In ein paar Megabyte Gewichte passt kein Weltwissen. Das Wissen
liegt deshalb auf der SD-Karte — ein Wikipedia-Suchindex und Wikidata-Tabellen —
und ein kleines ternäres Sprachmodell muss nur noch **lesen**, nicht wissen.

```
Frage
  → Gedächtnis      (was der Nutzer erzählt oder korrigiert hat, SD)
  → Smalltalk
  → Wikidata-Tabellen: Fakten, Werke, Ereignisse, Vergleiche, Ranglisten
  → Suche im Wikipedia-Index auf der SD (3 Absätze)
  → präzise Regel  → Modell liest  → Satz-Rückfall
Antwort
```

## Stand (September 2026)

Gemessen mit strenger Bewertung (nur ganze Wörter). MKQA und Mintaka sind
**unabhängige** Prüfsätze, an denen nicht entwickelt wurde; entwickelt wird an
eigenen `*_dev`-Sätzen.

| Prüfsatz | Fragen | richtig beantwortet | erfunden |
|---|---|---|---|
| MKQA-de (echte Suchanfragen) | 500 | 8,6 % | 0,2 % |
| Mintaka-de (Vergleiche, Superlative, Mehrschritt) | 500 | 11,8–12,8 % | 2,4 % |
| eigene natürliche Fragen | 195 | 58,5 % | 0 % |
| handgeschriebene Fragen | 60 | 66,7 % | 0 % |

Die großen Unterschiede zwischen den Sätzen sind ehrlich: Die eigenen Fragen
sind deutsch-alltäglich, MKQA ist US-lastig (Popkultur, Sport), und die Suche
findet bei MKQA die Antwort nur in etwa 20 % der Fälle unter den ersten drei
Absätzen. Die Suche ist der größte offene Engpass.

Was die größten Schritte gebracht hat:

- **Wikidata-Tabellen statt größerem Index** — die Wikipedia-Kopie hat keine
  Infoboxen, Höhen und Jahreszahlen fehlen im Text (`fakten.py`)
- **Index nach Bekanntheit der Artikel** statt Dateireihenfolge (`index_bauen.py`)
- **Schweigeregeln**: Tabellen antworten nur, wenn jedes Wort der Frage erklärt
  ist — Genauigkeit vor Reichweite
- **Lesetraining mit echten Fragen** (`lesen_daten.py`): erfundene Antworten
  von 19 % auf 2 %

## Aufbau

**Wissens-Kette**

| Datei | Zweck |
|---|---|
| `wissen_index.py` | Suche: BM25-artig, Titelbonus, Paarnähe; baut auch den Rohindex |
| `index_bauen.py` | gestufter Index nach Bekanntheit (Wikidata-Sitelinks) |
| `fakten.py` | Wikidata-Stufen: Tabellen, Ereignisse, Vergleiche, Ranglisten |
| `fakten_werk.py` | Lieder, Filme, Bücher (englische und deutsche Titel) |
| `fakten_wahl.py` | richtigen Wert wählen: Rang, Stichtag, Einheit |
| `regeln_breit.py` | präzise Regel-Extraktion aus Absätzen |
| `geraet.py` | Modell-Inferenz („Frage rein, Antwort raus“) |
| `esp32_sd/` | Geräte-Referenz: SD-Indexformat, Chat, **Gedächtnis** (`memory.py`), Tests |

**Messen**

| Datei | Zweck |
|---|---|
| `chat_test.py` | die ganze Kette auf einem Fragensatz (`--kette`, `--fakten`, `--regeln2`) |
| `pruefsatz_extern.py` | baut MKQA-/Mintaka-Prüf- und Entwicklungssätze |
| `stufe_genau.py`, `rangliste_test.py` | Genauigkeit einzelner Stufen |
| `chat_fragen_*.json` | Fragensätze (siehe Lizenzen unten) |

**Modell und Training**

`model.py` (Decoder, ternär über `ternaer.py`), `train.py`, `sft.py`,
`quantize.py`, `speicher.py` (passt es in Flash und PSRAM?), Tokenizer
(`train_tokenizer.py`, `tokenizer/`), Datenbauer (`ablenkung_daten.py`,
`kurzkontext.py`, `chat_daten2.py`, `lesen_daten.py` …) und das Rezept des
aktuellen Modells in `kette_gross.sh`.

**Sprache** (Versuchsstand): Spracherkennung `asr_*.py`, Sprachausgabe
`tts_*.py`, das gesprochene Gerät `geraet_sprache.py`.

## Loslegen

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest esp32_sd.test_chat esp32_sd.test_memory esp32_sd.test_sd_index \
    esp32_sd.test_smalltalk esp32_sd.test_extractive_answer esp32_sd.test_model_manifest
```

Daten, Indizes und Modelle liegen **nicht** im Repository (zusammen über
60 GB). Nachbauen:

```bash
# 1. Wikipedia (wikimedia/wikipedia, 20231101.de) als Parquet nach wiki_roh/, dann
python wissen_index.py bauen --quelle 'wiki_roh/*.parquet' --ziel wiki_voll --max-artikel 0 --max-pro-artikel 12
./fakten_holen.sh                      # Wikidata-Tabellen (über qlever.dev)
./fakten_werk_holen.sh                 # Werke
python index_bauen.py 140000:8,1000000:2 wiki_gestuft
python fakten.py --leitsaetze wiki_gestuft
# 2. Prüfsätze (MKQA und Mintaka nach pruefsaetze/ laden, siehe Datei)
python pruefsatz_extern.py
# 3. messen
python chat_test.py --fragen chat_fragen_mkqa_dev.json --regeln2 --fakten --absaetze 3 --zeichen 1500
```

## Datenquellen und Lizenzen

- **Wikipedia** (deutsch) — CC BY-SA 4.0, über `wikimedia/wikipedia` auf Hugging Face
- **Wikidata** — CC0, abgefragt über [QLever](https://qlever.dev)
- **MKQA** (Apple) — CC BY-SA 3.0; `chat_fragen_mkqa*.json` sind daraus abgeleitet
  und stehen unter derselben Lizenz. https://github.com/apple/ml-mkqa
- **Mintaka** (Amazon) — CC BY 4.0; `chat_fragen_mintaka*.json` sind daraus
  abgeleitet. https://github.com/amazon-science/mintaka
- **GermanQuAD** (deepset) — CC BY 4.0, nur für Trainingsdaten, nicht enthalten
- `chat_fragen_1000.json` enthält Auszüge aus Wikipedia-Absätzen (CC BY-SA 4.0)

Für den Code selbst ist noch keine Lizenz festgelegt.
