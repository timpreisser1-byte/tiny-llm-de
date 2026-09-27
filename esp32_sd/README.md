# ESP32-S3 SD-Chatbot Referenz

Dieser Ordner enthaelt den aktuellen Host-Prototypen fuer das Ziel:

Frage -> SD-Wikipedia-Suche -> passender Absatz -> extraktive Antwort.

Optional kann danach ein kleines Sprachmodell eine Antwort aus demselben Absatz
formulieren. Der robuste 8-MB-Zielpfad ist aber zuerst die belegte, extraktive
Antwort.

Das ist noch keine ESP32-S3-Firmware. Es ist die bewusst kleine Referenz, die
zeigt, welches Dateiformat, welche Suchlogik und welches Modellfenster auf den
Chip portiert werden sollen.

## Stand

- Kette im Chat (`chat.py`): Gedaechtnis -> Smalltalk -> Wikidata-Tabellen
  (`../fakten.py`) -> SD-Suche -> Regel-Extraktion bzw. optional Modell.
- `memory.py` ist das Gedaechtnis zum Mitlernen: "Merk dir: ...",
  "Falsch, richtig ist ...", "Stimmt.", "Vergiss ...". Es schreibt nur an
  eine JSONL-Datei auf der SD an (`gedaechtnis.jsonl`, nicht im Repository -
  sie enthaelt, was du dem Geraet ueber dich erzaehlst).
- `wiki_sd_gestuft/` ist der aktuelle Index (Artikel nach Bekanntheit gestuft,
  1,2 GB, erzeugt mit `../index_bauen.py` und dem Exporter unten); `wiki_sd/`
  der aeltere. Beide liegen nicht im Repository.
- `sd_index.py` liest Vokabular, Postinglisten und komprimierte Abschnitte ohne
  den ganzen Index in RAM zu laden.
- `chat.py` ist ein Desktop-Chat. Ohne Checkpoint gibt er eine kurze Antwort,
  die direkt aus dem besten Wikipedia-Treffer stammt. Mit Checkpoint erzeugt er
  optional eine freie Antwort aus demselben Auszug.
- `extractive_answer.py` enthaelt kleine, ESP32-taugliche Regeln fuer Fragen
  wie "was ist", "wo liegt", "wie hoch", "wie lang", "wie tief" und
  "wann wurde".
- `smalltalk.py` beantwortet kurze Dialogsaetze wie Begruessung, Identitaet,
  Hilfe, Dank und aktuelles Thema ohne SD-Suche.
- `eval_chat.py` testet den Chat als Ganzes mit einer groesseren Fragebatterie.
- `model_backend.py` laedt vorhandene PyTorch-Checkpoints auf CPU und kuerzt nur
  den Kontext so, dass die Frage im Modellfenster erhalten bleibt.
- `model_manifest.py` misst Checkpoints read-only und schreibt Sidecar-Berichte
  fuer Speicherabschaetzungen.

## Starten

Aus dem Projektwurzelordner:

```bash
./.venv/bin/python -m esp32_sd.chat "Wie hoch ist der Eiffelturm?"
```

Mit JSON-Ausgabe:

```bash
./.venv/bin/python -m esp32_sd.chat --json "Wie hoch ist der Eiffelturm?"
```

Mit generierter Modellantwort:

```bash
./.venv/bin/python -m esp32_sd.chat \
  --checkpoint ckpt_8mb/sft_neu.pt \
  --max-new-tokens 60 \
  "Wie hoch ist der Eiffelturm?"
```

Interaktiv:

```bash
./.venv/bin/python -m esp32_sd.chat
```

`/neu` setzt das Gespraechsthema zurueck, `/ende` beendet den Chat.

## Gemessene Ergebnisse auf dem Mac

Letzter Lauf:

```bash
./.venv/bin/python -m unittest \
  esp32_sd.test_chat esp32_sd.test_sd_index esp32_sd.test_model_manifest
```

Ergebnis: 40 Tests bestanden.

Breiter Chat-Test:

```bash
./.venv/bin/python -m esp32_sd.eval_chat
```

Ergebnis: 14 von 14 Chat-Faellen bestanden.

100-Fragen-Test:

```bash
./.venv/bin/python -m esp32_sd.eval_chat \
  --total 100 \
  --json > esp32_sd/reports/chat_eval_100.json
```

Die Berichte unter `esp32_sd/reports/` werden beim Aufruf erzeugt und liegen
nicht im Repository.

Ergebnis (September, alter Index): 84 von 100 streng bestandenen Faellen. Geprueft werden Smalltalk,
richtiger Artikel und erwarteter Antwortwert. Die Fehlerliste steht in
`esp32_sd/reports/chat_eval_100.json`; ein Teil sind echte Extraktionsluecken,
ein Teil sind schiefe automatisch erzeugte Erwartungswerte.

Retrieval-Messung:

```bash
./.venv/bin/python -m esp32_sd.benchmark \
  --questions chat_fragen.json \
  --output esp32_sd/reports/retrieval.json
```

Ergebnis:

- 200 Fragen
- 196 erwartete Titel in den besten 2 Treffern
- Median: 0.0406 Sekunden pro Suche auf dem Host
- Python-Tracemalloc-Spitze: 574176 Byte

Diese Zahl misst nur Titel-Retrieval. Sie beweist nicht, dass jeder Absatz die
Frage korrekt beantwortet.

Kurzer Modelltest:

```bash
./.venv/bin/python -m esp32_sd.chat \
  --json \
  --checkpoint ckpt_8mb/sft_neu.pt \
  --max-new-tokens 16 \
  "Wie hoch ist der Eiffelturm?"
```

Antwortbeginn:

```text
Der Eiffelturm ist ein 330 Meter hoher Eisenfachwer
```

Das zeigt, dass Suche und Modell zusammenlaufen. Mit nur 16 neuen Tokens ist
die Antwort absichtlich kurz und abgeschnitten.

Kurze extraktive Antworten ohne Checkpoint:

```text
Wie hoch ist der Eiffelturm? -> Eiffelturm: 330 Meter hoch.
Wann wurde die Berliner Mauer gebaut? -> Berliner Mauer: vom 13. August 1961.
Was ist ein Transistor? -> Ein Transistor ist ein elektronisches Halbleiter-Bauelement ...
```

## Speicherbild

Der Standardpfad ohne Sprachmodell muss nur kleine Suchzustaende, bis zu 30
Kandidaten, einen dekomprimierten Absatz und kurze Antwortpuffer halten. Der
große Index bleibt auf der SD-Karte. Der letzte Host-Lauf zeigte fuer diese
Suche 574176 Byte Python-Tracemalloc-Spitze; das ist kein ESP32-Beweis, aber
ein gutes Zeichen fuer einen festen C/C++-Speicherplan deutlich unter 8 MiB.

Der aktuelle `sft_neu`-Checkpoint hat laut Manifest:

- 13227370 eindeutige Parameter
- Sherry/TLM1-Messexport: 3068013 Byte
- FP32-KV-Cache: 1228800 Byte
- Gewichte plus FP32-KV: 4296813 Byte
- Rest vor Tokenizer, Suchpuffern, Aktivierungen, Stacks und Audio: 4091795 Byte

Das passt rechnerisch in 8 MiB PSRAM, ist aber noch kein Firmware-Beweis. Fuer
den ESP32-S3 muessen Tokenizer, SD-Suche, Dekompression, Aktivierungen und
Audio-Puffer zusammen gemessen werden.

## ESP32-S3 Portierungsplan

1. Binaerformat `wiki_sd/` auf SD-Karte kopieren.
2. `sd_index.py` in C/C++ portieren:
   - binaere Suche in `vocab.bin`
   - Posting-Merge ueber kleine Stream-Zustaende
   - zlib/miniz Dekompression einzelner Abschnitte
   - maximal 32 Suchterme
3. Modellruntime portieren:
   - vollstaendige Config aus dem Manifest lesen
   - Sherry/TLM1-Gewichte laden
   - KV-Cache fest in PSRAM reservieren
   - Tokenizer-Speicher separat messen
4. Erst Text-Chat ueber Serial oder kleines Display stabilisieren.
5. Danach Sprache:
   - Wakeword/Kommandos sind auf ESP32-S3 mit ESP-SR realistisch.
   - Freie deutsche Spracherkennung ist mit den Espressif-Fertigmodellen nicht
     abgedeckt.
   - ESP-SR TTS unterstuetzt laut aktueller Doku nur Chinesisch.

Quellen fuer die Hardwareannahmen:

- https://docs.espressif.com/projects/esp-idf/en/v5.4.1/esp32s3/api-guides/external-ram.html
- https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/speech_command_recognition/README.html
- https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/getting_started/readme.html

## Offene Punkte

- Noch keine ESP-IDF-Firmware.
- Noch kein echter Lauf auf ESP32-S3 mit 8 MiB PSRAM.
- Noch kein C/C++-Loader fuer das Modellformat.
- Noch keine robuste deutsche STT/TTS-Loesung auf dem Zielchip.
