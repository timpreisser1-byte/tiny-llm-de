# SD-Suchindex v1

`sd_index.py` ist ein Desktop-Exporter und ein Python-Referenzleser, keine ESP32-Firmware. Der Exporter lädt das alte JSON-Wörterbuch im Host-RAM; der Leser lädt weder Wörterbuch noch Postings vollständig. Die vorhandenen Wikipedia-Dateien bleiben unverändert.

Aufruf vom Projektordner:

```sh
python3 -m esp32_sd.sd_index export --source wiki --output esp32_sd/wiki_sd
python3 -m esp32_sd.sd_index search --index esp32_sd/wiki_sd 'Wo liegt München?'
python3 -m unittest esp32_sd.test_sd_index
```

Das Ziel muss neu sein. Ein fehlgeschlagener Export kann ein unvollständiges Ziel hinterlassen; erst `manifest.json` markiert den vollständigen Export. Quellpostings müssen aus dem vorhandenen little-endian uint32-Format stammen, sortiert sein und eindeutige Absatznummern enthalten.

Alle Zahlen sind little-endian. Dateien:

| Datei | Layout |
|---|---|
| `manifest.json` | Formatkennung, Absatz-/Wortanzahl, Größenlimit |
| `vocab.bin` | Sortiert nach UTF-8-Bytes, feste 104-Byte-Einträge: `80s Q I Q I`: nullgefülltes Wort, Textposting-Start/Anzahl, Titelposting-Start/Anzahl. Starts sind uint32-Elementpositionen. |
| `postings.bin`, `titles.bin` | uint32-Absatznummern; pro Wort streng aufsteigend |
| `chunks.idx` | 16 Byte pro Absatz: `Q I I`: Byteoffset, komprimierte Länge, Rohtextlänge |
| `chunks.bin` | Unabhängige zlib-Streams, UTF-8-Absätze einschließlich ursprünglichem Zeilenende |

Binäre Suche liest einen Wörterbucheintrag pro Schritt. Die Suche führt höchstens 64 Postingströme (32 Suchwörter mit Text und Titel) zusammen. Sie speichert lediglich 30 beste Kandidaten statt einer Punktetabelle über sämtliche Absätze. Titel-Nachsortierung entspricht dem bestehenden Python-Index; bei gleichen Punkten entscheidet die Absatznummer.

Grenzen: 4096 Zeichen pro Frage, 32 Suchwörter, 30 Ergebnisse, 64 KiB Rohtext pro Absatz. Die Speichergröße ist unabhängig von der Gesamtgröße des Index. Python-Interpreter und Laufzeit zählen zusätzlich; dies beweist keine ESP32-RAM-Belegung. Der aktuelle Referenzleser macht kleine Datei-Seeks pro Posting. SD-Karten-Latenz und ein gepufferter C/C++-Reader müssen am Board gemessen werden.

Der Leser prüft feste Dateigrößen, Postingbereiche/-nummern, aufsteigende Postings und begrenzte Dekompression. Dies ist keine vollständige Integritätsprüfung sämtlicher Daten: unbesuchte Wörter/Postings werden erst bei Zugriff geprüft; absichtlich verändert aber weiterhin gültige Inhalte werden nicht erkannt. Für Geräteauslieferung sind Prüfsummen und atomarer SD-Paketwechsel noch zu ergänzen.
