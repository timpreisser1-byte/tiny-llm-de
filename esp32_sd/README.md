# ESP32-S3 device reference

This folder is the host-side reference of the device: the file formats, the
search logic, the answer chain and the personal memory that are meant to be
ported to the ESP32-S3. It is **not firmware yet** — it runs in Python on a
desktop so that every design decision can be measured first.

## What is here

| File | Purpose |
|---|---|
| `chat.py` | The device chat: memory → smalltalk → Wikidata tables (`knowledge/facts.py`) → SD search → extraction rules, optionally a reader model. |
| `memory.py` | Personal memory — how the device learns from conversations (see below). |
| `sd_index.py` | The on-device search index format: exporter plus a reference reader that never loads the whole index into RAM. Format: [INDEX_FORMAT.md](INDEX_FORMAT.md). |
| `extractive_answer.py` | Small, ESP32-friendly extraction rules ("what is", "where is", "how high/long/deep", "when was") and the sentence fallback. |
| `smalltalk.py` | Greetings, identity, help, thanks, current topic — without touching the SD card. |
| `model_backend.py` | Loads a PyTorch checkpoint on CPU and trims the context so the question always fits the model window. |
| `model_manifest.py` | Read-only checkpoint audit that writes sidecar reports for memory estimates. |
| `benchmark.py` | Retrieval timing and hit rate on the host. |
| `eval_chat.py` | Black-box test of the chat with a larger question battery. |
| `test_*.py` | 55 unit tests. |

Not in the repository: the index folders `wiki_sd_gestuft/` (current,
popularity-tiered, 1.2 GB) and `wiki_sd/` (older), the reports in `reports/`,
and the memory log `gedaechtnis.jsonl` — it contains what you told the device
about yourself.

## Running

From the repository root:

```bash
python -m esp32_sd.chat "Wie hoch ist der Eiffelturm?"          # one question
python -m esp32_sd.chat --json "Wie hoch ist der Eiffelturm?"   # with stage and source
python -m esp32_sd.chat                                         # interactive
python -m esp32_sd.chat --ohne-fakten                           # without Wikidata tables (faster start)
python -m esp32_sd.chat --checkpoint ckpt_lesen/sft_b.pt        # with a reader model
```

In the interactive chat, `/neu` resets the topic and `/ende` quits.

## Learning from conversations

| You say | The device |
|---|---|
| `Merk dir: Mein Hund heißt Bello` (remember: …) | stores the fact; "Wie heißt mein Hund?" → Bello |
| `Falsch, richtig ist 452 Kilometer` (wrong, it is …) | stores a correction bound to the last question; it wins over every other stage from now on |
| `Stimmt.` (right) | caches the last answer as confirmed |
| `Vergiss meinen Hund` (forget …) | removes the entry and rewrites the log, so the text is really gone from the card |
| `Was weißt du über mich?` (what do you know about me?) | lists the stored facts |

Design choices, each covered by a test:

- **Append-only JSON lines** on the SD card: robust on FAT; a power cut can
  only lose the last line, which is skipped on the next start.
- **Newest wins:** a newer fact or correction replaces the older one.
- **Politeness is not a correction:** "Nein danke" (no thanks) does not store "danke".
- **Memory answers are never frozen:** confirming an answer that came from the
  memory itself does not lock it against later changes.
- **"My …" questions never go to Wikipedia:** "Wie heißt mein Hund?" (what is
  my dog called) would otherwise match the article "Mein Katalonien"
  (Homage to Catalonia); without a stored fact the device says it has not been told yet.

The weights are never trained on the device: training needs full-precision
shadow weights (≈ 140 MB for this model) and backpropagation, which neither fit
8 MiB of PSRAM nor finish in conversation time on a 240 MHz core.

## Memory picture

Without a language model the search only holds a few stream states, up to 30
candidates, one decompressed passage and short answer buffers; the large index
stays on the card. The last host run measured a Python tracemalloc peak of
574 176 bytes for the search — not a proof for the ESP32, but a good sign for a
fixed C/C++ memory plan well below 8 MiB.

## Porting plan

1. Copy the binary index (`wiki_sd_gestuft/`) and the Wikidata tables (as sorted
   files) to the SD card.
2. Port `sd_index.py` to C/C++: binary search in `vocab.bin`, posting merge over
   small stream states, zlib/miniz decompression of single passages, at most 32
   search terms.
3. Port the table lookups (`knowledge/facts.py`, `knowledge/works.py`) as binary
   search over sorted title files.
4. Port the model runtime: read the config from the manifest, load packed
   ternary weights, reserve a fixed KV cache in PSRAM, stream the per-layer
   embedding rows from the SD card.
5. Stabilise a text chat over serial or a small display first.
6. Then voice: wake word and commands are realistic on the ESP32-S3 with
   ESP-SR; free German speech recognition and German TTS are not covered by
   Espressif's ready-made models (see `speech/` for the own experiments).

Hardware references:

- https://docs.espressif.com/projects/esp-idf/en/v5.4.1/esp32s3/api-guides/external-ram.html
- https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/speech_command_recognition/README.html

## Open points

- No ESP-IDF firmware yet, and no run on a real ESP32-S3 with 8 MiB PSRAM.
- No C/C++ loader for the model format yet.
- SD latency and a buffered C reader have to be measured on the board.
- No robust German speech recognition / speech output on the target chip yet.
