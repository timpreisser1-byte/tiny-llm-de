# Architecture

This document explains how Tiny-LLM DE answers a question, where each piece
would live on the ESP32-S3, and what every file in the repository does.

- [The answer chain](#the-answer-chain)
- [The language model](#the-language-model)
- [Knowledge on the SD card](#knowledge-on-the-sd-card)
- [Personal memory](#personal-memory)
- [Memory and time budget on the ESP32-S3](#memory-and-time-budget-on-the-esp32-s3)
- [File by file](#file-by-file)

## The answer chain

A question passes through a series of stages. Each stage either answers with
confidence or stays silent and hands over to the next one. The order matters:
cheaper and more precise stages come first.

| # | Stage | Code | Answers when |
|---|---|---|---|
| 1 | Personal memory | `esp32_sd/memory.py` | the user taught or corrected this before, or asks about "my …" |
| 2 | Smalltalk | `esp32_sd/smalltalk.py` | greetings, thanks, "who are you" |
| 3 | Comparison | `knowledge/facts.py` (`vergleich`) | "Who is older, X or Y?" — both values exist in the tables |
| 4 | Ranking | `knowledge/facts.py` (`rangliste`) | "Which mountain is the highest in Europe?" — class, attribute and region are all recognised |
| 5 | Special forms | `knowledge/facts.py` | unit arithmetic, nicknames, "which city lies on river X", "who was X" |
| 6 | Works | `knowledge/works.py` | songs, films, books: performer, director, author, composer, release year |
| 7 | Fact tables | `knowledge/facts.py` (`tabelle`) | subject + attribute: height, length, area, population, founding, capital … |
| 8 | Search | `knowledge/search_index.py` | always — returns the 3 best Wikipedia passages |
| 9 | Precise rule | `knowledge/rules.py` | a passage contains an answer of the asked type (a height for "how high") |
| 10 | Reader model | `lm/inference.py` | the model finds the answer in the passages; otherwise it says "not in the text" |
| 11 | Sentence fallback | `esp32_sd/extractive_answer.py` | quote the best matching sentence |

### Why so many stages stay silent

Early versions of the table stage answered 18 of 500 Mintaka development
questions with a ranking — and only 2 were right. The stages now refuse unless
the question is fully understood:

- **Every word must be explained.** A ranking question may only contain a
  question word, a class ("mountain"), a superlative ("highest"), a recognised
  place and filler words. "Which state is *one of the five* largest…" stays silent.
- **The span must match exactly.** In "When was *the author of The Hunger Games*
  born?" the subject is not "The Hunger Games". Date and "who wrote" questions
  require that the rest of the question is exactly the found subject.
- **Relative clauses mean multi-hop.** A comma followed by "der/die/das/dessen…"
  switches the table stages off.
- **The answer type must fit.** A year is only given when the question asks
  "when" or "in which year".

With these rules the precision of the table stages rose from 30 % to 86 % on
Mintaka-dev and is 100 % on the hand-written set.

## The language model

A decoder-only transformer in the Llama style, kept deliberately simple so the
inference can be rewritten in C for the microcontroller (`lm/model.py`, preset
`form_gross`):

| | |
|---|---|
| Layers × width | 22 × 352 |
| Attention | 11 query heads, **1** key/value head (multi-query), QK-norm, RoPE |
| Feed-forward | SwiGLU, 928 |
| Extras | value residual, per-layer embeddings (256-dim table per layer, stored on SD) |
| Vocabulary | 8192 (German byte-level BPE, `tokenizer/`) |
| Weights | **ternary** (−1, 0, +1; BitNet-style training in `lm/ternary.py`), packed 4 per byte |
| Core parameters | 32.4 M (≈ 9.9 MiB packed) |
| Per-layer table | 46 M parameters on the SD card; 5.6 KB are read per token |

The model is not expected to *know* facts. It is trained to read three passages
and either copy the answer or say "Das steht nicht im Text." ("that is not in
the text"). Training history in [results.md](results.md).

## Knowledge on the SD card

### Wikipedia search index

- Source: German Wikipedia (`wikimedia/wikipedia`, snapshot 2023-11-01).
- **Selection by popularity** (`knowledge/build_index.py`): the 140 000 best-known
  articles with up to 8 passages each, the next ones up to rank 1 000 000 with 2
  passages each. Popularity = number of Wikipedia language editions (Wikidata
  sitelinks). This beat both "first 500 000 articles in file order" and "all
  2.85 M articles" on external questions.
- **Ranking**: BM25-style term weights, a title bonus, and a pair-proximity
  re-ranking of the top 200 candidates (`knowledge/search_index.py`).
- **On-device format** (`esp32_sd/sd_index.py`, [INDEX_FORMAT.md](../esp32_sd/INDEX_FORMAT.md)):
  a sorted vocabulary searched by binary search, `uint32` posting lists merged
  as streams while keeping only the best candidates, zlib-compressed passages.
  No table with one score per passage is ever held in memory.

### Wikidata tables

The Wikipedia dump has no infoboxes, so heights, depths, lengths and many dates
are simply missing from the text. They come from Wikidata instead
(`scripts/fetch_wikidata.sh`, `scripts/fetch_works.sh`, queried via QLever):

| Table | Examples |
|---|---|
| numbers with unit, rank and date | height, building height, depth, length, area, population, diameter |
| dates | founding, birth, death, start, end, dissolution, point in time |
| names | capital, currency, language, composer, inventor, author, founder, river |
| works (English and German titles) | performer, director, author, composer, lyricist, release date |
| rankings | class (mountain, river, lake, country, city, island, ocean, continent, planet), continent, country, mountain range |
| helpers | aliases, popularity (sitelinks), first sentence of each article |

Choosing the right value matters as much as having it: preferred rank beats
normal rank, the newest census wins for population, units are normalised
(`knowledge/fact_values.py`), and among several works with the same title the
best-known one (the original, not a cover version) wins.

On the device each table becomes a sorted file searched by binary search.

## Personal memory

`esp32_sd/memory.py` implements learning from conversations without touching
the weights:

- **Format:** one JSON object per line, append-only — robust on FAT, a power
  cut can only lose the last line. `{"id": 7, "art": "fakt", "text": "..."}`.
- **Signals:** facts ("Merk dir: …" — remember), corrections ("Falsch, richtig
  ist …" — wrong, correct is …), confirmations ("Stimmt." — right), deletions
  ("Vergiss …" — forget).
- **Priority:** corrections are bound to the question they answer and win over
  every other stage the next time.
- **Privacy:** forgetting rewrites the log so the text is really gone from the
  removable card; the log itself is never part of the repository.

## Memory and time budget on the ESP32-S3

| Resource | Budget | Used by |
|---|---|---|
| Flash (8 MiB) | 8.0 MiB | memory-mapped part of the ternary weights |
| PSRAM (8 MiB) | ≈ 5.9 MiB | remaining weights (≈ 3 MiB), KV cache, buffers |
| SD card (32 GB) | ≈ 2.3 GB | per-layer table, search index, Wikidata tables, memory log |

Measured on the desktop with a cost model (`knowledge/build_index.py` docstring):
per question the search reads a median 164 KB of posting lists and 200 short
passages. Each passage read is a random access (~1 ms on SD), so re-ranking
depth, not index size, dominates: 1000 candidates ≈ 1.3 s, 200 candidates
≈ 0.3 s at the same answer quality. These are estimates — nothing has been
measured on the real board yet.

## File by file

### `knowledge/` — the knowledge side

| File | What it does |
|---|---|
| `search_index.py` | Builds and queries the Wikipedia passage index (BM25-style, title bonus, pair proximity). |
| `build_index.py` | Builds the popularity-tiered index from the full dump and Wikidata sitelinks. |
| `facts.py` | All Wikidata stages: fact tables, events, comparisons, rankings, special forms, silence rules. |
| `fact_values.py` | Picks the right value among several Wikidata statements (rank, date, unit). |
| `works.py` | Songs, films, books and games with English and German titles. |
| `rules.py` | Maps a question to an answer type and extracts a typed answer from a passage. |

### `lm/` — the language model

| File | What it does |
|---|---|
| `model.py` | The transformer and its presets; parameter and memory report. |
| `ternary.py` | Ternary (BitNet-style) layers and training helpers. |
| `optimizer.py` | Muon optimizer for matrix parameters. |
| `quantize.py` | Packs and exports weights for the device. |
| `memory_budget.py` | Exports a model and weighs the result against flash and PSRAM. |
| `inference.py` | Loads a checkpoint and answers a question from passages ("the whole device in one file"). |
| `generate.py` | Plain text generation / chat with a checkpoint. |
| `chat_format.py` | The prompt format shared by fine-tuning and inference (`Wissen: …\nFrage: …`). |

### `training/` — building the model

| File | What it does |
|---|---|
| `prepare_data.py` | Collects the German text corpus. |
| `fineweb_data.py` | Downloads German web text (FineWeb-2 HQ) for large pretraining runs. |
| `build_mix.py` | Mixes Wikipedia and German dialogue into one corpus. |
| `train_tokenizer.py` | Trains the German byte-level BPE tokenizer. |
| `tokenize_corpus.py` | Turns the corpus into token ids (`.bin`). |
| `pretrain.py` | Pretraining (WSD schedule, curriculum window, ternary option). |
| `sft.py` | Supervised fine-tuning on chat / reading data. |
| `dialog_data.py` | Reads the various German dialogue datasets into one format. |
| `get_sft_data.py` | Downloads German question–answer data for fine-tuning. |
| `high_quality.py` | Curates clean, natively German dialogues. |
| `short_context.py` | Short look-up contexts — the reading regime the model needs. |
| `reading_templates.py` | Prompt templates for reading examples (restored from bytecode, see file). |
| `question_forms.py` | Many phrasings per question pattern, with one held out for testing. |
| `qa_pairs.py` | Question–answer pairs with varied forms and hard refusal examples. |
| `distractor_data.py` | Teaches picking the right passage among several distractors. |
| `real_reading_data.py` | Reading examples from real questions (GermanQuAD, Mintaka, MKQA) in the exact chain format. |

### `evaluation/` — measuring

| File | What it does |
|---|---|
| `run_chain.py` | Runs the whole chain on a question set: found / answered / read / invented. `--kette` is the device order. |
| `external_sets.py` | Builds the German MKQA and Mintaka test and development sets. |
| `stage_precision.py` | Precision of the table stages alone. |
| `ranking_test.py` | Precision of the ranking stage alone. |
| `generate_questions.py` | Generates questions with verifiable answers from the knowledge base. |
| `clean_test_set.py` | Builds a large held-out question set and blocks its passages from training. |
| `fixed_validation.py` | Fixed validation folds for comparable losses. |
| `compare_models.py` | Compares checkpoints on one fixed validation set. |
| `reading_test.py` | Can the model pick the number out of a given passage? |
| `context_damage.py` | Does a given passage take answers away from the model? |
| `battery.py` | Systematic test battery to find weak spots. |
| `broad100.py` | A hundred questions across everything. |
| `questions/` | Question sets — see below. |

| Question set | Size | Origin |
|---|---:|---|
| `mkqa.json`, `mkqa_dev.json` | 500 + 500 | MKQA, German, test and development split |
| `mintaka.json`, `mintaka_dev.json` | 500 + 500 | Mintaka, German, test and development split |
| `natural.json` | 195 | everyday questions written for this project |
| `hand.json`, `new60.json` | 60 + 60 | hand-written questions |
| `superlatives.json` | 31 | geographic superlatives, written before the ranking stage |
| `templated1000.json` | 1000 | generated from templates with the source passage (answers partly noisy) |

### `esp32_sd/` — the device reference

| File | What it does |
|---|---|
| `chat.py` | Desktop reference of the device chat: memory → smalltalk → tables → SD search → extraction / model. |
| `memory.py` | Personal memory (learning from conversations). |
| `sd_index.py` | On-device index format: exporter and bounded-memory reference reader. |
| `extractive_answer.py` | Small extraction rules and the sentence fallback. |
| `smalltalk.py` | Deterministic smalltalk. |
| `model_backend.py` | Optional model backend for the chat. |
| `model_manifest.py` | Read-only checkpoint audit for memory estimates. |
| `benchmark.py`, `eval_chat.py` | Retrieval and black-box chat measurements on the host. |
| `test_*.py` | Unit tests (55). |

### `speech/` — experimental voice

| File | What it does |
|---|---|
| `asr_data.py` | Turns speech data into features. |
| `asr_model.py` | CTC speech recogniser for free transcription. |
| `asr_beam.py` | Beam search with a lexicon during decoding. |
| `asr_lexicon.py` | Forces recognised output onto real German words. |
| `asr_lm_rescore.py` | Rescores recogniser hypotheses with the own language model. |
| `asr_synth.py` | Synthetic training speech from the own speech output. |
| `audio_data.py` | Generates recogniser training data with system voices. |
| `keyword_model.py` | Small keyword spotter (convolutional network on mel features). |
| `tts_phonemes.py` | German text to phonemes. |
| `tts_units.py` | How many sound units German speech output needs. |
| `tts_cut.py` | Cuts sound units from the public Thorsten-Voice recordings. |
| `tts_align.py` | Aligns phonemes to audio with the own recogniser. |
| `tts_speak.py` | Concatenative text-to-speech from the sound units. |
| `tts_model.py`, `tts_speaker.py`, `tts_train.py` | A small neural voice (three networks after sanoTTS): model, assembly, training. |
| `tts_grade.py` | Grades the voice with the recogniser. |
| `teacher_voice.py`, `teacher_data.py` | A Piper VITS voice as teacher for the small student voice. |
| `voice_device.py` | The whole device by voice: listen, look up, answer, speak. |

### `scripts/`

| File | What it does |
|---|---|
| `fetch_wikidata.sh` | Downloads all fact, ranking and value tables from Wikidata (QLever), with retries and truncation checks. |
| `fetch_works.sh` | Downloads the works tables (songs, films, books). |
| `train_big_model.sh` | End-to-end recipe for the 15 M device model with 1.25-bit weights: pretrain → SFT → reading test → battery. |
| `eval_readers.sh` | Evaluates reader checkpoints in the device chain on the development sets. |
| `train_speech.sh` | Trains the speech recogniser locally. |
