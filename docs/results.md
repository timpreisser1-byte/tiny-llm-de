# Results and lessons

What was measured, what helped, and — just as important — what did not.
Numbers are "answered correctly" unless stated otherwise.

## Current numbers

Strict scoring: the expected answer must appear as whole words (before
2026-09-25 a substring match was used, which counted "acht" inside "Nacht";
the switch lowered all numbers by 2–5 points).

| Set | Rules + tables | Chain with reader model B | Invented (model B) |
|---|---:|---:|---:|
| MKQA test (500) | 8.6 % | 8.6 % | 0.2 % |
| Mintaka test (500) | 12.8 % | 11.8 % | 2.4 % |
| MKQA dev (500) | 10.8 % | 10.0 % | 0.6 % |
| Mintaka dev (500) | 12.6 % | 11.8 % | 2.4 % |
| natural (195) | 58.5 % | 56.4 % | 0.5 % |
| hand (60) | 66.7 % | 61.7 % | 1.7 % |

"Rules + tables" ends with a sentence fallback that quotes a whole sentence;
a quoted sentence containing the answer counts as correct. The reader model
answers with a short phrase instead — the two are level on the external tests.

## The biggest lessons

### 1. Your own test set flatters you

A 195-question set written during development reported 50 %. The same chain
scored **5 % on MKQA and 8 % on Mintaka**. The table stages had been tuned on
exactly those 195 questions, and the old search index happened to suit them.
Since then: development only on `*_dev` splits, each test set measured once
per step, and every change is judged on the external sets.

A second leak: generated training data once contained 82 % of the test
questions verbatim. Test questions are now blocked from every training set
(`training/real_reading_data.py`).

### 2. The knowledge was missing, not the model

The Wikipedia dump has no infoboxes: "Die Zugspitze ist mit  der höchste
Gipfel" — the height is simply gone. A full 8 GB index of all 2.85 M articles
did not help (top-3 retrieval 52.3 → 50.3 % on the own set). Wikidata tables
did: 30.8 → 50.3 % (lenient scoring at the time).

### 3. Which articles are indexed matters more than how they are ranked

Top-3 retrieval (strict), same ranking code:

| Index | own 195 | MKQA-dev | Mintaka-dev | Size |
|---|---:|---:|---:|---:|
| first 500 k articles in file order | 52.3 | 14.6 | 19.2 | 0.2 GB |
| 140 k most popular × 8 passages | 55.4 | 19.6 | 25.2 | 0.5 GB |
| **tiered: 140 k × 8 + up to 1 M × 2** | 51.8 | **23.0** | **28.6** | 1.2 GB |
| all 2.85 M articles × 12 | 50.3 | 23.0 | 29.6 | 8.1 GB |

Ranking tweaks were all within ±1 point (noise): German stemming (CISTEM),
compound splitting, lead-paragraph bonus, popularity bonus, answer-type bonus.
Popularity works as a *selection* of articles, not as a score bonus.

### 4. Precision before coverage

The table stages first answered many questions wrongly: 30 % precision on
Mintaka-dev ("In which state …?" got a year; "the author of The Hunger Games"
got someone else's birth year). Silence rules raised it to 86 %:

- a year only for "when" questions,
- the rest of the question must be exactly the subject,
- relative clauses switch the tables off,
- ranking questions must be fully explained word by word.

### 5. Reading must be trained — in the exact format it is used

| Change | Effect |
|---|---|
| Short-context reading data in the fine-tuning mix | without it the model does not read at all |
| Refusal examples of the right form | invented answers 45 → 23 %; rephrased questions 45 → 1.5 % |
| Distractor training (right passage among 1–3 hard distractors) | 3 passages 44.5 → 47.2 %, 4 passages 33.0 → 42.5 % |
| Real questions in chain format (GermanQuAD, Mintaka-train, MKQA rest) | invented 18.6 → 2.4 % and read rate 38 → 47 % (Mintaka-dev / MKQA-dev) |

Reader fine-tuning variants (from the distractor-trained model, lr 5e-5):

| Variant | natural | hand | MKQA-dev | Mintaka-dev | invented (Mintaka-dev) |
|---|---:|---:|---:|---:|---:|
| before | 54.4 | 58.3 | 8.2 | 7.0 | 18.6 % |
| A: real reading × 1, 1 epoch | 55.9 | 61.7 | 10.4 | 10.2 | 2.4 % |
| **B: real reading × 2, 1 epoch** | **56.4** | **61.7** | 10.0 | **11.8** | 2.4 % |
| C: real reading × 2, 2 epochs | 54.4 | 56.7 | 10.0 | 11.8 | 4.8 % (overfits) |

### 6. Knowledge stages that paid off

| Stage | Effect |
|---|---|
| Wikidata fact tables + value choice by rank/date/unit | see lesson 2 |
| Works (songs, films, books; English + German titles) | MKQA test 6.4 → 8.6 % |
| Events + comparisons | Mintaka test 10.6 → 11.8 % |
| Rankings (class × attribute × region) | Mintaka test 11.8 → 12.8 %, own 54.9 → 58.5 % |

Data traps found on the way: Olympus Mons (on Mars) as the world's highest
mountain, a reservoir with its area given "in metres", rivers inheriting
"Europe" through Russia, QLever silently dropping rows under a numeric FILTER
and truncating large results. Each is handled by a general rule, not by a
list of exceptions.

## Ideas that did not work

| Idea | Result |
|---|---|
| BM25 length normalisation (textbook default) | −20 points: passages are already equal in length |
| Continuing pretraining from a fine-tuned checkpoint | −14.5 points |
| Training the answer position (answer at the back of the context) | no gain; the position effect was mostly a difficulty confound |
| Expanding queries with attribute words ("lang", "hoch") | 52.3 → 50.8 % |
| Full Wikipedia index instead of a selection | worse on the own set, 5× slower |
| Two epochs of reading training | overfits, more invented answers |
| Question-to-question answer cache from other MKQA questions | almost no paraphrase overlap between MKQA questions — nothing to reuse |
| Gated per-layer embeddings (Gemma-style gate) | costs ~0.3 validation loss vs. ungated |

## Model architecture findings

- **Per-layer embeddings on the SD card** (256-dim, no gate): −0.23 validation
  loss — as much as all earlier improvements together, at almost no flash cost.
- **Ternary weights** (−1, 0, +1) make a 32 M-parameter core fit in ≈ 10 MiB.
- **Embedding quantisation:** 8 bit costs 0.4 % loss, 6 bit 1.7 %, 4 bit 8.5 %.
- **Multi-query attention** (one KV head) keeps the KV cache small for PSRAM.
