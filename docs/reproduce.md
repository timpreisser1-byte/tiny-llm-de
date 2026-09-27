# Rebuilding the data

Datasets, indexes and checkpoints are not in the repository — together they are
more than 60 GB. This page rebuilds everything the answer chain needs. Run all
commands from the repository root.

Disk space needed: about 20 GB (raw Wikipedia 5.5 GB, full passage file 6 GB,
tiered index 1.5 GB, Wikidata tables 1 GB). RAM: 16 GB is enough; building the
*full* 12-passage index took most of a day because of swapping.

## 1. Wikipedia passages

Download the German Wikipedia snapshot (`wikimedia/wikipedia`, config
`20231101.de`, 20 Parquet files, 5.5 GB) into `wiki_roh/`:

```bash
mkdir -p wiki_roh && cd wiki_roh
for i in $(seq -w 0 19); do
  curl -sfLO "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.de/train-000${i}-of-00020.parquet"
done
cd ..
```

Cut every article into passages (up to 12 per article). This is the source for
the tiered index:

```bash
python -m knowledge.search_index bauen --quelle 'wiki_roh/*.parquet' --ziel wiki_voll \
    --max-artikel 0 --max-pro-artikel 12
```

## 2. Wikidata tables

Queried from the public [QLever](https://qlever.dev) endpoint. The scripts retry
empty answers and detect truncated downloads.

```bash
./scripts/fetch_wikidata.sh      # facts, dates, aliases, popularity, value ranks, ranking data
./scripts/fetch_works.sh         # songs, films, books (English and German titles)
```

This fills `fakten/`, `fakten_rang/`, `fakten_rangliste/` and `fakten_werk/`.

## 3. The tiered search index

```bash
python -m knowledge.build_index 140000:8,1000000:2 wiki_gestuft   # 1 M articles, 1.5 GB
python -m knowledge.facts --leitsaetze wiki_gestuft               # first sentence per article ("who was X?")
```

Optional — the on-device binary format used by the device reference chat:

```bash
python -m esp32_sd.sd_index export --source wiki_gestuft --output esp32_sd/wiki_sd_gestuft
```

## 4. External test sets

Download into `pruefsaetze/`:

- `mkqa.jsonl.gz` from https://github.com/apple/ml-mkqa (`dataset/mkqa.jsonl.gz`)
- `mintaka_train.json`, `mintaka_dev.json`, `mintaka_test.json` from
  https://github.com/amazon-science/mintaka (`data/`)

```bash
python -m evaluation.external_sets     # writes evaluation/questions/{mkqa,mkqa_dev,mintaka,mintaka_dev}.json
```

The selection is seeded, so the files match the ones in the repository.

## 5. Measure the chain

```bash
# rules + tables, no model
python -m evaluation.run_chain --fragen evaluation/questions/mkqa_dev.json \
    --regeln2 --fakten --absaetze 3 --zeichen 1500

# device order with a reader model
python -m evaluation.run_chain --fragen evaluation/questions/mkqa_dev.json \
    --kette --fakten --absaetze 3 --zeichen 1500 --strafe 1.05 --device mps \
    --ckpt ckpt_lesen/sft_b.pt

# precision of the table stages alone
python -m evaluation.stage_precision natural hand mkqa_dev mintaka_dev
```

Flags are German: `--fragen` questions, `--absaetze` passages, `--zeichen`
characters of context, `--fakten` use the Wikidata tables, `--regeln2` rules
with sentence fallback, `--kette` the device order (rule → model → fallback),
`--strafe` repetition penalty.

## 6. Training the reader (optional, GPU recommended)

Reading examples from real questions, in the exact format of the chain. Needs
GermanQuAD (the original download is offline; a copy is at
[`philschmid/prompted-germanquad`](https://huggingface.co/datasets/philschmid/prompted-germanquad),
saved as `pruefsaetze/germanquad_prompted.parquet`) and the Mintaka training split:

```bash
python -m training.real_reading_data     # -> data/sft/lesen_echt.jsonl
```

Fine-tune a reader from an existing chat checkpoint (the recipe of model B):

```bash
python -m training.sft --init <chat checkpoint> --out ckpt_lesen/sft_b.pt \
    --epochs 1 --batch-size 16 --lr 5e-5 --device cuda \
    --data data/sft/lesen_echt.jsonl:2 data/sft/ablenkung2.jsonl data/sft/ablehnung_sauber.jsonl \
           data/sft/kurzkontext.jsonl data/sft/grenzen_de.jsonl:10 data/sft/smalltalk_de.jsonl:10
```

On an RTX 4090 this takes about 12 minutes. `file:N` repeats a file N times.
The other files in the mix: `ablenkung2.jsonl` comes from
`training.distractor_data`, `ablehnung_sauber.jsonl` from `training.qa_pairs`
(both with `--ohne` to keep test passages out), `kurzkontext.jsonl` from
`training.short_context`. `grenzen_de.jsonl` (what the assistant cannot do) and
`smalltalk_de.jsonl` are small existing dialogue sets without a builder script.

Pretraining a model from scratch is covered by `scripts/train_big_model.sh`
(corpus → tokenizer → pretraining → fine-tuning → tests).
