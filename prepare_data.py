"""Schritt 1: deutschen Textkorpus einsammeln.

Quellen:
  --source wiki    deutsche Wikipedia. Die Parquet-Dateien werden direkt von
                   Hugging Face geladen (Shard 0 ~780 MB, Shard 1 ~450 MB, ...).
                   Ein Shard reicht fuer den ersten Trainingslauf locker.
  --source local   alle .txt/.md Dateien aus data/raw/ (dein eigenes Wissen)

Ergebnis: eine UTF-8-Textdatei, Dokumente durch eine Leerzeile getrennt.

Beispiele:
    python prepare_data.py --source wiki --shards 1
    python prepare_data.py --source wiki --shards 3 --out data/corpus_gross.txt
    python prepare_data.py --source local
"""

import argparse
import os
import re
import time
import urllib.request

REPO = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.de"
SHARD = "train-{i:05d}-of-00020.parquet"
MIN_CHARS = 250  # zu kurze Artikel bringen dem Modell nichts

# Abschnitte, die fast nur aus Listen/Links bestehen -> abschneiden
CUT_MARKERS = ("\nSiehe auch\n", "\nWeblinks\n", "\nLiteratur\n",
               "\nEinzelnachweise\n", "\nAnmerkungen\n", "\nQuellen\n")


def clean(text: str) -> str:
    text = text.replace(" ", " ").replace("​", "")
    for marker in CUT_MARKERS:
        i = text.find(marker)
        if i > MIN_CHARS:
            text = text[:i]
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def download(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  schon da: {dest}")
        return
    print(f"  lade {url}")
    tmp = dest + ".part"
    t0 = [time.time()]

    def hook(blocks, bs, total):
        if total > 0 and blocks % 200 == 0:
            done = blocks * bs
            print(f"\r    {done/1e6:7.0f} / {total/1e6:.0f} MB "
                  f"({done/1e6/max(time.time()-t0[0],1e-6):5.1f} MB/s)", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, hook)
    print()
    os.rename(tmp, dest)


def from_wiki(n_shards: int, cache_dir: str):
    import pyarrow.parquet as pq

    os.makedirs(cache_dir, exist_ok=True)
    for i in range(n_shards):
        name = SHARD.format(i=i)
        path = os.path.join(cache_dir, name)
        download(f"{REPO}/{name}", path)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
            for t in batch.column("text").to_pylist():
                yield clean(t)


def from_local(folder: str):
    if not os.path.isdir(folder):
        raise SystemExit(f"{folder} gibt es nicht - lege dort deine .txt-Dateien ab.")
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith((".txt", ".md")):
                with open(os.path.join(root, f), encoding="utf-8", errors="ignore") as fh:
                    yield clean(fh.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["wiki", "local"], default="wiki")
    ap.add_argument("--shards", type=int, default=1, help="Wikipedia-Shards (0-20)")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = alle")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--cache-dir", default="data/parquet")
    ap.add_argument("--out", default="data/corpus.txt")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    gen = from_wiki(args.shards, args.cache_dir) if args.source == "wiki" \
        else from_local(args.raw_dir)

    t0, chars, kept = time.time(), 0, 0
    with open(args.out, "w", encoding="utf-8") as out:
        for doc in gen:
            if len(doc) < MIN_CHARS:
                continue
            out.write(doc + "\n\n")
            chars += len(doc)
            kept += 1
            if kept % 20000 == 0:
                print(f"  {kept:>8,} Dokumente | {chars/1e6:7.1f} Mio. Zeichen "
                      f"| {time.time()-t0:5.0f}s", flush=True)
            if args.max_docs and kept >= args.max_docs:
                break

    print(f"\nFertig: {kept:,} Dokumente, {chars/1e6:.1f} Mio. Zeichen -> {args.out}")
    print(f"Das sind grob ~{chars/3.5/1e6:.0f} Mio. Token.")


if __name__ == "__main__":
    main()
