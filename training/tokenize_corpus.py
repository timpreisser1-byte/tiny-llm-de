"""Schritt 3: Korpus in Token-IDs umwandeln und als .bin ablegen.

Ausgabe sind zwei flache uint16-Dateien (train/val), die beim Training
per memmap gelesen werden - kein RAM-Problem, egal wie gross der Korpus wird.

Beispiel:
    python -m training.tokenize_corpus --input data/corpus.txt
"""

import argparse
import os

import numpy as np
from tokenizers import Tokenizer

from lm.chat_format import BOS as BOS_TEXT, EOS as EOS_TEXT

DOC_CHUNK = 2000  # Dokumente pro Encode-Batch


def docs(path):
    """Liest die Korpusdatei dokumentweise (getrennt durch Leerzeile)."""
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if buf:
                    yield "".join(buf).strip()
                    buf = []
            else:
                buf.append(line)
    if buf:
        yield "".join(buf).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/corpus.txt")
    ap.add_argument("--tokenizer", default="tokenizer/de_bpe.json")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.005)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    assert tok.get_vocab_size() <= 65535, "uint16 reicht nur bis 65535 Token"
    bos, eos = tok.token_to_id("<s>"), tok.token_to_id("</s>")

    os.makedirs(args.out_dir, exist_ok=True)
    train_f = open(os.path.join(args.out_dir, "train.bin"), "wb")
    val_f = open(os.path.join(args.out_dir, "val.bin"), "wb")

    n_train = n_val = n_docs = 0
    batch = []

    def flush(batch):
        nonlocal n_train, n_val, n_docs
        if not batch:
            return
        for text, enc in zip(batch, tok.encode_batch(batch)):
            # Dialoge aus training/build_mix.py bringen <s> und </s> schon mit; nur reiner
            # Fliesstext (Wikipedia, eigene Dateien) braucht die Marken noch.
            # Ohne diese Pruefung stuende im Korpus "<s><s>" und "</s></s>" -
            # das Modell wuerde lernen, die Marken zu verdoppeln.
            ids = enc.ids
            if not text.startswith(BOS_TEXT):
                ids = [bos] + ids
            if not text.endswith(EOS_TEXT):
                ids = ids + [eos]
            ids = np.array(ids, dtype=np.uint16)
            # jedes 1/val_frac-te Dokument wandert in die Validierung
            if args.val_frac > 0 and n_docs % max(2, int(1 / args.val_frac)) == 0:
                val_f.write(ids.tobytes()); n_val += len(ids)
            else:
                train_f.write(ids.tobytes()); n_train += len(ids)
            n_docs += 1

    for doc in docs(args.input):
        batch.append(doc)
        if len(batch) >= DOC_CHUNK:
            flush(batch); batch = []
            print(f"  {n_docs:>8,} Dokumente | {(n_train+n_val)/1e6:8.2f} Mio. Token", flush=True)
    flush(batch)

    train_f.close(); val_f.close()
    print(f"\nFertig: {n_docs:,} Dokumente")
    print(f"  train.bin : {n_train/1e6:.2f} Mio. Token")
    print(f"  val.bin   : {n_val/1e6:.2f} Mio. Token")
    print(f"\nFaustregel: ein 5M-Modell will mindestens ~100 Mio. Token sehen, "
          f"besser 300-500 Mio.")


if __name__ == "__main__":
    main()
