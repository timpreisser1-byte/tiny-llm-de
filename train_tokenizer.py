"""Schritt 2: deutschen Tokenizer trainieren (Byte-Level-BPE).

Byte-Level heisst: jedes Byte ist darstellbar, es gibt kein <unk>, Umlaute und
Emojis funktionieren immer. Genau dieses Verfahren laesst sich spaeter in C
auf dem ESP32 nachbauen (Merge-Tabelle + Byte-Alphabet).

Der Tokenizer wird EINMAL trainiert und danach eingefroren. Wenn du ihn spaeter
aenderst, musst du das Modell komplett neu trainieren.

Beispiel:
    python train_tokenizer.py --input data/corpus.txt --vocab-size 8192
"""

import argparse
import json
import os

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# Steuertoken. Die IDs 0-4 bleiben fest, damit die C-Engine sie hart kodieren kann.
SPECIALS = ["<pad>", "<unk>", "<s>", "</s>", "\n<user>\n", "\n<assistant>\n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/corpus.txt")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--out", default="tokenizer/de_bpe.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    tok = Tokenizer(models.BPE(unk_token=None))
    # Ziffern einzeln -> Rechnen/Zahlen werden fuer das Modell viel lernbarer
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )
    print(f"Trainiere Tokenizer auf {args.input} (vocab={args.vocab_size}) ...")
    tok.train([args.input], trainer)
    tok.save(args.out)

    meta = {"vocab_size": tok.get_vocab_size(),
            "specials": {s: tok.token_to_id(s) for s in SPECIALS}}
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nGespeichert: {args.out}  (vocab_size={meta['vocab_size']})")
    print("Steuertoken:", meta["specials"])

    # Qualitaetskontrolle: wie viele Zeichen pro Token schafft der Tokenizer?
    proben = [
        "Wie funktioniert der ESP32-P4?",
        "Ein Kondensator speichert elektrische Ladung zwischen zwei Platten.",
        "Guten Morgen! Kannst du mir das Ohmsche Gesetz erklaeren?",
        "Die Straße war während der Überschwemmung gesperrt.",
    ]
    print("\nProbe:")
    ges_c = ges_t = 0
    for s in proben:
        ids = tok.encode(s).ids
        ges_c += len(s)
        ges_t += len(ids)
        print(f"  {len(ids):3d} Token | {s}")
        print(f"      {[tok.decode([i]) for i in ids][:14]}")
    print(f"\nSchnitt: {ges_c/ges_t:.2f} Zeichen pro Token "
          f"(gut fuer Deutsch sind >3.0)")


if __name__ == "__main__":
    main()
