"""Schritt 6: mit dem Modell reden (oder einfach Text weiterschreiben lassen).

    python -m lm.generate                              # Chat mit ckpt/best.pt
    python -m lm.generate --ckpt ckpt/sft.pt           # nach dem Finetuning
    python -m lm.generate --raw --prompt "Der ESP32-P4 ist"   # reine Textfortsetzung
"""

import argparse
import sys

import torch
from tokenizers import Tokenizer

from lm.chat_format import build_prompt, mit_wissen
from lm.model import Config, TinyLM
from training.pretrain import pick_device


def load(ckpt_path, tokenizer_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TinyLM(Config(**ck["config"]))
    # ueber "art" statt ueber das alte Flag "ternaer": sonst wird ein
    # Sherry-Modell ohne Beschraenkung geladen, laeuft mit Schattengewichten
    # und faellt von 28/30 auf 1/30 - gemessen am 30.08.
    art = ck.get("art") or ("ternaer" if ck.get("ternaer") else None)
    if art:
        from lm.ternary import beschraenke
        beschraenke(model, art)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    tok = Tokenizer.from_file(tokenizer_path)
    print(f"{ckpt_path}: {model.num_params():,} Parameter, "
          f"Schritt {ck.get('step','?')}, val loss {ck.get('val', float('nan')):.4f}")
    return model, tok


def stream(model, tok, text, device, **kw):
    ids = torch.tensor([tok.encode(text, add_special_tokens=False).ids], device=device)
    if ids.size(1) >= model.cfg.block_size:
        ids = ids[:, -model.cfg.block_size + 1:]
    out = []
    for t in model.generate(ids, eos_id=tok.token_to_id("</s>"), **kw):
        if t == tok.token_to_id("</s>"):
            break
        out.append(t)
        # stueckweise dekodieren, damit Umlaute (2 Bytes) nicht zerhackt werden
        piece = tok.decode(out)
        sys.stdout.write(piece[len(getattr(stream, "_last", "")):])
        sys.stdout.flush()
        stream._last = piece
    stream._last = ""
    print()
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/best.pt")
    ap.add_argument("--tokenizer", default="tokenizer/de_bpe.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--raw", action="store_true", help="kein Chat, nur Textfortsetzung")
    ap.add_argument("--wissen", default=None,
                    help="Kontext, den das Modell zum Antworten benutzen soll")
    ap.add_argument("--prompt", default=None, help="einmalig statt interaktiv")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    device = pick_device(args.device)
    model, tok = load(args.ckpt, args.tokenizer, device)
    gen_kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                  top_k=args.top_k, top_p=args.top_p)

    if args.prompt is not None:
        frage = mit_wissen(args.prompt, args.wissen) if args.wissen else args.prompt
        text = frage if args.raw else build_prompt([{"role": "user", "content": frage}])
        stream(model, tok, text, device, **gen_kw)
        return

    print("\nTippe deine Frage. 'neu' loescht den Verlauf, 'exit' beendet.\n")
    history = []
    while True:
        try:
            user = input("du> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user:
            continue
        if user in ("exit", "quit"):
            break
        if user == "neu":
            history = []; print("(Verlauf geloescht)"); continue

        if args.raw:
            print("ki> ", end="")
            stream(model, tok, user, device, **gen_kw)
            continue

        history.append({"role": "user", "content": user})
        print("ki> ", end="")
        answer = stream(model, tok, build_prompt(history), device, **gen_kw)
        history.append({"role": "assistant", "content": answer})
        # Verlauf begrenzen, sonst sprengt er den 256-Token-Kontext
        if len(history) > 6:
            history = history[-6:]


if __name__ == "__main__":
    main()
