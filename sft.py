"""Schritt 5b: Chat-Finetuning - aus dem Textmodell wird ein Gespraechspartner.

Startet vom Pretraining-Checkpoint und trainiert auf Dialogen. Der Loss zaehlt
nur auf den Antworten (siehe chat_format.encode_example).

    python sft.py --init ckpt/best.pt --epochs 3
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict

import torch
from tokenizers import Tokenizer

from chat_format import encode_example
from model import Config, TinyLM
from train import amp_kontext, pick_device


def load_jsonl(path, tok, block_size):
    beispiele = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            msgs = json.loads(line)["messages"]
            ids, labels = encode_example(tok, msgs, block_size)
            if sum(1 for l in labels if l != -100) < 4:
                continue  # nichts zu lernen
            beispiele.append((ids, labels))
    return beispiele


def make_batch(beispiele, idxs, pad_id, device):
    n = max(len(beispiele[i][0]) for i in idxs)
    x = torch.full((len(idxs), n), pad_id, dtype=torch.long)
    y = torch.full((len(idxs), n), -100, dtype=torch.long)
    for r, i in enumerate(idxs):
        ids, labels = beispiele[i]
        x[r, :len(ids)] = torch.tensor(ids)
        # Ziel ist jeweils das naechste Token
        y[r, :len(ids) - 1] = torch.tensor(labels[1:])
    return x.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="ckpt/best.pt", help="Pretraining-Checkpoint")
    ap.add_argument("--data", nargs="+", default=["data/sft/train.jsonl"],
                    help="eine oder mehrere Dateien, optional mit Gewicht: pfad:3")
    ap.add_argument("--tokenizer", default="tokenizer/de_bpe.json")
    ap.add_argument("--out", default="ckpt/sft.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    # Zielordner sofort anlegen, nicht erst beim Speichern: Sonst laeuft ein
    # Training stundenlang durch und stuerzt in der letzten Zeile ab.
    ziel = os.path.dirname(args.out)
    if ziel:
        os.makedirs(ziel, exist_ok=True)

    device = pick_device(args.device)
    tok = Tokenizer.from_file(args.tokenizer)
    pad_id = tok.token_to_id("<pad>")

    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    cfg = Config(**ck["config"])
    model = TinyLM(cfg)
    art = ck.get("art") or ("ternaer" if ck.get("ternaer") else None)
    if art:
        from ternaer import beschraenke
        beschraenke(model, art)
        print(f"Beschränktes Modell erkannt ({art}) - bleibt beim Finetuning")
    model.load_state_dict(ck["model"])
    model.to(device).train()
    print(f"Start von {args.init} ({model.num_params():,} Parameter) auf {device}")

    # Mehrere Quellen zusammenlegen. So kommen Spezialdaten (Kontextfragen,
    # eigene Ratgeber-Beispiele) zu den allgemeinen Dialogen dazu, ohne dass
    # der grosse Datensatz neu gebaut werden muss.
    daten = []
    for eintrag in args.data:
        pfad, _, gewicht = eintrag.rpartition(":")
        if not pfad or not gewicht.isdigit():
            pfad, gewicht = eintrag, "1"
        if not os.path.exists(pfad):
            print(f"  ! {pfad} fehlt, wird uebersprungen")
            continue
        teil = load_jsonl(pfad, tok, cfg.block_size)
        daten.extend(teil * int(gewicht))
        print(f"  {os.path.basename(pfad):22} {len(teil):>7,} x {gewicht}")
    if not daten:
        raise SystemExit(
            "\nKeine Trainingsdialoge geladen - Abbruch.\n"
            "Sonst wuerde ein vorhandenes Modell mit einem untrainierten "
            "ueberschrieben.\n"
            "Fehlt data/sft/train.jsonl? In Colab erst die Datenzelle ausfuehren, "
            "sie holt die Datei aus Drive zurueck.")

    random.seed(0); random.shuffle(daten)
    n_val = max(1, int(len(daten) * args.val_frac))
    val, train = daten[:n_val], daten[n_val:]
    print(f"{len(train):,} Trainings- und {len(val):,} Validierungsdialoge")

    steps_pro_epoche = max(1, len(train) // args.batch_size)
    total = steps_pro_epoche * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.05)
    ctx, scaler = amp_kontext(device)

    @torch.no_grad()
    def val_loss():
        model.eval()
        s = c = 0
        for i in range(0, len(val), args.batch_size):
            idxs = list(range(i, min(i + args.batch_size, len(val))))
            x, y = make_batch(val, idxs, pad_id, device)
            with ctx:
                _, loss = model(x, y)
            s += loss.item(); c += 1
        model.train()
        return s / max(c, 1)

    print(f"{total:,} Schritte ({args.epochs} Epochen)\n")
    t0, step, running, best = time.time(), 0, None, float("inf")
    for ep in range(args.epochs):
        order = list(range(len(train)))
        random.shuffle(order)
        # aehnlich lange Beispiele zusammen -> weniger Padding
        order.sort(key=lambda i: len(train[i][0]) // 32)
        batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        random.shuffle(batches)

        for idxs in batches:
            lr = args.lr * (step + 1) / args.warmup if step < args.warmup else \
                args.min_lr + 0.5 * (1 + math.cos(math.pi * (step - args.warmup) /
                                                  max(1, total - args.warmup))) * (args.lr - args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr

            x, y = make_batch(train, idxs, pad_id, device)
            with ctx:
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            (scaler.scale(loss) if scaler else loss).backward()
            if scaler and scaler.is_enabled():
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler and scaler.is_enabled():
                scaler.step(opt); scaler.update()
            else:
                opt.step()

            running = loss.item() if running is None else 0.9 * running + 0.1 * loss.item()
            if step % 50 == 0:
                print(f"epoche {ep+1} | step {step:6d}/{total} | loss {running:6.3f} "
                      f"| lr {lr:.2e} | {(time.time()-t0)/60:5.1f} min", flush=True)
            step += 1

        vl = val_loss()
        print(f"  ---- Epoche {ep+1}: val loss {vl:.4f}")
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "config": asdict(cfg),
                        "art": art, "ternaer": art == "ternaer",
                        "step": step, "val": vl}, args.out)
            print(f"       gespeichert -> {args.out}")

    print(f"\nFertig. Bester val loss {best:.4f}.  Test: python chat.py --ckpt {args.out}")


if __name__ == "__main__":
    main()
