"""Schritt 4: Pretraining - das Modell lernt Deutsch.

Laeuft auf CUDA, Apple MPS oder CPU (automatisch erkannt).
Speichert regelmaessig nach ckpt/ und kann jederzeit fortgesetzt werden.

Beispiel:
    python train.py --preset 5m --steps 20000
    python train.py --resume            # weitermachen, wo es aufgehoert hat
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from model import Config, TinyLM, PRESETS


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_kontext(device):
    """Passenden Rechendatentyp waehlen - je nach Grafikkarte.

    bfloat16 ist am bequemsten, weil es keinen Loss-Scaler braucht. Aeltere
    NVIDIA-Karten (Turing, z.B. die T4 in Google Colab) koennen es aber nicht -
    dort muss float16 mit GradScaler benutzt werden, sonst laufen die
    Gradienten in die Null.
    """
    if device.type == "cuda":
        # Achtung: torch.cuda.is_bf16_supported() meldet auch dann True, wenn
        # die Karte bf16 nur emuliert (z.B. Tesla T4 in Google Colab) - das ist
        # dann deutlich langsamer als float16. Echte bf16-Hardware gibt es erst
        # ab Ampere, also compute capability 8.0.
        haupt, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if haupt >= 8 else torch.float16
    elif device.type == "mps":
        dtype = torch.bfloat16
    else:
        return torch.autocast("cpu", enabled=False), None

    scaler = torch.amp.GradScaler(device.type, enabled=(dtype is torch.float16))
    print(f"Rechendatentyp: {str(dtype).replace('torch.','')}"
          f"{' mit GradScaler' if scaler.is_enabled() else ''}")
    return torch.autocast(device.type, dtype=dtype), scaler


class Batches:
    """Zieht zufaellige Fenster aus den Token-Daten.

    Wichtig: es wird immer aus einem RAM-Puffer gelesen, nie direkt aus der
    memmap. Zufaellige Zugriffe auf eine Datei sind um Faktor 10-20 langsamer,
    das Training haengt dann an der Platte statt an der GPU. Ist die Datei
    groesser als das Fenster, wird abschnittsweise nachgeladen - so bleibt der
    Speicherbedarf klein, auch wenn der Korpus Gigabytes gross ist.
    """

    def __init__(self, path, block_size, batch_size, device,
                 window_tokens=50_000_000, window_steps=1500, seed=None):
        self.mm = np.memmap(path, dtype=np.uint16, mode="r")
        self.n_total = len(self.mm)
        if self.n_total < block_size + 1:
            raise SystemExit(f"{path} ist zu klein ({self.n_total} Token)")
        self.bs, self.T, self.device = batch_size, block_size, device
        # Fenster: ein zusammenhaengender Ausschnitt wird sequenziell in den RAM
        # gelesen und fuer window_steps Schritte benutzt. Danach ein neuer
        # Ausschnitt. Haelt den Speicherbedarf klein und die Platte sequenziell.
        self.win_n = min(window_tokens, self.n_total)
        self.win_steps = window_steps
        self.used = window_steps  # erzwingt Laden beim ersten get()
        self.data = None
        # Mit seed sehen zwei Laeufe exakt dieselben Daten. Fuer Vergleiche
        # unverzichtbar: ohne das schwankt der Verlust zwischen zwei gleichen
        # Laeufen um bis zu 0,04 - mehr als mancher gemessene Unterschied.
        self.seed = seed
        self.rng = np.random.default_rng(seed) if seed is not None else np.random

    def zuruecksetzen(self):
        """Von vorn anfangen - fuer Versuchsreihen unverzichtbar.

        Ohne das teilen sich alle Varianten einer Reihe EINEN Zufallsstrom:
        die erste sieht Stapel 0 bis n, die zweite n bis 2n, und so weiter.
        Sie trainieren dann auf verschiedenen Daten, und der gemessene
        Unterschied ist teils Verfahren, teils Datenausschnitt. Gemessen sind
        das bis zu 0,04 Verlust - mehr als die meisten Effekte, die man sucht.
        """
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        self.used = self.win_steps      # erzwingt Neuladen des Fensters
        return self
        print(f"  {path}: {self.n_total/1e6:.1f} Mio. Token, "
              f"Fenster {self.win_n/1e6:.0f} Mio. ({self.win_n*2/1e6:.0f} MB RAM)")

    def _reload(self):
        if self.win_n >= self.n_total:
            if self.data is None:
                self.data = np.array(self.mm)  # echte Kopie im RAM
        else:
            start = int(self.rng.integers(0, self.n_total - self.win_n))\
                if hasattr(self.rng, "integers") else \
                    np.random.randint(0, self.n_total - self.win_n)
            self.data = np.array(self.mm[start:start + self.win_n])  # sequenziell
        self.used = 0

    def __len__(self):
        return self.n_total

    def get(self):
        if self.used >= self.win_steps:
            self._reload()
        self.used += 1
        hoch = len(self.data) - self.T - 1
        ix = (self.rng.integers(0, hoch, size=self.bs)
              if hasattr(self.rng, "integers")
              else np.random.randint(0, hoch, size=self.bs))
        # ein einziger fancy-index Zugriff statt bs Einzelscheiben
        win = ix[:, None] + np.arange(self.T + 1)[None, :]
        block = self.data[win].astype(np.int64)
        x, y = torch.from_numpy(block[:, :-1]), torch.from_numpy(block[:, 1:])
        if self.device.type == "cuda":
            return x.pin_memory().to(self.device, non_blocking=True), \
                   y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)


def lr_at(step, warmup, total, lr, min_lr, plan="kosinus"):
    """Kosinus faellt von Anfang an, WSD haelt lange konstant.

    Gemessen: WSD schlaegt Kosinus bei jeder geprueften Lernrate um vier bis
    fuenf Prozent. Und praktisch: Ein WSD-Lauf, der mittendrin abbricht, ist
    nicht halb kaputt - die Lernrate war ja noch auf ihrem Arbeitswert.
    """
    if step < warmup:
        return lr * (step + 1) / warmup
    if step > total:
        return min_lr
    r = (step - warmup) / max(1, total - warmup)
    if plan == "wsd":
        # 80 % konstant, dann linear auf min_lr herunter
        if r < 0.8:
            return lr
        return min_lr + (lr - min_lr) * (1 - r) / 0.2
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (lr - min_lr)


@torch.no_grad()
def evaluate(model, batches, iters, ctx):
    model.eval()
    losses = torch.zeros(iters)
    for i in range(iters):
        x, y = batches.get()
        with ctx:
            _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="5m", choices=list(PRESETS))
    ap.add_argument("--init", default=None,
                    help="Gewichte aus einem fertigen Checkpoint übernehmen und "
                         "neu trainieren (z.B. float -> ternär umstellen)")
    ap.add_argument("--kv-head", type=int, default=0,
                    help="Anzahl KV-Köpfe (GQA/MQA). 0 = so viele wie Abfrageköpfe")
    ap.add_argument("--qk-norm", action="store_true",
                    help="Abfragen und Schlüssel normieren (8-12 %, 640 Parameter)")
    ap.add_argument("--wert-residuum", action="store_true",
                    help="Werte der ersten Schicht durchreichen (nochmal ~3 %)")
    ap.add_argument("--kontext", type=int, default=0,
                    help="Kontextlänge überschreiben (Vorgabe 256). Der KV-Cache "
                         "wächst linear mit; 384 kostet 6,3 statt 4,2 MB")
    ap.add_argument("--flash", type=int, default=0,
                    help="Nachschlagetabelle je Schicht, Breite je Schicht")
    ap.add_argument("--flash-tor", action="store_true",
                    help="Tor: Modell entscheidet je Token, wie stark die Tabelle zaehlt")
    ap.add_argument("--flash-kontext", action="store_true",
                    help="Kontextanteil aus der Worteinbettung zusaetzlich zur Tabelle")
    ap.add_argument("--teilen", type=int, default=1,
                    help="Schichten mehrfach durchlaufen: 2 = halber Speicher, "
                         "doppelte Rechenzeit je Token")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--plan", default="kosinus", choices=["kosinus", "wsd"],
                    help="Verlauf der Lernrate")
    ap.add_argument("--muon-lr", type=float, default=0.0,
                    help="Muon für die Matrizen, mit dieser eigenen Lernrate "
                         "(gemessenes Optimum 2e-2). 0 = nur AdamW")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--window-tokens", type=int, default=50_000_000,
                    help="wie viele Token gleichzeitig im RAM liegen")
    ap.add_argument("--window-steps", type=int, default=1500,
                    help="Schritte pro Fenster, danach wird neu geladen")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compile", action="store_true", help="torch.compile (nur CUDA sinnvoll)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sherry", action="store_true",
                    help="3 von 4 Gewichten ungleich null - 1,25 Bit statt 2,0")
    ap.add_argument("--ternaer", action="store_true",
                    help="Gewichte auf -1/0/+1 beschränken (BitNet-Art)")
    ap.add_argument("--binaer", action="store_true",
                    help="Gewichte auf -1/+1 beschränken (1 Bit je Gewicht)")
    args = ap.parse_args()
    if args.ternaer and args.binaer:
        raise SystemExit("--ternaer und --binaer schliessen sich aus")
    art = "sherry" if getattr(args, "sherry", False) else "ternaer" if args.ternaer else "binaer" if args.binaer else None
    vor_art = None

    device = pick_device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.manual_seed(1337)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ctx, scaler = amp_kontext(device)

    # --- Modell ---
    ckpt_path = os.path.join(args.ckpt_dir, "last.pt")
    start_step, best_val = 0, float("inf")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = Config(**ck["config"])
        model = TinyLM(cfg)
        # aeltere Checkpoints kennen nur das Feld "ternaer"
        art = ck.get("art") or ("ternaer" if ck.get("ternaer") else None) or art
        if art:
            from ternaer import beschraenke
            beschraenke(model, art)
        model.load_state_dict(ck["model"])
        start_step, best_val = ck["step"], ck.get("best_val", float("inf"))
        print(f"Fortsetzung ab Schritt {start_step} (val loss {best_val:.4f})")
    elif args.init:
        # Umstellung: fertige Gewichte uebernehmen, aber bei Schritt 0 anfangen.
        # Die Beschraenkung kommt NACH dem Laden - die beschraenkten Schichten
        # greifen dann auf dieselben Gewichtstensoren zu.
        vor = torch.load(args.init, map_location="cpu", weights_only=False)
        cfg = Config(**vor["config"])
        aenderungen = {}
        if args.teilen > 1 and cfg.geteilt != args.teilen:
            aenderungen["geteilt"] = args.teilen
        if aenderungen:
            cfg = Config(**{**asdict(cfg), **aenderungen})
        model = TinyLM(cfg)
        vor_art = vor.get("art") or ("ternaer" if vor.get("ternaer") else None)
        if vor_art:
            from ternaer import beschraenke
            beschraenke(model, vor_art)
        fehlend, unerwartet = model.load_state_dict(vor["model"], strict=False)
        print(f"Start von {args.init} (Schritt {vor.get('step',0)}, "
              f"val {vor.get('val', float('nan')):.4f})")
        if fehlend or unerwartet:
            print(f"  {len(fehlend)} Gewichte neu, {len(unerwartet)} übrig")
        if vor_art and vor_art != art:
            print(f"  Beschränkung wechselt: {vor_art} -> {art or 'keine'}")
        ck = None
    else:
        cfg = PRESETS[args.preset]
        neu = {}
        if args.teilen > 1:
            neu["geteilt"] = args.teilen
        if args.kv_head:
            neu["n_kv_head"] = args.kv_head
        if args.flash:
            neu["d_flash"] = args.flash
        if args.flash_tor:
            neu["flash_tor"] = True
        if args.flash_kontext:
            neu["flash_kontext"] = True
        if args.kontext:
            neu["block_size"] = args.kontext
        if args.qk_norm:
            neu["qk_norm"] = True
        if args.wert_residuum:
            neu["wert_residuum"] = True
        if neu:
            cfg = Config(**{**asdict(cfg), **neu})
        # Vokabulargroesse aus dem echten Tokenizer uebernehmen
        meta = os.path.join("tokenizer", "de_bpe_meta.json")
        if os.path.exists(meta):
            v = json.load(open(meta))["vocab_size"]
            if v != cfg.vocab_size:
                print(f"Tokenizer hat {v} Token -> vocab_size angepasst")
                cfg = Config(**{**asdict(cfg), "vocab_size": v})
        model = TinyLM(cfg)
        ck = None
    if art and not (args.resume and ck) and not (args.init and vor_art == art):
        # Muss VOR dem Optimizer passieren: die beschraenkten Schichten teilen
        # sich die Gewichtstensoren, aber der Optimizer merkt sich Referenzen.
        from ternaer import beschraenke
        n_ersetzt = beschraenke(model, art)
        stufen = "-1/0/+1" if art == "ternaer" else "-1/+1"
        print(f"{art.capitalize()}: {n_ersetzt} Schichten auf {stufen} beschränkt")

    model.to(device)
    n = model.num_params()
    if cfg.geteilt > 1:
        print(f"Geteilt: {cfg.n_layer} Durchgänge über "
              f"{cfg.n_layer//cfg.geteilt} eigene Schichten")
    if cfg.kv_heads != cfg.n_head:
        kv = 2 * cfg.n_layer * cfg.block_size * cfg.kv_heads * cfg.head_dim * 2
        print(f"KV-Köpfe: {cfg.kv_heads} statt {cfg.n_head} "
              f"-> Cache {kv/1e6:.2f} MB statt "
              f"{kv*cfg.n_head//cfg.kv_heads/1e6:.2f} MB")
    if cfg.qk_norm or cfg.wert_residuum:
        teile = [n for n, an in (("QK-Norm", cfg.qk_norm),
                                 ("Wert-Residuum", cfg.wert_residuum)) if an]
        print(f"Zusätze: {', '.join(teile)}")
    if cfg.d_flash:
        tab = cfg.vocab_size * cfg.n_layer * cfg.d_flash
        print(f"Flash-Tabelle: {tab:,} Werte = {tab*0.5/1e6:.2f} MB bei 4 Bit, "
              f"je Token {cfg.n_layer*cfg.d_flash*0.5:.0f} Byte")
    # bei --init/--resume kommt die Form aus dem Checkpoint, nicht aus --preset
    name = (args.preset if not (args.init or (args.resume and ck))
            else f"{cfg.n_layer}x{cfg.d_model}")
    print(f"Modell '{name}': {n:,} Parameter "
          f"({model.num_params(True):,} ohne Embedding) auf {device}")

    # --- Daten ---
    train_b = Batches(os.path.join(args.data_dir, "train.bin"), cfg.block_size,
                      args.batch_size, device,
                      window_tokens=args.window_tokens, window_steps=args.window_steps)
    val_b = Batches(os.path.join(args.data_dir, "val.bin"), cfg.block_size,
                    args.batch_size, device, window_tokens=10**12)
    tokens_per_step = args.batch_size * args.grad_accum * cfg.block_size
    print(f"Daten: {len(train_b)/1e6:.1f} Mio. Train-Token, "
          f"{len(val_b)/1e6:.1f} Mio. Val-Token")
    print(f"{tokens_per_step:,} Token pro Schritt -> "
          f"{args.steps * tokens_per_step/1e6:.0f} Mio. Token gesamt "
          f"({args.steps * tokens_per_step/len(train_b):.1f} Epochen)")

    # --- Optimizer: weight decay nur auf Matrizen, nicht auf Norms ---
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    if args.muon_lr:
        from optimierer import baue
        opts, basis_lr = baue(model, args.lr, args.muon_lr, args.weight_decay)
        print(f"Muon für die Matrizen (lr {args.muon_lr:.0e}), "
              f"AdamW für den Rest (lr {args.lr:.0e})")
    else:
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay},
             {"params": nodecay, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
            fused=(device.type == "cuda"),
        )
        opts, basis_lr = [opt], [args.lr]
    if ck is not None and "optimizer" in ck and len(opts) == 1:
        opts[0].load_state_dict(ck["optimizer"])

    if args.compile:
        model = torch.compile(model)

    def save(name, step, val):
        torch.save({"model": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                    "optimizer": opts[0].state_dict() if len(opts) == 1 else None,
                    "config": asdict(cfg), "art": art,
                    # "ternaer" bleibt fuer aeltere Leser (status.py, sft.py)
                    "ternaer": art == "ternaer",
                    "step": step, "best_val": best_val, "val": val},
                   os.path.join(args.ckpt_dir, name))

    print("\nStart. Abbruch mit Strg+C ist ungefaehrlich - der letzte "
          "Checkpoint bleibt erhalten.\n")
    t0 = time.time()
    running = None
    try:
        for step in range(start_step, args.steps):
            lr = lr_at(step, args.warmup, args.steps, args.lr, args.min_lr,
                       args.plan)
            anteil = lr / args.lr           # gilt fuer alle Optimierer gleich
            for o, b in zip(opts, basis_lr):
                for g in o.param_groups:
                    g["lr"] = b * anteil

            for o in opts:
                o.zero_grad(set_to_none=True)
            for _ in range(args.grad_accum):
                x, y = train_b.get()
                with ctx:
                    _, loss = model(x, y)
                skaliert = loss / args.grad_accum
                (scaler.scale(skaliert) if scaler else skaliert).backward()
            if scaler and scaler.is_enabled():
                for o in opts:
                    scaler.unscale_(o)     # vor dem Clipping zurueckskalieren
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler and scaler.is_enabled():
                for o in opts:
                    scaler.step(o)
                scaler.update()
            else:
                for o in opts:
                    o.step()

            l = loss.item()
            running = l if running is None else 0.9 * running + 0.1 * l
            if step % 20 == 0:
                dt = time.time() - t0
                tps = tokens_per_step * (step - start_step + 1) / max(dt, 1e-6)
                print(f"step {step:6d} | loss {running:6.3f} | ppl {math.exp(min(running,20)):8.1f} "
                      f"| lr {lr:.2e} | {tps:7.0f} tok/s | {dt/60:5.1f} min", flush=True)

            if step > 0 and step % args.eval_every == 0:
                vl = evaluate(model, val_b, args.eval_iters, ctx)
                mark = ""
                if vl < best_val:
                    best_val = vl
                    save("best.pt", step, vl)
                    mark = "  <- neuer Bestwert, best.pt gespeichert"
                save("last.pt", step, vl)
                print(f"  ---- val loss {vl:.4f} | ppl {math.exp(min(vl,20)):.1f}{mark}", flush=True)
    except KeyboardInterrupt:
        print("\nAbgebrochen.")

    vl = evaluate(model, val_b, args.eval_iters, ctx)
    if vl < best_val:
        best_val = vl
        save("best.pt", args.steps, vl)
    save("last.pt", args.steps, vl)
    print(f"\nFertig. val loss {vl:.4f} | bester {best_val:.4f} | "
          f"Checkpoints in {args.ckpt_dir}/")


if __name__ == "__main__":
    main()
