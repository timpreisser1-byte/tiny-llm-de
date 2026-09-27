#!/usr/bin/env python3
"""lm/memory_budget.py - passt das Modell in den ESP32-S3?

Drei Grenzen, die nicht verhandelbar sind:

    Flash    8 MB   Programm, Gewichte, Tokenizer - alles, was beim Start da ist
    PSRAM    8 MB   KV-Cache, Zwischenwerte, Puffer - und der Teil der Gewichte,
                    der nicht ins Flash passt und beim Start von der SD-Karte
                    geladen wird
    SD      32 GB   Wissen und alles, was je Token einmal gelesen und dann
                    wieder vergessen wird

Dieses Skript SCHAETZT NICHT, es exportiert wirklich und wiegt die Datei.
Die Schaetzung war am 19.09.2026 zweimal falsch:

  * Ternaere Matrizen werden mit 2,00 Bit gepackt (vier Trits je Byte), nicht
    mit 1,58. Der Unterschied sind bei 30 Mio. Gewichten 1,6 MB.
  * Die Worteinbettung bleibt nicht in voller Genauigkeit, sondern wird auf
    `--bits` quantisiert. Sie ist der groesste Einzelposten des Modells:
    8192 x 352 = 2,9 Mio. Gewichte, bei 8 Bit also 2,9 MB.

Zusammen hiess das: ein Aufbau, den die Schaetzung mit 7,57 MB auswies,
brauchte in Wahrheit 12,79 MB.

Gemessen am 19.09. mit `quantize.py --alle-stufen`: die Einbettung mit 8 Bit
kostet 0,4 % Loss, mit 6 Bit 1,7 %, mit 4 Bit 8,5 %. Voreinstellung ist
deshalb 8 Bit.

    python -m lm.memory_budget                       # alle gebauten Formen
    python -m lm.memory_budget --ckpt ckpt_gross/sft.pt
    python -m lm.memory_budget --form form_gross --bits 6
"""
import argparse
import os
import tempfile

FLASH = 8 * 1024 * 1024
PSRAM = 8 * 1024 * 1024
SD = 32 * 1000 ** 3

TOKENIZER_BYTE = 553_826          # tokenizer/de_bpe.json
LAUFZEIT_BYTE = 600 * 1024        # ESP-IDF, WLAN aus, eigener Rechenkern
PUFFER_BYTE = 64 * 1024           # SD-Lesepuffer, doppelt fuer Vorauslesen

F = "\033[1m"; G = "\033[32m"; R = "\033[31m"; Y = "\033[33m"
D = "\033[2m"; N = "\033[0m"


def gewichte_byte(cfg, bits, packung="trits"):
    """Wirkliche Groesse der Gewichtsdatei - durch Export gemessen.

    Die Nachschlagetabelle (`flash_tab`) wird ausgelassen: sie liegt auf der
    SD-Karte und wird je Token einmal gelesen.
    """
    from lm import model as M
    from lm import quantize as Q
    from lm.ternary import beschraenke, BitLinear

    netz = M.TinyLM(cfg)
    beschraenke(netz, "sherry" if packung == "sherry" else "ternaer")
    tern = {n + ".weight" for n, mod in netz.named_modules()
            if isinstance(mod, BitLinear)}
    sd = {k: v for k, v in netz.state_dict().items() if k != "flash_tab.weight"}
    pfad = tempfile.mktemp(suffix=".tlm")
    Q.schreibe_datei(pfad, cfg, sd, bits, ternaer=tern, packung=packung)
    groesse = os.path.getsize(pfad)
    os.unlink(pfad)
    tabelle = (netz.flash_tab.weight.numel() if cfg.d_flash else 0)
    kern = sum(v.numel() for v in sd.values())
    return groesse, kern, tabelle


def bilanz(cfg, bits=8, packung="trits", kv_bits=8):
    gew, kern, tabelle = gewichte_byte(cfg, bits, packung)

    # --- was fest im Flash steht ---
    fest = TOKENIZER_BYTE + LAUFZEIT_BYTE
    flash_frei = FLASH - fest
    im_flash = min(gew, flash_frei)
    # Der Rest wird beim Start von der SD-Karte ins PSRAM geladen.
    im_psram = gew - im_flash

    # --- was waehrend des Rechnens lebt ---
    kv = cfg.n_layer * 2 * cfg.block_size * cfg.kv_heads * cfg.head_dim * kv_bits / 8
    d_ff = 32 * ((cfg.d_ff + 31) // 32)
    zwischen = (cfg.d_model + 2 * d_ff) * 4
    tab_zeile = cfg.d_flash * 4 if cfg.d_flash else 0
    logits = cfg.vocab_size * 4
    psram = kv + zwischen + tab_zeile + logits + PUFFER_BYTE + im_psram

    return {
        "kern": kern, "gewichte": gew, "tabelle": tabelle,
        "flash": fest + im_flash, "psram": psram, "psram_gewichte": im_psram,
        "kv": kv, "sd_je_token": cfg.n_layer * cfg.d_flash if cfg.d_flash else 0,
    }


MIB = 1024 * 1024


def zeile(name, b):
    """Alles in MiB: der Chip hat 8 MiB = 8.388.608 Byte, nicht 8 Mio."""
    ff = G if b["flash"] < FLASH * .95 else (Y if b["flash"] <= FLASH else R)
    pf = G if b["psram"] < PSRAM * .95 else (Y if b["psram"] <= PSRAM else R)
    laden = f"{b['psram_gewichte']/MIB:.1f}M" if b["psram_gewichte"] else "—"
    print(f"  {name:<24}{b['kern']/1e6:>7.1f}M{b['gewichte']/MIB:>8.2f}M"
          f"{ff}{b['flash']/MIB:>8.2f}{N}{D}/8{N}"
          f"{pf}{b['psram']/MIB:>8.2f}{N}{D}/8{N}"
          f"{laden:>8}{b['tabelle']/1e6:>8.0f}M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--form", default=None, help="Preset aus model.PRESETS")
    ap.add_argument("--bits", type=int, default=8,
                    help="Genauigkeit der Einbettung: 8 kostet 0,4 %% Loss, "
                         "6 kostet 1,7 %%, 4 kostet 8,5 %% (gemessen)")
    ap.add_argument("--packung", default="trits", choices=["trits", "sherry"])
    args = ap.parse_args()

    from lm import model as M
    from dataclasses import asdict

    print()
    print(f"{F}  Modell                     Kern  Gewichte   Flash/MB  PSRAM/MB"
          f"  nachladen      SD{N}")
    print("  " + "-" * 76)

    if args.ckpt:
        import torch
        for p in args.ckpt:
            if not os.path.exists(p):
                print(f"  {os.path.basename(p)}: fehlt"); continue
            d = torch.load(p, map_location="cpu", weights_only=False)
            roh = d.get("config") or d.get("cfg")
            cfg = M.Config(**roh) if isinstance(roh, dict) else roh
            if cfg is None:
                print(f"  {os.path.basename(p)}: keine Form im Checkpoint"); continue
            zeile(os.path.basename(p), bilanz(cfg, args.bits, args.packung))
    elif args.form:
        zeile(args.form, bilanz(M.PRESETS[args.form], args.bits, args.packung))
    else:
        grund = M.PRESETS["form_16L"]
        for name, kw in [
            ("16L x 256, Ktx 384", dict(d_flash=256)),
            ("16L x 256, Ktx 2048", dict(d_flash=256, block_size=2048)),
            ("18L x 288, Ktx 2048", dict(n_layer=18, d_model=288, n_head=9,
                                         d_ff=768, d_flash=256, block_size=2048)),
            ("20L x 320, Ktx 2048", dict(n_layer=20, d_model=320, n_head=10,
                                         d_ff=832, d_flash=256, block_size=2048)),
            ("22L x 352, Ktx 2048", dict(n_layer=22, d_model=352, n_head=11,
                                         d_ff=928, d_flash=256, block_size=2048)),
            ("24L x 384, Ktx 2048", dict(n_layer=24, d_model=384, n_head=12,
                                         d_ff=1024, d_flash=256, block_size=2048)),
        ]:
            zeile(name, bilanz(M.Config(**{**asdict(grund), **kw}),
                               args.bits, args.packung))

    print()
    print(f"  {D}Gewichte = wirklich exportierte Datei ({args.packung}, "
          f"Einbettung {args.bits} Bit), ohne die Tabelle.{N}")
    print(f"  {D}Was nicht ins Flash passt, wird beim Start von der SD-Karte "
          f"ins PSRAM geladen - Spalte 'nachladen'.{N}")
    print(f"  {D}Flash/PSRAM in MiB - der Chip hat 8 MiB = 8.388.608 Byte.{N}")
    print(f"  {D}Flash = Gewichtsteil + Tokenizer 0,53 MiB + Laufzeit 0,59 MiB{N}")
    print(f"  {D}PSRAM = KV-Cache + Zwischenwerte + Logits + 64 KB Puffer "
          f"+ nachgeladene Gewichte{N}")
    print()


if __name__ == "__main__":
    main()
