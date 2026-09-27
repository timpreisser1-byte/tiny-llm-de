"""Schritt 6: Quantisierung - Gewichte klein machen fuer den ESP32-P4.

Verfahren: blockweise symmetrische Quantisierung. Die Gewichte einer Matrix
werden in Bloecke zu je BLOCK Zahlen zerlegt. Pro Block wird nur der groesste
Betrag gemerkt (die Skala, als float16), alle Werte darin werden auf wenige
Bits abgebildet:

    w_quant = round(w / skala * max_wert)      beim Packen
    w  ~=  w_quant * skala / max_wert          beim Rechnen

Warum blockweise und nicht fuer die ganze Matrix? Ein einziger Ausreisser
wuerde sonst die Skala fuer Millionen Gewichte verderben. Mit 64er-Bloecken
kostet die Skala nur 16 Bit pro 64 Werte - also 0,25 Bit pro Gewicht.

    python -m lm.quantize --ckpt ckpt/sft.pt --bits 4 --messen
    python -m lm.quantize --ckpt ckpt/sft.pt --alle-stufen     # 8/4/3/2 vergleichen

Ausgabeformat (--out): eigenes, kompaktes Binaerformat fuer die spaetere
C-Engine. Aufbau siehe schreibe_datei() weiter unten.
"""

import argparse
import json
import math
import os
import struct

import numpy as np
import torch

from lm.model import Config, TinyLM

MAGIC = b"TLM1"          # Tiny Language Model, Version 1
BLOCK = 64               # Gewichte pro Skala


def quantisiere(w: torch.Tensor, bits: int, block: int = BLOCK):
    """Tensor -> (Quantisierte Werte int8, Skalen float16). Symmetrisch, blockweise."""
    flach = w.detach().float().flatten()
    n = flach.numel()
    auffuellen = (-n) % block
    if auffuellen:
        flach = torch.cat([flach, torch.zeros(auffuellen)])
    bloecke = flach.view(-1, block)

    grenze = 2 ** (bits - 1) - 1                      # 4 Bit -> 7
    max_betrag = bloecke.abs().amax(dim=1, keepdim=True)
    skalen = (max_betrag / grenze).clamp(min=1e-12)
    q = torch.round(bloecke / skalen).clamp(-grenze - 1, grenze).to(torch.int8)
    return q, skalen.half().flatten(), n


def entquantisiere(q, skalen, n, block: int = BLOCK):
    """Zurueckrechnen - so wie es die C-Engine spaeter auch macht."""
    w = (q.float() * skalen.float().unsqueeze(1)).flatten()[:n]
    return w


def packe_bits(q: torch.Tensor, bits: int) -> bytes:
    """int8-Werte in einen dichten Bitstrom pressen (4 Bit -> 2 Werte pro Byte).

    Vor dem Packen wird der Nullpunkt verschoben (aus -8..7 wird 0..15), damit
    nur positive Zahlen gespeichert werden. Die Bits liegen LSB zuerst -
    genau so, wie eine C-Schleife sie wieder herausholt.
    """
    werte = (q.flatten().to(torch.int16) + (1 << (bits - 1))).numpy().astype(np.uint8)
    if bits == 8:
        return werte.tobytes()
    # (n, bits) Matrix aus Einzelbits, dann dicht zusammenschieben
    einzel = ((werte[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).astype(np.uint8)
    strom = einzel.reshape(-1)
    rest = (-len(strom)) % 8
    if rest:
        strom = np.concatenate([strom, np.zeros(rest, dtype=np.uint8)])
    return np.packbits(strom, bitorder="little").tobytes()


def entpacke_bits(rohdaten: bytes, anzahl: int, bits: int) -> np.ndarray:
    """Gegenstueck zu packe_bits - dient als Probe, dass das Format stimmt."""
    if bits == 8:
        werte = np.frombuffer(rohdaten, dtype=np.uint8)[:anzahl].astype(np.int16)
    else:
        strom = np.unpackbits(np.frombuffer(rohdaten, dtype=np.uint8),
                              bitorder="little")[:anzahl * bits]
        einzel = strom.reshape(anzahl, bits).astype(np.int16)
        werte = (einzel << np.arange(bits, dtype=np.int16)).sum(axis=1)
    return werte - (1 << (bits - 1))


def fehler(original, zurueck):
    """Relativer Fehler in Prozent - wie stark hat sich das Gewicht verschoben?"""
    return (original - zurueck).abs().mean().item() / (original.abs().mean().item() + 1e-12) * 100


def analysiere(state_dict, bits_liste=(8, 7, 6, 5, 4, 3, 2)):
    """Was kostet welche Bitzahl - an Groesse und an Genauigkeit?"""
    print(f"\n{'Bits':>5} {'Groesse':>10} {'pro Gewicht':>12} {'mittl. Fehler':>14}")
    print("-" * 46)
    ergebnisse = {}
    for bits in bits_liste:
        gesamt_bytes = 0
        fehler_summe = gewichte = 0
        for name, w in state_dict.items():
            if w.dim() < 2:                 # Normen bleiben float16, viel zu klein
                gesamt_bytes += w.numel() * 2
                continue
            if name == "lm_head.weight":    # teilt sich die Gewichte mit dem Embedding
                continue
            q, skalen, n = quantisiere(w, bits)
            gesamt_bytes += math.ceil(n * bits / 8) + skalen.numel() * 2
            zurueck = entquantisiere(q, skalen, n)
            fehler_summe += fehler(w.flatten().float(), zurueck) * n
            gewichte += n
        mittel = fehler_summe / max(gewichte, 1)
        pro_gewicht = gesamt_bytes * 8 / max(gewichte, 1)
        ergebnisse[bits] = (gesamt_bytes, pro_gewicht, mittel)
        print(f"{bits:>5} {gesamt_bytes/1e6:>8.2f} MB {pro_gewicht:>10.2f} bit "
              f"{mittel:>12.1f} %")
    return ergebnisse


@torch.no_grad()
def messe_qualitaet(model, cfg, bits, daten="data/val.bin", batches=25):
    """Perplexitaet vorher/nachher auf echten Daten - der ehrliche Test.

    Kleiner Fehler in den Gewichten heisst nicht automatisch gleiche Qualitaet.
    Deshalb wird das quantisierte Modell wirklich ausgefuehrt.
    """
    if not os.path.exists(daten):
        print(f"({daten} fehlt - Qualitaetsmessung uebersprungen)")
        return None, None

    d = np.fromfile(daten, dtype=np.uint16)
    rng = np.random.default_rng(0)
    proben = []
    for _ in range(batches):
        i = rng.integers(0, len(d) - cfg.block_size - 1)
        proben.append(torch.from_numpy(
            d[i:i + cfg.block_size + 1].astype(np.int64)))
    x = torch.stack([p[:-1] for p in proben])
    y = torch.stack([p[1:] for p in proben])

    model.eval()
    _, loss_vorher = model(x, y)

    # Gewichte durch ihre quantisierte Fassung ersetzen
    original = {}
    for name, p in model.named_parameters():
        if p.dim() < 2:
            continue
        original[name] = p.detach().clone()
        q, skalen, n = quantisiere(p, bits)
        p.copy_(entquantisiere(q, skalen, n).view_as(p))

    _, loss_nachher = model(x, y)

    for name, p in model.named_parameters():   # wiederherstellen
        if name in original:
            p.copy_(original[name])

    return loss_vorher.item(), loss_nachher.item()


def packe_trits(w):
    """Ternaere Matrix -> vier Trits je Byte plus eine Skala.

    Ein ternaeres Gewicht braucht theoretisch log2(3) = 1,585 Bit. Fuenf Trits
    passten in ein Byte (3^5 = 243), das waeren 1,6 Bit - aber das Auspacken
    braucht Divisionen durch 3. Vier Trits je Byte kosten 2,0 Bit und lassen
    sich mit Schieben auspacken, was auf dem ESP32 deutlich billiger ist.
    Dieselbe Wahl trifft die Atome-Engine.
    """
    from lm.ternary import ternarisiere
    skala = w.abs().mean().clamp(min=1e-8)
    stufen = (w / skala).round().clamp(-1, 1).to(torch.int8) + 1     # 0,1,2
    flach = stufen.flatten().numpy().astype(np.uint8)
    auffuellen = (-len(flach)) % 4
    if auffuellen:
        flach = np.concatenate([flach, np.ones(auffuellen, dtype=np.uint8)])
    v = flach.reshape(-1, 4)
    gepackt = (v[:, 0] | (v[:, 1] << 2) | (v[:, 2] << 4) | (v[:, 3] << 6))
    return gepackt.astype(np.uint8), float(skala)



def packe_sherry(w):
    """3:4-ternaere Matrix -> fuenf Bit je vier Gewichte, plus eine Skala.

    In jedem Viererblock ist genau eines null. Vier Moeglichkeiten fuer seine
    Lage mal acht Vorzeichenmuster ergeben 32 Zustaende - exakt fuenf Bit.

        Bit 4-3   welches der vier Gewichte ist null
        Bit 2-0   Vorzeichen der drei anderen, 1 = minus

    Acht solcher Fuenfergruppen ergeben vierzig Bit, also genau fuenf Bytes.
    Damit bleibt alles an Bytegrenzen ausgerichtet und das Auspacken auf dem
    ESP32 braucht nur Schieben und Maskieren - keine Division, kein Rest.

    1,25 Bit je Gewicht statt der 2,0 der Trit-Packung.
    """
    from lm.ternary import sherrisiere
    q = sherrisiere(w)
    skala = q.abs().max().clamp(min=1e-8)
    stufen = (q / skala).round().to(torch.int8).flatten().numpy()
    auffuellen = (-len(stufen)) % 32
    if auffuellen:
        stufen = np.concatenate([stufen, np.tile([1, 1, 1, 0], auffuellen // 4)])
    v = stufen.reshape(-1, 4)
    null = np.argmin(np.abs(v), axis=1).astype(np.uint8)          # Lage der Null
    vz = (v < 0).astype(np.uint8)
    # die drei Vorzeichen in fester Reihenfolge, die Nullstelle uebersprungen
    bits = np.zeros(len(v), dtype=np.uint8)
    for k in range(4):
        maske = null != k
        # Position des Bits: wie viele Nicht-Null-Stellen liegen vor k
        stelle = (k - (null < k)).astype(np.int64)
        bits[maske] |= (vz[maske, k] << (2 - stelle[maske])).astype(np.uint8)
    code = (null << 3) | bits                                      # 0..31
    # acht Fuenfbit-Codes in fuenf Bytes
    g = code.reshape(-1, 8).astype(np.uint64)
    wort = np.zeros(len(g), dtype=np.uint64)
    for k in range(8):
        wort |= g[:, k] << np.uint64(5 * k)
    raus = np.zeros((len(g), 5), dtype=np.uint8)
    for b in range(5):
        raus[:, b] = ((wort >> np.uint64(8 * b)) & np.uint64(0xFF)).astype(np.uint8)
    return raus.flatten(), float(skala)


def entpacke_sherry(daten, n):
    """Gegenprobe zum Packer - dieselbe Logik, wie sie in C stehen wird."""
    g = daten.reshape(-1, 5).astype(np.uint64)
    wort = np.zeros(len(g), dtype=np.uint64)
    for b in range(5):
        wort |= g[:, b] << np.uint64(8 * b)
    code = np.zeros((len(g), 8), dtype=np.uint8)
    for k in range(8):
        code[:, k] = ((wort >> np.uint64(5 * k)) & np.uint64(31)).astype(np.uint8)
    code = code.flatten()
    null, bits = code >> 3, code & 7
    raus = np.zeros((len(code), 4), dtype=np.int8)
    for k in range(4):
        maske = null != k
        stelle = (k - (null < k)).astype(np.int64)
        vz = (bits[maske] >> (2 - stelle[maske])) & 1
        raus[maske, k] = np.where(vz == 1, -1, 1)
    return raus.flatten()[:n]


def schreibe_datei(pfad, cfg, state_dict, bits, tokenizer_meta=None, ternaer=(),
                   packung="trits"):
    """Kompaktes Binaerformat fuer die C-Engine.

        "TLM1" | version u32 | bits u32 | block u32
        | vocab u32 | n_layer u32 | n_head u32 | d_model u32 | d_ff u32 | block_size u32
        | anzahl_tensoren u32
        dann je Tensor:
            namenlaenge u32 | name | dims u32 | shape u32... | anzahl u32
            | typ u32 (0 = quantisiert, 1 = float16 roh)
            | daten | skalen (nur bei typ 0)

    Alles little-endian, wie beim ESP32.
    """
    with open(pfad, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<10I", 1, bits, BLOCK, cfg.vocab_size, cfg.n_layer,
                            cfg.n_head, cfg.d_model, cfg.d_ff, cfg.block_size,
                            sum(1 for k in state_dict if k != "lm_head.weight")))
        for name, w in state_dict.items():
            if name == "lm_head.weight":
                continue
            roh = name.encode()
            f.write(struct.pack("<I", len(roh))); f.write(roh)
            f.write(struct.pack("<I", w.dim()))
            for s in w.shape:
                f.write(struct.pack("<I", s))
            f.write(struct.pack("<I", w.numel()))
            if name in ternaer and w.dim() == 2:
                # Typ 2: vier Trits je Byte (2,00 Bit), Typ 3: Sherry 3:4
                # (1,25 Bit). Beide mit einer Skala fuer die ganze Matrix.
                if packung == "sherry":
                    gepackt, skala = packe_sherry(w.detach().float())
                    f.write(struct.pack("<I", 3))
                else:
                    gepackt, skala = packe_trits(w.detach().float())
                    f.write(struct.pack("<I", 2))
                f.write(struct.pack("<f", skala))
                f.write(gepackt.tobytes())
            elif w.dim() < 2:
                f.write(struct.pack("<I", 1))
                f.write(w.detach().half().numpy().tobytes())
            else:
                q, skalen, n = quantisiere(w, bits)
                f.write(struct.pack("<I", 0))
                f.write(packe_bits(q, bits))
                f.write(skalen.numpy().tobytes())
    if tokenizer_meta and os.path.exists(tokenizer_meta):
        ziel = os.path.splitext(pfad)[0] + "_tokenizer.json"
        with open(tokenizer_meta) as q, open(ziel, "w") as z:
            z.write(q.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/best.pt")
    ap.add_argument("--bits", type=int, default=4, choices=[8, 7, 6, 5, 4, 3, 2])
    ap.add_argument("--out", default=None, help="Binaerdatei schreiben")
    ap.add_argument("--packung", default="trits", choices=["trits", "sherry"],
                    help="trits = 2,00 Bit, sherry = 1,25 Bit (3 von 4)")
    ap.add_argument("--messen", action="store_true",
                    help="Perplexitaet vorher/nachher auf data/val.bin")
    ap.add_argument("--alle-stufen", action="store_true",
                    help="alle Bitzahlen von 8 bis 2 vergleichen")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["config"])
    art = ck.get("art") or ("ternaer" if ck.get("ternaer") else None)
    model = TinyLM(cfg)
    if art:
        from lm.ternary import beschraenke
        beschraenke(model, art)
    model.load_state_dict(ck["model"])
    sd = ck["model"]

    # Welche Matrizen sind ternaer? Alles, was beschraenkt wurde - also die
    # Schichten in den Bloecken, aber nicht Embedding und Ausgabe.
    # Auf die Klasse pruefen, nicht auf den Namen der Beschraenkung: sonst
    # faellt ein Sherry-Modell durch und alle Matrizen werden mit vier Bit
    # geschrieben statt mit 1,25. Gemessen kostete das 7,04 statt 3,15 MB.
    BESCHRAENKT = {"BitLinear", "SherryLinear", "SchwellLinear",
                   "SherrySkala", "BinLinear", "BinSkala"}
    ternaer_namen = set()
    for n_, m in model.named_modules():
        if type(m).__name__ in BESCHRAENKT:
            ternaer_namen.add(f"{n_}.weight")
    if ternaer_namen:
        # Sherry nur packen, wenn das Modell auch so trainiert wurde
        if art == "sherry" and args.packung == "trits":
            args.packung = "sherry"
        bit = 1.25 if args.packung == "sherry" else 2.0
        print(f"Beschränktes Modell ({art}): {len(ternaer_namen)} Matrizen "
              f"mit {bit} Bit, Rest mit {args.bits} Bit")

    n = sum(v.numel() for k, v in sd.items() if k != "lm_head.weight")
    print(f"{args.ckpt}: {n:,} Parameter, val loss {ck.get('val', float('nan')):.4f}")
    print(f"unquantisiert: fp32 {n*4/1e6:.1f} MB | fp16 {n*2/1e6:.1f} MB")

    stufen = (8, 7, 6, 5, 4, 3, 2) if args.alle_stufen else (args.bits,)
    analysiere(sd, stufen)

    if args.messen or args.alle_stufen:
        print(f"\n{'Bits':>5} {'loss vorher':>12} {'loss nachher':>13} "
              f"{'ppl nachher':>12} {'Verschlechterung':>17}")
        print("-" * 64)
        for bits in stufen:
            vorher, nachher = messe_qualitaet(model, cfg, bits)
            if vorher is None:
                break
            print(f"{bits:>5} {vorher:>12.4f} {nachher:>13.4f} "
                  f"{math.exp(min(nachher,20)):>12.1f} "
                  f"{(nachher-vorher)/vorher*100:>15.1f} %")
        print("\nFaustregel: bis etwa +2 % Loss merkt man beim Reden nichts.\n"
              "Ab +10 % wird es deutlich schlechter.")

    if args.out:
        schreibe_datei(args.out, cfg, sd, args.bits,
                       "tokenizer/de_bpe_meta.json", ternaer_namen,
                       packung=args.packung)
        groesse = os.path.getsize(args.out)
        print(f"\nGeschrieben: {args.out} ({groesse/1e6:.2f} MB, {args.bits} Bit)")
        # Die Gewichte liegen im PSRAM, nur die Nachschlagetabelle im Flash -
        # aus ihr werden je Token nur wenige hundert Byte gelesen.
        tab = cfg.vocab_size * cfg.n_layer * cfg.d_flash * args.bits / 8 if cfg.d_flash else 0
        print(f"  davon PSRAM {(groesse-tab)/1e6:6.2f} MB   Flash {tab/1e6:5.2f} MB")
        for name, ps, fl in (("8 MB / 8 MB", 8e6, 8e6), ("32 MB / 16 MB", 32e6, 16e6)):
            passt = (groesse - tab) < ps * 0.75 and tab < fl * 0.75
            print(f"  Board {name:14} {'ja' if passt else 'NEIN'}"
                  f"   (25 % Reserve für Programm und Puffer)")

        # Probe: einen Tensor wieder auspacken und mit dem Original vergleichen.
        # Wenn das hier stimmt, kann die C-Engine dieselbe Rechnung machen.
        name, w = next((k, v) for k, v in sd.items()
                       if v.dim() >= 2 and k != "lm_head.weight")
        q, skalen, n = quantisiere(w, args.bits)
        zurueck = entpacke_bits(packe_bits(q, args.bits), n + (-n % BLOCK), args.bits)
        gleich = np.array_equal(zurueck, q.flatten().numpy().astype(np.int16))
        print(f"Rücklese-Probe an '{name}': "
              f"{'identisch' if gleich else 'FEHLER im Bitformat'}")


if __name__ == "__main__":
    main()
