"""Messprobe vor dem grossen Training: Wie schnell ist DIESE Maschine wirklich?

Bevor man Tage in ein Erkennertraining steckt, will man wissen, ob Grafikkarte
und Prozessor es in Tagen oder in Monaten schaffen. Dieses Skript misst beides
und rechnet auf die geplante Datenmenge hoch.

    1. Prozessor:  Opus dekodieren (falls soundfile es kann) + Mel-Spektrogramm
                   -- das macht beim Training der Datenlader.
    2. Grafikkarte: Ein Conformer-CTC-Netz (~14 Mio. Parameter, 25 Merkmals-
                   zeilen pro Sekunde) lernt einen Schritt auf Zufallsdaten.
                   Das Netz ist eine Naeherung an die Groesse, die auf den
                   ESP32 passen soll -- kein Modell, das man behalten wollte.

Langsamer von beiden bestimmt, wie lange eine Runde ueber alle Daten dauert.
Die Zahlen gelten fuer das Netz hier; ein anderes Netz kann abweichen.

    python -m speech.bench_train_speed
    python -m speech.bench_train_speed --stunden 2000 --praezision fp16 bf16
    python -m speech.bench_train_speed --nur-cpu --arbeiter 8

AMD unter Linux: ROCm-PyTorch meldet sich als "cuda". Die RX 6600 ist nicht
offiziell unterstuetzt, meist hilft  HSA_OVERRIDE_GFX_VERSION=10.3.0
"""

import argparse
import multiprocessing as mp
import os
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RATE = 16000
N_MEL, N_FFT, SPRUNG, FENSTER = 80, 512, 160, 400   # wie beim Kommandoerkenner, nur 80 Baender
CLIP_SEKUNDEN = 15.0
VOKABULAR = 1024                                     # Teilwoerter, Blank wird angehaengt


# --------------------------------------------------------------------------
# 1. Prozessor: Datenlader
# --------------------------------------------------------------------------

_STATE = {}


def _arbeiter_start(opus_datei):
    torch.set_num_threads(1)               # sonst streiten sich die Prozesse um die Kerne
    _STATE["fenster"] = torch.hann_window(FENSTER)
    # Zufallsfilter: kostet genauso viel wie ein echtes Mel-Filter, wir brauchen nur die Zeit
    _STATE["filter"] = torch.rand(N_FFT // 2 + 1, N_MEL)
    _STATE["opus"] = opus_datei
    _STATE["rauschen"] = torch.randn(int(RATE * CLIP_SEKUNDEN)) * 0.1


def _ein_clip(_):
    """Ein Clip: dekodieren (wenn moeglich), dann Mel-Spektrogramm."""
    if _STATE["opus"]:
        import soundfile as sf
        daten, _ = sf.read(_STATE["opus"], dtype="float32")
        ton = torch.from_numpy(daten)
    else:
        ton = _STATE["rauschen"]
    spek = torch.stft(ton, N_FFT, SPRUNG, FENSTER, _STATE["fenster"],
                      return_complex=True).abs() ** 2
    mel = torch.log(spek.T @ _STATE["filter"] + 1e-6)
    return mel.shape[0]


def _opus_testdatei():
    """Schreibt 15 s Rauschen als Opus, wenn libsndfile das kann. Sonst None."""
    try:
        import soundfile as sf
        if "OPUS" not in sf.available_formats():
            return None
        pfad = os.path.join(tempfile.mkdtemp(), "test.opus")
        rng = np.random.default_rng(0)
        sf.write(pfad, rng.normal(0, 0.1, int(RATE * CLIP_SEKUNDEN)).astype("float32"),
                 RATE, format="OGG", subtype="OPUS")
        return pfad
    except Exception:
        return None


def messe_prozessor(arbeiter, clips):
    opus = _opus_testdatei()
    ctx = mp.get_context("spawn")           # fork nach CUDA-Start ist verboten, spawn ist sicher
    with ctx.Pool(arbeiter, initializer=_arbeiter_start, initargs=(opus,)) as pool:
        pool.map(_ein_clip, range(arbeiter * 2))             # Aufwaermen (Import, Caches)
        t0 = time.time()
        pool.map(_ein_clip, range(clips))
        dauer = time.time() - t0
    rate = clips * CLIP_SEKUNDEN / dauer
    return rate, opus is not None


# --------------------------------------------------------------------------
# 2. Grafikkarte: kleines Conformer-CTC-Netz
# --------------------------------------------------------------------------

class Block(nn.Module):
    """Conformer-Block: halbes FF, Selbstaufmerksamkeit, Faltung, halbes FF."""

    def __init__(self, d, koepfe, ff, kern=15):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.SiLU(), nn.Linear(ff, d))
        self.ln_att = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, koepfe, batch_first=True)
        self.ln_conv = nn.LayerNorm(d)
        self.punkt1 = nn.Conv1d(d, 2 * d, 1)
        self.tief = nn.Conv1d(d, d, kern, padding=kern // 2, groups=d)
        self.norm = nn.BatchNorm1d(d)
        self.punkt2 = nn.Conv1d(d, d, 1)
        self.ff2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.SiLU(), nn.Linear(ff, d))
        self.ln_aus = nn.LayerNorm(d)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        h = self.ln_att(x)
        x = x + self.att(h, h, h, need_weights=False)[0]
        h = self.ln_conv(x).transpose(1, 2)
        h = F.glu(self.punkt1(h), dim=1)
        h = self.punkt2(F.silu(self.norm(self.tief(h))))
        x = x + h.transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return self.ln_aus(x)


class Erkenner(nn.Module):
    def __init__(self, d=256, schichten=10, koepfe=4, ff=1024):
        super().__init__()
        # zweimal Schrittweite 2: 100 Zeilen/s -> 25 Zeilen/s (40 ms)
        self.eingang = nn.Sequential(
            nn.Conv1d(N_MEL, d, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv1d(d, d, 3, stride=2, padding=1), nn.SiLU())
        self.bloecke = nn.ModuleList(Block(d, koepfe, ff) for _ in range(schichten))
        self.kopf = nn.Linear(d, VOKABULAR + 1)

    def forward(self, mel):                        # mel: (B, 80, T)
        x = self.eingang(mel).transpose(1, 2)
        for b in self.bloecke:
            x = b(x)
        return self.kopf(x)                        # (B, T/4, V+1)


def messe_grafikkarte(dev, praezision, batch_sek, schritte, clip_sek):
    torch.manual_seed(0)
    modell = Erkenner().to(dev)
    opt = torch.optim.AdamW(modell.parameters(), 1e-4)
    n_param = sum(p.numel() for p in modell.parameters())
    b = max(1, int(batch_sek // clip_sek))
    zeilen = int(clip_sek * 100)
    mel = torch.randn(b, N_MEL, zeilen, device=dev)
    ziel_len = int(clip_sek * 4)                   # ~4 Teilwoerter pro Sekunde
    ziel = torch.randint(0, VOKABULAR, (b, ziel_len), device=dev)
    ziel_l = torch.full((b,), ziel_len, dtype=torch.long)

    dtyp = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(praezision)
    skala = torch.amp.GradScaler(dev) if (praezision == "fp16" and dev == "cuda") else None

    def schritt():
        with torch.autocast(dev, dtype=dtyp, enabled=dtyp is not None):
            lp = modell(mel)
        lp = lp.float().log_softmax(-1).transpose(0, 1)       # CTC rechnet in float32
        ein_l = torch.full((b,), lp.shape[0], dtype=torch.long)
        verlust = F.ctc_loss(lp, ziel, ein_l, ziel_l, blank=VOKABULAR,
                             reduction="sum", zero_infinity=True) / ziel_l.sum()
        opt.zero_grad(set_to_none=True)
        if skala:
            skala.scale(verlust).backward()
            skala.step(opt)
            skala.update()
        else:
            verlust.backward()
            opt.step()

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()
        elif dev == "mps":
            torch.mps.synchronize()

    for _ in range(2):                                         # Aufwaermen
        schritt()
    sync()
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(schritte):
        schritt()
    sync()
    dauer = time.time() - t0
    gb = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else float("nan")
    return b * clip_sek * schritte / dauer, gb, n_param


# --------------------------------------------------------------------------

def stunden_text(sekunden):
    h = sekunden / 3600
    return f"{h:.1f} h" if h < 48 else f"{h / 24:.1f} Tage"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stunden", type=float, default=2000, help="Stunden Trainingsaudio je Runde")
    ap.add_argument("--arbeiter", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="Prozesse fuer den Datenlader")
    ap.add_argument("--clips", type=int, default=200, help="Clips fuer die Prozessor-Messung")
    ap.add_argument("--praezision", nargs="+", default=["fp32", "fp16"],
                    choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--batch-sek", type=float, nargs="+", default=[60, 120, 240],
                    help="Sekunden Audio pro Schritt; kleiner, wenn der Speicher knapp ist")
    ap.add_argument("--clip-sek", type=float, default=12.0)
    ap.add_argument("--schritte", type=int, default=5)
    ap.add_argument("--nur-cpu", action="store_true", help="Grafikkarte auslassen")
    args = ap.parse_args()

    ziel_sek = args.stunden * 3600
    print(f"\ntorch {torch.__version__}  HIP {getattr(torch.version, 'hip', None)}"
          f"  CUDA {torch.version.cuda}")

    # Prozessor zuerst: Der Pool wird vor dem GPU-Start gebildet (siehe messe_prozessor)
    print(f"\n[Prozessor]  {args.arbeiter} Arbeiter, {args.clips} Clips a {CLIP_SEKUNDEN:.0f} s")
    cpu_rate, mit_opus = messe_prozessor(args.arbeiter, args.clips)
    if mit_opus:
        hinweis = "mit Opus-Dekodierung"
    else:
        hinweis = "OHNE Opus-Dekodierung: soundfile kann kein Opus, echter Wert liegt niedriger"
    print(f"  {cpu_rate:7.0f} Audio-Sekunden pro Sekunde  ({hinweis})")
    print(f"  -> eine Runde ueber {args.stunden:.0f} h: {stunden_text(ziel_sek / cpu_rate)}")

    gpu_rate = None
    if not args.nur_cpu:
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
        name = torch.cuda.get_device_name(0) if dev == "cuda" else dev
        vram = (f", {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB"
                if dev == "cuda" else "")
        print(f"\n[Grafikkarte]  {name}{vram}  (Geraet: {dev})")
        if dev == "cpu":
            print("  Keine GPU gefunden -- Messung auf dem Prozessor, nur zum Ausprobieren.")
        if dev == "mps":
            print("  Hinweis: ctc_loss laeuft auf MPS ggf. nur ueber CPU-Rueckfall "
                  "(PYTORCH_ENABLE_MPS_FALLBACK=1).")
        beste = 0.0
        for praez in args.praezision:
            for bs in args.batch_sek:
                try:
                    rate, gb, n = messe_grafikkarte(dev, praez, bs, args.schritte, args.clip_sek)
                except torch.cuda.OutOfMemoryError:
                    print(f"  {praez:5} {bs:5.0f} s/Schritt   Speicher voll (OOM)")
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:                       # z.B. bf16 nicht unterstuetzt
                    print(f"  {praez:5} {bs:5.0f} s/Schritt   Fehler: {type(e).__name__}: "
                          f"{str(e)[:80]}")
                    continue
                speicher = f"{gb:.1f} GB" if gb == gb else "   - "
                print(f"  {praez:5} {bs:5.0f} s/Schritt  {rate:7.0f} Audio-s/s  "
                      f"Speicher {speicher}   ({n / 1e6:.1f} Mio. Parameter)")
                beste = max(beste, rate)
        gpu_rate = beste or None
        if gpu_rate:
            print(f"  -> eine Runde ueber {args.stunden:.0f} h: {stunden_text(ziel_sek / gpu_rate)}")

    print("\n[Ergebnis]")
    raten = {"Prozessor": cpu_rate}
    if gpu_rate:
        raten["Grafikkarte"] = gpu_rate
    engpass = min(raten, key=raten.get)
    runde = ziel_sek / raten[engpass]
    print(f"  Engpass: {engpass}.  Eine Runde ueber {args.stunden:.0f} h dauert etwa "
          f"{stunden_text(runde)};")
    print(f"  20 Runden also etwa {stunden_text(20 * runde)}.")
    print("  (Zufallsdaten und ein Ersatznetz: Richtwert, kein Versprechen.)\n")


if __name__ == "__main__":
    main()
