"""Phase 1: Sprachdaten in Merkmale umwandeln.

Aus Multilingual LibriSpeech Deutsch (frei zugaenglich, keine Anmeldung) wird
ein kompaktes Merkmalsarchiv - der Gegenpart zu train.bin beim Sprachmodell.

    Opus-Audio  ->  16 kHz  ->  Mel-Spektrogramm (40 x T)  ->  float16-Datei
    Transkript  ->  Zeichenfolge  ->  Indexliste

Warum float16: Die Merkmale brauchen so nur halb so viel Platz, und die
Genauigkeit reicht voellig - der ESP32 rechnet spaeter ohnehin in Festkomma.

    python -m speech.asr_data --portion 1_hours
    python -m speech.asr_data --portion 9_hours
    python -m speech.asr_data --portion test
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import time
import urllib.request

import numpy as np

BASIS = ("https://huggingface.co/datasets/facebook/multilingual_librispeech"
         "/resolve/main/german")
PORTIONEN = {
    "1_hours": "1_hours-00000-of-00001.parquet",
    "9_hours": "9_hours-00000-of-00001.parquet",
    "dev": "dev-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
for i in range(63):                       # die grossen Trainingsdateien
    PORTIONEN[f"train{i}"] = f"train-{i:05d}-of-00063.parquet"

RATE = 16000
N_MEL, N_FFT, SPRUNG = 40, 512, 320       # identisch zum Kommandoerkenner
MAX_SEKUNDEN = 20.0                       # laengere Saetze sprengen den Speicher

# Das Alphabet. Index 0 bleibt fuer das CTC-Leerzeichen reserviert.
ZEICHEN = " abcdefghijklmnopqrstuvwxyzäöüß'-"
ZU_INDEX = {z: i + 1 for i, z in enumerate(ZEICHEN)}


def text_zu_indizes(text: str):
    text = text.lower().strip()
    return [ZU_INDEX[z] for z in text if z in ZU_INDEX]


def indizes_zu_text(indizes):
    return "".join(ZEICHEN[i - 1] for i in indizes if 0 < i <= len(ZEICHEN))


def mel_filter():
    """Dreiecksfilter auf der Mel-Skala - wie in speech/keyword_model.py."""
    def hz_zu_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_zu_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    punkte = mel_zu_hz(np.linspace(hz_zu_mel(20), hz_zu_mel(RATE / 2), N_MEL + 2))
    bins = np.floor((N_FFT + 1) * punkte / RATE).astype(int)
    f = np.zeros((N_MEL, N_FFT // 2 + 1), dtype=np.float32)
    for i in range(N_MEL):
        links, mitte, rechts = bins[i], bins[i + 1], bins[i + 2]
        for j in range(links, mitte):
            f[i, j] = (j - links) / max(mitte - links, 1)
        for j in range(mitte, rechts):
            f[i, j] = (rechts - j) / max(rechts - mitte, 1)
    return f


MEL = mel_filter()
FENSTER = np.hanning(N_FFT).astype(np.float32)


def merkmale(audio: np.ndarray) -> np.ndarray:
    """Audio -> log-Mel (T, 40). Reines numpy, damit es sich in C nachbauen laesst."""
    n = 1 + (len(audio) - N_FFT) // SPRUNG
    if n < 1:
        return np.zeros((0, N_MEL), dtype=np.float16)
    rahmen = np.lib.stride_tricks.as_strided(
        audio, shape=(n, N_FFT),
        strides=(audio.strides[0] * SPRUNG, audio.strides[0])) * FENSTER
    spektrum = np.abs(np.fft.rfft(rahmen, axis=1)) ** 2
    mel = spektrum @ MEL.T
    return np.log(mel + 1e-6).astype(np.float16)


def _eine_aufnahme(auftrag):
    """Eine Aufnahme dekodieren und in Merkmale wandeln.

    Laeuft in einem eigenen Prozess. Das Dekodieren der Opus-Dateien ist der
    Flaschenhals - auf zwei Kernen halbiert sich damit die Wartezeit, auf
    einem Mac mit acht Kernen wird es ein Vielfaches schneller.
    """
    import soundfile as sf

    rohdaten, text, dauer = auftrag
    if dauer > MAX_SEKUNDEN:
        return None
    try:
        audio, rate = sf.read(io.BytesIO(rohdaten), dtype="float32")
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != RATE:
        m = int(len(audio) * RATE / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, m),
                          np.arange(len(audio)), audio).astype(np.float32)
    ziele = text_zu_indizes(text)
    m = merkmale(np.ascontiguousarray(audio))
    # CTC braucht mehr Zeitschritte als Zielzeichen
    if len(m) // 2 <= len(ziele) + 2 or not ziele:
        return None
    return m, ziele, text.lower().strip(), dauer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--portion", default="1_hours", choices=list(PORTIONEN))
    ap.add_argument("--dir", default="data/asr")
    ap.add_argument("--prozesse", type=int, default=0,
                    help="0 = alle Kerne benutzen")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    os.makedirs(args.dir, exist_ok=True)
    quelle = os.path.join(args.dir, f"{args.portion}.parquet")
    if not os.path.exists(quelle):
        url = f"{BASIS}/{PORTIONEN[args.portion]}"
        print(f"Lade {url}")
        urllib.request.urlretrieve(url, quelle)
    print(f"{quelle}: {os.path.getsize(quelle)/1e6:.0f} MB")

    merkmal_datei = os.path.join(args.dir, f"{args.portion}.f16")
    index = []
    t0, position, sekunden, uebersprungen = time.time(), 0, 0.0, 0

    prozesse = args.prozesse or os.cpu_count() or 2
    print(f"{prozesse} Prozesse")

    with open(merkmal_datei, "wb") as aus, mp.Pool(prozesse) as pool:
        pf = pq.ParquetFile(quelle)
        for batch in pf.iter_batches(batch_size=200,
                                     columns=["audio", "transcript", "audio_duration"]):
            auftraege = [(a["bytes"], t, d) for a, t, d in
                         zip(batch.column("audio").to_pylist(),
                             batch.column("transcript").to_pylist(),
                             batch.column("audio_duration").to_pylist())]
            for ergebnis in pool.imap(_eine_aufnahme, auftraege, chunksize=8):
                if ergebnis is None:
                    uebersprungen += 1
                    continue
                m, ziele, text, dauer = ergebnis
                aus.write(m.tobytes())
                index.append({"pos": position, "rahmen": len(m),
                              "text": text, "ziele": ziele})
                position += m.size
                sekunden += dauer
            print(f"\r  {len(index):>6,} Aufnahmen | {sekunden/60:7.1f} min "
                  f"| {time.time()-t0:5.0f}s", end="", flush=True)
        print()

    with open(os.path.join(args.dir, f"{args.portion}.json"), "w", encoding="utf-8") as f:
        json.dump({"n_mel": N_MEL, "rate": RATE, "sprung": SPRUNG,
                   "zeichen": ZEICHEN, "aufnahmen": index}, f, ensure_ascii=False)

    print(f"\n{len(index):,} Aufnahmen, {sekunden/60:.1f} Minuten Sprache")
    print(f"  {uebersprungen} übersprungen (zu lang oder zu wenig Zeitschritte)")
    print(f"  {os.path.getsize(merkmal_datei)/1e6:.0f} MB Merkmale -> {merkmal_datei}")
    if index:
        b = index[0]
        print(f"\nBeispiel: {b['rahmen']} Zeitschritte, {len(b['ziele'])} Zeichen")
        print(f"  \"{b['text'][:90]}\"")


if __name__ == "__main__":
    main()
