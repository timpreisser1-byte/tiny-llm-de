"""Deutschen Webtext von FineWeb2-HQ holen - die Datengrundlage fuer grosse Laeufe.

Der bisherige Korpus hat 64 Mio. Token. Fuer ein Modell mit 50 Mio. Parametern
ist das viel zu wenig: Als Faustregel gelten rund 20 Token je Parameter, also
eine Milliarde. Mit 64 Mio. wuerde das Modell die Daten auswendig lernen statt
Sprache.

FineWeb2-HQ ist die von EPFL nachgefilterte Fassung von FineWeb2 - deutscher
Webtext, bereits nach Qualitaet vorsortiert. Ein Shard bringt rund 120 Mio.
Token. Hier wird deshalb nur noch nach Sprachsicherheit und Mindestlaenge
gesiebt; eine zweite Qualitaetsschwelle wuerde zwei Drittel wegwerfen, die
bereits als gut eingestuft wurden.

    python -m training.fineweb_data --shards 3 --out data/corpus_web.txt
"""

import argparse
import os
import time
import urllib.request

REPO = "https://huggingface.co/datasets/epfml/FineWeb2-HQ/resolve/main/deu_Latn"
SHARD = "000_{i:05d}.parquet"
SPALTEN = ["text", "language_score", "quality_score"]


def lade(i: int, ordner: str) -> str:
    ziel = os.path.join(ordner, f"de_{i:03d}.parquet")
    if os.path.exists(ziel) and os.path.getsize(ziel) > 1_000_000_000:
        print(f"  schon da: {ziel} ({os.path.getsize(ziel)/1e9:.2f} GB)")
        return ziel
    url = f"{REPO}/{SHARD.format(i=i)}"
    print(f"  lade {url}")
    t0 = [time.time()]

    def fortschritt(block, groesse, gesamt):
        if gesamt <= 0 or block % 400:
            return
        fertig = block * groesse
        dauer = time.time() - t0[0]
        print(f"\r    {fertig/1e9:5.2f} / {gesamt/1e9:.2f} GB  "
              f"({fertig/max(dauer,1e-6)/1e6:.0f} MB/s)", end="", flush=True)

    tmp = ziel + ".teil"
    urllib.request.urlretrieve(url, tmp, fortschritt)
    os.replace(tmp, ziel)
    print(f"\r    fertig: {os.path.getsize(ziel)/1e9:.2f} GB{' '*20}")
    return ziel


def schreibe(dateien, out, min_zeichen, sprachsicherheit, guete=0.0):
    """Nur die Textspalte lesen - die Einbettungen machen 80 % der Datei aus."""
    import pyarrow.parquet as pq

    behalten = verworfen = zeichen = 0
    with open(out, "w", encoding="utf-8") as f:
        for pfad in dateien:
            datei = pq.ParquetFile(pfad)
            for g in range(datei.num_row_groups):
                tabelle = datei.read_row_group(g, columns=SPALTEN)
                texte = tabelle["text"].to_pylist()
                werte = tabelle["language_score"].to_pylist()
                guete_werte = tabelle["quality_score"].to_pylist()
                for text, sicher, g in zip(texte, werte, guete_werte):
                    # FineWeb2-HQ ist bereits vorgefiltert; die zweite Schwelle
                    # nimmt gemessen bei 0,1 noch die schlechtesten 30 % heraus
                    # und behaelt 71 % des Textes.
                    if (sicher is None or sicher < sprachsicherheit
                            or len(text) < min_zeichen
                            or (g is not None and g < guete)):
                        verworfen += 1
                        continue
                    f.write(text.strip() + "\n\n")
                    behalten += 1
                    zeichen += len(text)
                if g % 20 == 0:
                    print(f"\r  {os.path.basename(pfad)}  Gruppe {g+1}/{datei.num_row_groups}"
                          f"  {zeichen/1e6:.0f} Mio. Zeichen", end="", flush=True)
            print()
    return behalten, verworfen, zeichen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--ab", type=int, default=1, help="mit welchem Shard beginnen")
    ap.add_argument("--cache-dir", default="data/fineweb")
    ap.add_argument("--out", default="data/corpus_web.txt")
    ap.add_argument("--min-zeichen", type=int, default=400)
    ap.add_argument("--sprachsicherheit", type=float, default=0.95)
    ap.add_argument("--guete", type=float, default=0.0,
                    help="Mindest-quality_score; 0,1 behält 71 %% des Textes")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    dateien = [lade(i, args.cache_dir) for i in range(args.ab, args.ab + args.shards)]

    t0 = time.time()
    behalten, verworfen, zeichen = schreibe(dateien, args.out, args.min_zeichen,
                                            args.sprachsicherheit, args.guete)
    print(f"\n{behalten:,} Dokumente behalten, {verworfen:,} verworfen")
    print(f"{zeichen/1e6:.0f} Mio. Zeichen -> {args.out} "
          f"({os.path.getsize(args.out)/1e9:.2f} GB) in {(time.time()-t0)/60:.1f} min")
    print(f"ergibt grob {zeichen/3.4/1e6:.0f} Mio. Token")


if __name__ == "__main__":
    main()
