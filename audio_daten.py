"""Trainingsdaten fuer die Spracherkennung selbst erzeugen.

Warum nicht Common Voice herunterladen: Fuer den ESP32-P4 kommt keine freie
Diktaterkennung in Frage (Whisper braucht im kleinsten Modell 42 MB, der Chip
hat 32 MB PSRAM). Realistisch ist ein Erkenner fuer feste Kommandowoerter -
und dafuer sind selbst erzeugte Daten sogar im Vorteil: Wir bestimmen genau
die Woerter, die das Geraet koennen soll.

Die Sprecher kommen von deinem Mac: acht deutsche Systemstimmen plus die
Stimmvarianten von eSpeak. Aus jedem Wort werden durch Variation von
Sprechgeschwindigkeit, Lautstaerke, zeitlicher Verschiebung und Rauschen
Hunderte Beispiele.

Wichtig zu wissen: Synthetische Stimmen decken echte Mikrofonaufnahmen nur
teilweise ab. Deshalb kann man mit `--eigene` spaeter eigene Aufnahmen
dazulegen - die zaehlen beim Training doppelt.

    python audio_daten.py --woerter woerter.txt
    python audio_daten.py --zeige
"""

import argparse
import json
import os
import random
import subprocess
import sys

import numpy as np
import soundfile as sf

RATE = 16000          # Abtastrate, wie sie der ESP32 vom Mikrofon bekommt
DAUER = 1.0           # Sekunden pro Beispiel - feste Laenge macht das Modell klein

# Kommandowoerter fuer den Anfang. Die letzte Klasse faengt alles ab,
# was kein Kommando ist - ohne sie wuerde das Modell jedes Geraeusch
# einem Kommando zuordnen.
STANDARD_WOERTER = [
    "licht an", "licht aus", "wie spät", "temperatur", "musik an",
    "musik aus", "lauter", "leiser", "stopp", "weiter",
]
FUELLWOERTER = [
    "guten morgen", "das Wetter", "einundzwanzig", "wo ist das", "vielleicht",
    "ich gehe jetzt", "computer", "keine Ahnung", "morgen früh", "warte kurz",
    "sehr gut", "was machst du", "die Straße", "hallo zusammen", "kein Problem",
]


def mac_stimmen():
    """Deutsche Systemstimmen des Macs."""
    try:
        aus = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return []
    return [z.split("  ")[0].strip() for z in aus.splitlines() if "de_DE" in z]


def sprich_mac(text, stimme, tempo, ziel):
    aiff = ziel + ".aiff"
    subprocess.run(["say", "-v", stimme, "-r", str(tempo), "-o", aiff, text],
                   check=True, capture_output=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{RATE}", "-c", "1",
                    aiff, ziel], check=True, capture_output=True)
    os.unlink(aiff)


def sprich_espeak(text, variante, tempo, ziel):
    subprocess.run(["espeak-ng", "-v", f"de+{variante}", "-s", str(tempo),
                    "-w", ziel, text], check=True, capture_output=True)


def lade(pfad):
    daten, rate = sf.read(pfad, dtype="float32")
    if daten.ndim > 1:
        daten = daten.mean(axis=1)
    if rate != RATE:                      # eSpeak liefert 22050 Hz
        n = int(len(daten) * RATE / rate)
        daten = np.interp(np.linspace(0, len(daten) - 1, n),
                          np.arange(len(daten)), daten).astype(np.float32)
    return daten


def auf_laenge(daten, versatz=0.0):
    """Auf genau DAUER Sekunden bringen, mit zufaelliger Verschiebung."""
    n = int(RATE * DAUER)
    raus = np.zeros(n, dtype=np.float32)
    start = int(versatz * RATE)
    stueck = daten[:n - start] if start >= 0 else daten[-start:n]
    start = max(start, 0)
    raus[start:start + len(stueck)] = stueck[:n - start]
    return raus


def variiere(daten, rng):
    """Aus einer Aufnahme viele machen: Lautstaerke, Versatz, Rauschen, Tonhöhe."""
    d = auf_laenge(daten, versatz=rng.uniform(-0.15, 0.25))
    d = d * rng.uniform(0.3, 1.0)                       # Abstand zum Mikrofon
    if rng.random() < 0.8:                              # Umgebungsgeraeusch
        d = d + rng.normal(0, rng.uniform(0.001, 0.02), len(d)).astype(np.float32)
    if rng.random() < 0.3:                              # dumpfer Klang
        kern = np.ones(rng.integers(2, 5)) / 3
        d = np.convolve(d, kern, mode="same").astype(np.float32)
    spitze = np.abs(d).max()
    if spitze > 0.99:
        d = d / spitze * 0.99
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ziel", default="audio")
    ap.add_argument("--pro-aufnahme", type=int, default=12,
                    help="wie viele Varianten je Stimme und Wort")
    ap.add_argument("--zeige", action="store_true", help="nur Stimmen auflisten")
    args = ap.parse_args()

    stimmen = mac_stimmen()
    varianten = ["m1", "m3", "m5", "f2", "f4"]          # eSpeak-Stimmfarben
    if args.zeige:
        print("macOS-Stimmen:", ", ".join(stimmen) or "keine")
        print("eSpeak-Varianten:", ", ".join(varianten))
        return
    if not stimmen:
        sys.exit("Keine deutschen Systemstimmen gefunden.")

    os.makedirs(args.ziel, exist_ok=True)
    roh = os.path.join(args.ziel, "roh")
    os.makedirs(roh, exist_ok=True)
    rng = np.random.default_rng(0)

    klassen = STANDARD_WOERTER + ["_unbekannt"]
    index = []
    print(f"{len(stimmen)} Systemstimmen + {len(varianten)} eSpeak-Varianten\n")

    for klasse in klassen:
        texte = FUELLWOERTER if klasse == "_unbekannt" else [klasse]
        ordner = os.path.join(args.ziel, klasse.replace(" ", "_"))
        os.makedirs(ordner, exist_ok=True)
        n = 0
        for text in texte:
            for stimme in stimmen:
                for tempo in (160, 190, 220):
                    tmp = os.path.join(roh, "tmp.wav")
                    try:
                        sprich_mac(text, stimme, tempo, tmp)
                    except subprocess.CalledProcessError:
                        continue
                    grund = lade(tmp)
                    for i in range(args.pro_aufnahme):
                        datei = os.path.join(ordner, f"{n:05d}.wav")
                        sf.write(datei, variiere(grund, rng), RATE)
                        index.append({"datei": datei, "klasse": klasse,
                                      "quelle": f"mac:{stimme}:{tempo}"})
                        n += 1
            for variante in varianten:
                for tempo in (140, 175):
                    tmp = os.path.join(roh, "tmp.wav")
                    try:
                        sprich_espeak(text, variante, tempo, tmp)
                    except subprocess.CalledProcessError:
                        continue
                    grund = lade(tmp)
                    for i in range(args.pro_aufnahme):
                        datei = os.path.join(ordner, f"{n:05d}.wav")
                        sf.write(datei, variiere(grund, rng), RATE)
                        index.append({"datei": datei, "klasse": klasse,
                                      "quelle": f"espeak:{variante}:{tempo}"})
                        n += 1
        print(f"  {klasse:14} {n:>6,} Beispiele")

    with open(os.path.join(args.ziel, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"rate": RATE, "dauer": DAUER, "klassen": klassen,
                   "beispiele": index}, f, ensure_ascii=False)
    print(f"\n{len(index):,} Beispiele -> {args.ziel}/index.json")
    print(f"{len(index)*RATE*DAUER*4/1e6:.0f} MB Audio, {len(klassen)} Klassen")


if __name__ == "__main__":
    main()
