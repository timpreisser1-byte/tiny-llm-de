"""Der Piper-Lehrer: deutscher VITS, aus dem der kleine Schueler lernt.

Nach sanoTTS (arXiv 2608.21378) wird das winzige Geraetemodell nicht direkt
auf Aufnahmen trainiert, sondern aus einem grossen Lehrer destilliert. Der
Lehrer ist hier `de_DE-thorsten-medium` aus dem Piper-Projekt - dieselbe
Stimme, deren Rohaufnahmen wir schon haben, nur als fertiges VITS-Modell.

Der Weg vom Text zum Ton:

    Text  --eSpeak-->  Phoneme  --Tabelle-->  IDs  --VITS-->  22,05 kHz

eSpeak liefert die Phoneme, die Piper erwartet; die Zuordnung zu Zahlen steht
in der mitgelieferten JSON. Zwischen je zwei Phonemen steht eine Null, und der
Satz wird von ^ und $ eingerahmt - so hat Piper es trainiert.

    python -m speech.teacher_voice "Guten Morgen, wie geht es dir?"
"""

import argparse
import json
import subprocess

import numpy as np


class Lehrer:
    def __init__(self, modell="lehrer/de_DE-thorsten-medium.onnx"):
        import onnxruntime
        self.sitzung = onnxruntime.InferenceSession(
            modell, providers=["CPUExecutionProvider"])
        c = json.load(open(modell + ".json", encoding="utf-8"))
        self.karte = c["phoneme_id_map"]
        self.rate = c["audio"]["sample_rate"]
        self.regler = c.get("inference", {})

    def phoneme(self, text):
        """eSpeak-Phoneme fuer deutschen Text, als einzelne Zeichen."""
        aus = subprocess.run(
            ["espeak-ng", "-v", "de", "-q", "--ipa=3", "--sep=_", text],
            capture_output=True, text=True).stdout
        return aus.replace("_", "").replace("\n", " ").strip()

    def ids(self, phoneme):
        raus = list(self.karte["^"])
        for z in phoneme:
            if z in self.karte:
                raus.extend(self.karte[z])
                raus.extend(self.karte["_"])       # Null zwischen den Phonemen
        raus.extend(self.karte["$"])
        return raus

    def sprich(self, text, laenge=1.0):
        p = self.phoneme(text)
        ids = np.array([self.ids(p)], dtype=np.int64)
        skalen = np.array([self.regler.get("noise_scale", 0.667),
                           laenge * self.regler.get("length_scale", 1.0),
                           self.regler.get("noise_w", 0.8)], dtype=np.float32)
        ton = self.sitzung.run(None, {
            "input": ids,
            "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
            "scales": skalen})[0]
        return ton.squeeze().astype(np.float32), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--out", default="lehrer/probe.wav")
    args = ap.parse_args()
    import soundfile as sf
    t = " ".join(args.text) or "Guten Morgen, wie geht es dir?"
    L = Lehrer()
    ton, p = L.sprich(t)
    sf.write(args.out, ton, L.rate)
    print(f"\n  {t}")
    print(f"  Phoneme: {p[:70]}")
    print(f"  {len(ton)/L.rate:.2f} s bei {L.rate} Hz -> {args.out}\n")


if __name__ == "__main__":
    main()
