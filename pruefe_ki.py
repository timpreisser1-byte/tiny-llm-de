"""Testbatterie: die KI systematisch abfragen und die Schwachstellen finden.

Statt zufaellig herumzuprobieren wird hier eine feste Liste von Fragen gestellt,
nach Bereichen sortiert. Fragen mit erwartetem Schluesselwort werden automatisch
bewertet, der Rest zum Nachlesen ausgegeben.

    python pruefe_ki.py --ckpt ckpt_colab/sft.pt
    python pruefe_ki.py --ckpt ckpt_colab/sft.pt --bereich smalltalk
    python pruefe_ki.py --ckpt ckpt_colab/sft.pt --schreibe fehler.txt

Der Ablauf danach: Fehler anschauen, passende Beispiele in
data/sft/*.jsonl ergaenzen, neu finetunen, wieder pruefen.
"""

import argparse
import json
import re

import torch
from tokenizers import Tokenizer

from chat_format import build_prompt, mit_wissen
from model import Config, TinyLM
from train import pick_device

GRAU, GRUEN, ROT, GELB, FETT, AUS = (
    "\033[90m", "\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[0m")

# (Frage, erwartetes Schluesselwort oder None, Wissen oder None)
BATTERIE = {
    "smalltalk": [
        ("hi", None, None), ("Hi", None, None), ("hallo", None, None),
        ("Hallo!", None, None), ("hey", None, None), ("Moin", None, None),
        ("Guten Morgen", None, None), ("Wie geht es dir?", None, None),
        ("Was machst du gerade?", None, None), ("Danke!", None, None),
        ("Tschüss", None, None), ("Wer bist du?", None, None),
        ("Was kannst du?", None, None), ("Bist du ein Mensch?", None, None),
    ],
    "elektronik": [
        ("Was ist ein Widerstand?", "Ohm", None),
        ("Was ist ein Kondensator?", "Ladung", None),
        ("Was macht ein Transistor?", "verstärk|schalt", None),
        ("Erkläre das Ohmsche Gesetz.", "U|Spannung", None),
        ("Was ist eine LED?", "Leucht|leucht", None),
        ("Wofür braucht eine LED einen Vorwiderstand?", "Strom", None),
        ("Was ist ein Mikrocontroller?", "Chip|Computer", None),
        ("Was ist I2C?", "SDA|Bus|Leitung", None),
        ("Wie funktioniert ein Lautsprecher?", "Membran|Spule", None),
        ("Was ist Volt?", "Spannung", None),
    ],
    "ratgeber": [
        ("Ich kann abends nicht einschlafen.", "Bildschirm|kühl|Zeit", None),
        ("Ich habe schlecht geschlafen, was hilft?", None, None),
        ("Ich bin gestresst.", None, None),
        ("Wie bleibe ich motiviert?", None, None),
        ("Ich kann mich nicht konzentrieren.", "Pause|Handy|Minuten", None),
        ("Wie gewöhne ich mir eine Gewohnheit an?", None, None),
    ],
    "wissen": [
        ("Wie viel PSRAM hat der ESP32-P4?", "32",
         "Der ESP32-P4 ist ein Mikrocontroller von Espressif mit zwei RISC-V-Kernen "
         "und bis zu 32 MB PSRAM."),
        ("Wo liegt Quedlinburg?", "Sachsen-Anhalt",
         "Die Stadt Quedlinburg liegt in Sachsen-Anhalt und hat rund 24.000 Einwohner."),
        ("Wann wurde das Museum gegründet?", "1892",
         "1892 wurde das Bomann-Museum mit volkskundlichen Sammlungen gegründet."),
        ("Wie hoch ist der Turm?", "74",
         "1913 wurde der 74 m hohe Glockenturm der Stadtkirche errichtet."),
        ("Welche Farbe hat das Gehäuse?", "blau",
         "Das Gehäuse des Messgeräts ist blau und besteht aus Kunststoff."),
        # Die Faehigkeit, einen Wert aus dem mitgelieferten Text herauszugreifen,
        # ist der Kern des Geraetekonzepts - Wissen kommt von der SD-Karte, nicht
        # aus den Gewichten. Fuenf Fragen waren zu wenig, um sie zu messen:
        # zwischen zwei Modellen lagen 6 und 12 Treffer bei je 15 Versuchen.
        ("Wie viele Einwohner hat der Ort?", "8.400",
         "Das Dorf Reinsdorf hat 8.400 Einwohner und liegt an der Unstrut."),
        ("Wie schwer ist das Bauteil?", "45",
         "Der Sensor wiegt 45 Gramm und misst 30 mal 20 Millimeter."),
        ("In welchem Jahr wurde die Brücke gebaut?", "1954",
         "Die Brücke über den Fluss wurde 1954 fertiggestellt und 1998 saniert."),
        ("Wie lang ist der Tunnel?", "1.200",
         "Der Tunnel ist 1.200 Meter lang und wurde in zwei Jahren gegraben."),
        ("Welche Spannung braucht das Gerät?", "12",
         "Das Steuergerät arbeitet mit 12 Volt Gleichspannung und 2 Ampere."),
        ("Wer hat das Buch geschrieben?", "Kellermann",
         "Das Buch über die Harzer Schmalspurbahn stammt von Anna Kellermann."),
        ("Wie viele Stockwerke hat das Haus?", "sieben",
         "Das Wohnhaus am Markt hat sieben Stockwerke und einen Aufzug."),
        ("Wann öffnet das Geschäft?", "9",
         "Der Laden öffnet um 9 Uhr und schließt um 18 Uhr."),
        ("Wie tief ist der See?", "38",
         "Der Bergsee ist an der tiefsten Stelle 38 Meter tief."),
        ("Welches Material wurde verwendet?", "Aluminium",
         "Das Gehäuse besteht aus Aluminium und wiegt 300 Gramm."),
    ],
    "grenzen": [
        ("Wie wird das Wetter morgen?", "weiß|keine|offline", None),
        ("Wie spät ist es?", "Uhr|weiß|keine", None),
        ("Wer gewinnt die Wahl 2027?", "weiß|keine", None),
        ("Was ist 384 mal 27?", None, None),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_colab/sft.pt")
    ap.add_argument("--tokenizer", default="tokenizer/de_bpe.json")
    ap.add_argument("--bereich", default=None, choices=list(BATTERIE))
    ap.add_argument("--temperatur", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--laenge", type=int, default=60)
    ap.add_argument("--schreibe", default=None, help="Fehler als Datei sichern")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = Tokenizer.from_file(args.tokenizer)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    netz = TinyLM(Config(**ck["config"]))
    # ueber "art" statt ueber das alte Flag "ternaer": sonst wird ein
    # Sherry-Modell ohne Beschraenkung geladen, laeuft mit Schattengewichten
    # und faellt von 28/30 auf 1/30 - gemessen am 30.08.
    art = ck.get("art") or ("ternaer" if ck.get("ternaer") else None)
    if art:
        from ternaer import beschraenke
        beschraenke(netz, art)
    netz.load_state_dict(ck["model"])
    netz.to(device).eval()
    eos = tok.token_to_id("</s>")

    print(f"\n{FETT}{args.ckpt}{AUS}  {netz.num_params():,} Parameter, "
          f"val {ck.get('val', float('nan')):.4f}")
    print(f"{GRAU}Temperatur {args.temperatur} (bei Wissensfragen 0), "
          f"Top-K {args.top_k}{AUS}")

    bereiche = [args.bereich] if args.bereich else list(BATTERIE)
    fehler, gesamt_ok, gesamt_bewertet = [], 0, 0

    for bereich in bereiche:
        print(f"\n{FETT}── {bereich} {'─' * (66 - len(bereich))}{AUS}")
        for frage, erwartet, wissen in BATTERIE[bereich]:
            text = mit_wissen(frage, wissen) if wissen else frage
            ids = tok.encode(build_prompt([{"role": "user", "content": text}]),
                             add_special_tokens=False).ids
            # bei Wissensfragen zaehlt Genauigkeit, nicht Abwechslung
            temp = 0.0 if wissen else args.temperatur
            raus = [t for t in netz.generate(
                torch.tensor([ids], device=device), max_new_tokens=args.laenge,
                temperature=temp, top_k=args.top_k, eos_id=eos) if t != eos]
            antwort = tok.decode(raus).strip()

            marke = " "
            if erwartet:
                gesamt_bewertet += 1
                if re.search(erwartet, antwort, re.I):
                    marke = f"{GRUEN}✓{AUS}"; gesamt_ok += 1
                else:
                    marke = f"{ROT}✗{AUS}"
                    fehler.append((bereich, frage, antwort, erwartet))
            print(f" {marke} {frage[:42]:44} {antwort[:80]}")

    print(f"\n{FETT}{'─'*72}{AUS}")
    if gesamt_bewertet:
        print(f"Automatisch bewertbar: {gesamt_ok}/{gesamt_bewertet} richtig "
              f"({gesamt_ok/gesamt_bewertet*100:.0f} %)")
    if fehler:
        print(f"\n{GELB}Diese Fragen brauchen neue Trainingsbeispiele:{AUS}")
        for bereich, frage, antwort, erwartet in fehler:
            print(f"  [{bereich}] {frage}")
            print(f"     bekam:    {antwort[:70]}")
            print(f"     erwartet: etwas mit '{erwartet}'")
    if args.schreibe:
        with open(args.schreibe, "w", encoding="utf-8") as f:
            json.dump([{"bereich": b, "frage": f_, "antwort": a, "erwartet": e}
                       for b, f_, a, e in fehler], f, ensure_ascii=False, indent=2)
        print(f"\nFehler gesichert in {args.schreibe}")


if __name__ == "__main__":
    main()
