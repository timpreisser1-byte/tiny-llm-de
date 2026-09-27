"""Hundert Fragen quer durch alles - der breite Test.

Die Prüfung in evaluation/battery.py hat 49 Fragen und ist auf die Bereiche zugeschnitten,
die wir gezielt trainiert haben. Sie sagt deshalb wenig darüber, wie sich das
Modell bei Fragen verhält, die es so nie gesehen hat.

Hier stehen hundert Fragen, davon der grösste Teil ungeübt: Alltag, Sprache,
Rechnen, Meinungen, Unsinn. Automatisch bewertbar ist nur ein Teil - der Rest
wird ausgegeben, damit man selbst urteilen kann.

    python -m evaluation.broad100 --ckpt ckpt_56m/sft_v2.pt
    python -m evaluation.broad100 --ckpt ckpt_56m/sft_v2.pt --nur wissen
"""

import argparse
import random

# (Frage, erwartet oder None, Kontext oder None)
FRAGEN = {
 "gespräch": [
    ("Erzähl mir was über dich.", None, None),
    ("Was ist dein Lieblingsessen?", None, None),
    ("Hast du Gefühle?", None, None),
    ("Kannst du Witze?", None, None),
    ("Wie alt bist du?", None, None),
    ("Wo wohnst du?", None, None),
    ("Bist du schlau?", None, None),
    ("Was hältst du von Menschen?", None, None),
    ("Magst du Musik?", None, None),
    ("Was war gestern?", None, None),
    ("Ich bin traurig.", None, None),
    ("Mir ist langweilig.", None, None),
    ("Ich habe heute Geburtstag.", None, None),
    ("Kannst du mir helfen?", None, None),
    ("Sprichst du Englisch?", None, None),
    ("Wie heißt du?", None, None),
    ("Gute Nacht.", None, None),
    ("Bis morgen!", None, None),
    ("Was soll ich kochen?", None, None),
    ("Erzähl einen Witz.", None, None),
 ],
 "sprache": [
    ("Was ist der Plural von Haus?", "Häuser", None),
    ("Was ist das Gegenteil von warm?", "kalt", None),
    ("Wie schreibt man 'Fahrrad'?", "Fahrrad", None),
    ("Nenne drei Farben.", None, None),
    ("Was reimt sich auf Baum?", None, None),
    ("Was bedeutet 'schnell'?", None, None),
    ("Ist 'gehen' ein Verb?", "Verb|ja", None),
    ("Was ist ein Substantiv?", None, None),
    ("Übersetze 'Haus' ins Englische.", "house", None),
    ("Wie viele Buchstaben hat das Alphabet?", "26", None),
    ("Nenne vier Wochentage.", None, None),
    ("Was ist die Mehrzahl von Kind?", "Kinder", None),
    ("Vervollständige: Alle guten Dinge sind ...", "drei", None),
    ("Was ist ein Adjektiv?", None, None),
    ("Was bedeutet 'trotzdem'?", None, None),
 ],
 "wissen_ohne": [
    ("Was ist die Hauptstadt von Deutschland?", "Berlin", None),
    ("Wie viele Bundesländer hat Deutschland?", "16|sechzehn", None),
    ("Welche Farbe hat der Himmel?", "blau", None),
    ("Wie viele Beine hat eine Spinne?", "8|acht", None),
    ("In welchem Erdteil liegt Deutschland?", "Europa", None),
    ("Was ist Wasser chemisch?", "H2O|H₂O|Wasserstoff", None),
    ("Wie viele Tage hat ein Jahr?", "365", None),
    ("Welcher Planet ist der Erde am nächsten?", None, None),
    ("Was ist ein Säugetier?", None, None),
    ("Wie heißt der längste Fluss Deutschlands?", None, None),
 ],
 "elektronik": [
    ("Was ist Strom?", None, None),
    ("Was macht ein Schalter?", "unterbricht|schließt|öffnet", None),
    ("Was ist ein Kurzschluss?", None, None),
    ("Wofür ist eine Sicherung da?", None, None),
    ("Was ist Wechselstrom?", None, None),
    ("Was macht ein Trafo?", "Spannung", None),
    ("Was ist ein Sensor?", "misst", None),
    ("Wie funktioniert ein Relais?", None, None),
    ("Was ist SPI?", "Bus|Schnittstelle", None),
    ("Was ist eine Diode?", "Richtung|Strom", None),
    ("Was ist PWM?", None, None),
    ("Wofür braucht man einen Kondensator?", None, None),
    ("Was ist Erdung?", None, None),
    ("Was ist ein Widerstand in Reihe?", None, None),
    ("Was ist Gleichstrom?", None, None),
 ],
 "wissen": [
    ("Wie viele Zimmer hat die Wohnung?", "drei",
     "Die Wohnung im dritten Stock hat drei Zimmer und einen Balkon."),
    ("Wann beginnt der Kurs?", "18",
     "Der Kurs beginnt um 18 Uhr im Raum 204 und dauert zwei Stunden."),
    ("Wie teuer ist das Gerät?", "249",
     "Das Messgerät kostet 249 Euro und hat zwei Jahre Garantie."),
    ("Aus welchem Jahr stammt das Bild?", "1887",
     "Das Gemälde aus dem Jahr 1887 zeigt eine Landschaft im Herbst."),
    ("Wie weit ist es bis zum Bahnhof?", "800",
     "Vom Hotel sind es 800 Meter bis zum Bahnhof, etwa zehn Minuten zu Fuß."),
    ("Welche Sprache spricht man dort?", "Portugiesisch",
     "In Brasilien spricht man Portugiesisch, anders als im übrigen Südamerika."),
    ("Wie heißt der Erfinder?", "Diesel",
     "Der Dieselmotor wurde von Rudolf Diesel entwickelt und 1893 patentiert."),
    ("Wie hoch ist die Gebühr?", "12",
     "Für die Anmeldung wird eine Gebühr von 12 Euro erhoben."),
    ("Wie viele Teilnehmer waren da?", "340",
     "An der Veranstaltung nahmen 340 Personen teil, mehr als im Vorjahr."),
    ("Welches Betriebssystem läuft darauf?", "Linux",
     "Der Server läuft unter Linux und wird zweimal jährlich gewartet."),
    ("Wann wurde die Firma gegründet?", "1962",
     "Die Firma wurde 1962 in Hamburg gegründet und hat 200 Mitarbeiter."),
    ("Wie schnell fährt der Zug?", "160",
     "Der Regionalzug fährt maximal 160 Stundenkilometer."),
    ("Wie viel wiegt das Paket?", "2,4",
     "Das Paket wiegt 2,4 Kilogramm und misst 40 mal 30 Zentimeter."),
    ("In welcher Stadt steht das Gebäude?", "Leipzig",
     "Das Rathaus steht in Leipzig am Markt und wurde 1905 fertiggestellt."),
    ("Wie lange dauert die Garantie?", "24",
     "Auf das Gerät gibt der Hersteller 24 Monate Garantie."),
 ],
 "grenzen": [
    ("Was denke ich gerade?", "weiß|kann|nicht", None),
    ("Wie viel Geld habe ich auf dem Konto?", "weiß|kann|nicht", None),
    ("Was passiert morgen?", "weiß|kann|nicht", None),
    ("Wie ist mein Name?", "weiß|kann|nicht", None),
    ("Ruf meine Mutter an.", "kann|nicht", None),
    ("Schalte das Licht aus.", None, None),
    ("Wer hat die Wahl 2030 gewonnen?", "weiß|kann|nicht", None),
    ("Wie geht es meinem Hund?", "weiß|kann|nicht", None),
    ("Bestelle mir eine Pizza.", "kann|nicht", None),
    ("Was steht in meiner E-Mail?", "weiß|kann|nicht", None),
 ],
 "rechnen": [
    ("Was ist 2 plus 2?", "4|vier", None),
    ("Was ist 10 mal 10?", "100|hundert", None),
    ("Was ist die Hälfte von 50?", "25", None),
    ("Was ist 7 minus 3?", "4|vier", None),
    ("Wie viel ist ein Drittel von 9?", "3|drei", None),
 ],
 "unsinn": [
    ("asdfgh", None, None),
    ("?????", None, None),
    ("Warum ist die Banane krumm?", None, None),
    ("Wie viele Wolken passen in einen Eimer?", None, None),
    ("Erkläre mir Quantenphysik in einem Wort.", None, None),
    ("Kannst du fliegen?", None, None),
    ("Bist du ein Toaster?", None, None),
    ("Sing ein Lied.", None, None),
    ("Male mir ein Bild.", None, None),
    ("42", None, None),
 ],
}


def main():
    import torch
    from lm.generate import load, build_prompt
    from lm.chat_format import mit_wissen

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_56m/sft_v2.pt")
    ap.add_argument("--tokenizer", default="tokenizer/de_bpe.json")
    ap.add_argument("--nur", default=None)
    ap.add_argument("--laenge", type=int, default=48)
    ap.add_argument("--temperatur", type=float, default=0.4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from training.pretrain import pick_device
    model, tok = load(args.ckpt, args.tokenizer, pick_device(args.device))
    bereiche = [args.nur] if args.nur else list(FRAGEN)

    treffer = pruefbar = gesamt = 0
    for bereich in bereiche:
        print(f"\n\033[1m── {bereich} {'─'*(58-len(bereich))}\033[0m")
        for frage, erwartet, wissen in FRAGEN[bereich]:
            # Format zeichengenau wie im Training - "Nutze diesen Text:"
            # stand nie in den Daten, und das Modell lehnt dann ab, statt
            # nachzusehen. Gemessen: 3/15 mit der falschen Form, 13/15 mit
            # dieser.
            text = mit_wissen(frage, wissen) if wissen else frage
            ids = tok.encode(build_prompt([{"role": "user", "content": text}]),
                             add_special_tokens=False).ids
            x = torch.tensor([ids], device=model.lm_head.weight.device)
            ende = tok.token_to_id("</s>")
            raus = []
            with torch.no_grad():
                # generate liefert Token einzeln, nicht als Block
                for t in model.generate(x, max_new_tokens=args.laenge,
                                        temperature=0.0 if wissen else args.temperatur,
                                        top_k=20, repetition_penalty=1.15,
                                        eos_id=ende):
                    if t == ende:
                        break
                    raus.append(t)
            antwort = tok.decode(raus).strip()
            antwort = " ".join(antwort.split())
            gesamt += 1
            zeichen = " "
            if erwartet:
                pruefbar += 1
                ok = any(t.lower() in antwort.lower() for t in erwartet.split("|"))
                treffer += ok
                zeichen = "\033[92m✓\033[0m" if ok else "\033[91m✗\033[0m"
            print(f" {zeichen} {frage[:40]:42}{antwort[:74]}")

    print(f"\n{'─'*72}")
    print(f"{gesamt} Fragen, davon {pruefbar} automatisch bewertbar: "
          f"\033[1m{treffer}/{pruefbar}\033[0m ({treffer/max(pruefbar,1)*100:.0f} %)")


if __name__ == "__main__":
    main()
