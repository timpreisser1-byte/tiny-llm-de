"""Zwei Korpora aus derselben Quelle: einmal roh, einmal zum Nachschlagen geformt.

Die Kernthese der Recherche lautet, dass ein 64M-Modell nichts wissen kann,
aber lesen lernen kann - und dass man das Lesen schon im Vortraining ueben
muss, nicht erst beim Feintuning. Wer es erst spaeter uebt, hat vorher
Kapazitaet ans Auswendiglernen verbraucht, die nicht zurueckkommt.

Diese These wird hier pruefbar gemacht. Aus denselben Wikipedia-Absaetzen
entstehen zwei Korpora mit derselben Tokenzahl:

    A (roh)     jeder Absatz als reiner Text - so wie bisher trainiert
    B (geformt) jeder fuenfte Absatz stattdessen in der Form, in der das
                Geraet spaeter fragt:

                    Wissen: <Absatz>
                    Frage: <Satz mit Luecke>
                    Antwort: <Zahl>

Der Unterschied ist also nicht mehr Text, sondern anders angeordneter. Wenn
die These stimmt, liest B besser, ohne dass B mehr gesehen haette.

    python leseform.py --mb 70

Wiederhergestellt am 27.09.2026: Der Quelltext dieser Datei war verloren, nur
die kompilierte Fassung im Cache war noch da. Zurueckgeholt sind die
Frage-Vorlagen (VORLAGEN), die kurzkontext.py braucht. Die Funktionen zum
Formen der beiden Korpora (forme, verfaelsche, main) sind nicht mehr als
Quelltext vorhanden.
"""

# Frage-Vorlagen fuer Lesebeispiele: {a} Absatz, {f} Frage, {z} Antwort.
VORLAGEN = [
    'Wissen: {a}\nFrage: {f} Nenne nur die Zahl.\nAntwort: {z}\n',
    'Text: {a}\nFrage: {f}\nAntwort: {z}\n',
    'Quelle: {a}\nWelche Zahl fehlt? {f}\nLösung: {z}\n',
    'Abschnitt: {a}\nErgänze: {f}\n{z}\n',
    '{a}\nFrage dazu: {f}\nDie fehlende Zahl ist {z}.\n',
    'Vorgelegt: {a}\nGesucht: {f}\nGefunden: {z}\n',
    'Frage: {f}\nDazu dieser Text: {a}\nAntwort: {z}\n',
    'Lies den Abschnitt: {a}\nWas gehört in die Lücke? {f}\nEs ist {z}.\n',
    'Grundlage: {a}\nAufgabe: {f}\nErgebnis: {z}\n',
    '{a}\nErgänze die Lücke: {f}\nAntwort: {z}\n',
    'Notiz: {a}\nFrage: {f}\n{z}\n',
    'Im Text steht: {a}\nWelche Zahl gehört hier hin? {f}\nLösung: {z}\n',
    'Beleg: {a}\nDaraus: {f}\nDie Zahl lautet {z}.\n',
    '{a}\nWelche Zahl fehlt in diesem Satz? {f}\nEs ist {z}.\n',
    'Unterlage: {a}\nZu ergänzen: {f}\nEingesetzt: {z}\n',
    'Vorlage: {a}\nFrage: {f}\nAus dem Text: {z}\n',
    '{a}\nGefragt ist: {f}\nAntwort: {z}\n',
    'Ausschnitt: {a}\nErgänze die fehlende Zahl: {f}\nSie lautet {z}.\n',
]
