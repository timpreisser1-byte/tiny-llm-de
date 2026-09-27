"""Viele Frageformen je Muster - und eine davon bleibt zurückgehalten.

Der Befund des Projekts zum Lesen, übertragen auf das Fragen:

    eine Vorlage      +42 auf der geübten Form,  +3 auf jeder anderen
    18 Vorlagen       +50 auf der geübten Form, +44,7 auf fremder Form

Beim Fragen stehen wir noch bei "eine Vorlage je Muster". Gemessen: 87 % auf
der geübten Form, 26,7 % auf handgeschriebenen Fragen - schlechter als das
Modell vor dem Training. Also dieselbe Medizin.

Jedes Muster bekommt hier vier Formulierungen. Drei gehen ins Training, die
vierte NUR in die Prüfung. Damit ist die zurückgehaltene Form garantiert
ungeübt, ohne dass ich mich darauf verlassen muss, sie nicht versehentlich
doch erzeugt zu haben.
"""

# (Regex, [drei Trainingsformen, eine Pruefform], Antwortart)
# Die vierte Form ist bewusst weiter entfernt als die ersten drei - andere
# Wortstellung, anderes Verb, teils ohne "wie/wann/wer" am Anfang.
MUSTER = [
    (r"\b(\d{1,4}(?:[.,]\d+)?)\s*(?:m|Meter)\s+(?:hoch|hohe[rnms]?)\b",
     ["Wie hoch ist {t}?", "Welche Höhe hat {t}?", "Wie hoch ragt {t} auf?",
      "Auf welche Höhe kommt {t}?"], "Höhe"),
    (r"\b(\d{1,4}(?:[.,]\d+)?)\s*(?:m|Meter)\s+(?:tief|tiefe[rnms]?)\b",
     ["Wie tief ist {t}?", "Welche Tiefe hat {t}?", "Wie weit reicht {t} in die Tiefe?",
      "Bis zu welcher Tiefe geht {t}?"], "Tiefe"),
    (r"\b(\d{1,5}(?:[.,]\d+)?)\s*(?:km|Kilometer)\s+(?:lang|lange[rnms]?)\b",
     ["Wie lang ist {t}?", "Welche Länge hat {t}?", "Wie weit erstreckt sich {t}?",
      "Über welche Strecke zieht sich {t}?"], "Länge"),
    (r"\b(\d{1,3}(?:\.\d{3})*(?:,\d+)?)\s*(?:Einwohner|Einwohnern)\b",
     ["Wie viele Einwohner hat {t}?", "Wie viele Menschen leben in {t}?",
      "Wie groß ist die Einwohnerzahl von {t}?",
      "Von wie vielen Menschen wird {t} bewohnt?"], "Anzahl"),
    (r"\b(\d{1,3}(?:\.\d{3})*(?:,\d+)?)\s*(?:km²|Quadratkilometer)\b",
     ["Wie groß ist die Fläche von {t}?", "Welche Fläche hat {t}?",
      "Wie viel Fläche nimmt {t} ein?", "Auf welche Größe kommt {t}?"], "Fläche"),
    (r"\bgeboren[^.]{0,40}?\b(1[0-9]{3}|20[0-2][0-9])\b|"
     r"\b(?:1[0-9]{3}|20[0-2][0-9])\b[^.]{0,40}?\bgeboren",
     ["Wann wurde {t} geboren?", "In welchem Jahr wurde {t} geboren?",
      "Wann kam {t} zur Welt?", "Welches Geburtsjahr hat {t}?"], "Jahr"),
    (r"\b(1[0-9]{3}|20[0-2][0-9])\b[^.]{0,50}?\bgegründet|"
     r"\bgegründet[^.]{0,40}?\b(1[0-9]{3}|20[0-2][0-9])\b",
     ["Wann wurde {t} gegründet?", "In welchem Jahr entstand {t}?",
      "Wann ist {t} gegründet worden?", "Aus welchem Jahr datiert {t}?"], "Jahr"),
    (r"\b(1[0-9]{3}|20[0-2][0-9])\b[^.]{0,50}?\b(?:eröffnet|erbaut|errichtet)",
     ["Wann wurde {t} eröffnet?", "In welchem Jahr wurde {t} gebaut?",
      "Wann ging {t} in Betrieb?", "Seit wann gibt es {t}?"], "Jahr"),
    (r"\bstarb[^.]{0,50}?\b(1[0-9]{3}|20[0-2][0-9])\b",
     ["Wann starb {t}?", "In welchem Jahr starb {t}?",
      "Wann ist {t} gestorben?", "Welches Todesjahr hat {t}?"], "Jahr"),
    (r"\bvon\s+([A-ZÄÖÜ][\wäöüß-]{2,}(?:\s+[A-ZÄÖÜ][\wäöüß-]{2,})?)\s+"
     r"(?:entwickelt|erfunden|entworfen|konstruiert)",
     ["Wer hat {t} entwickelt?", "Von wem stammt {t}?",
      "Wer steckt hinter {t}?", "Auf wen geht {t} zurück?"], "Person"),
    (r"\bvon\s+([A-ZÄÖÜ][\wäöüß-]{2,}(?:\s+[A-ZÄÖÜ][\wäöüß-]{2,})?)\s+"
     r"(?:gegründet|erbaut|errichtet|geschrieben|komponiert|verfasst)",
     ["Wer hat {t} geschaffen?", "Wer war der Urheber von {t}?",
      "Von wessen Hand stammt {t}?", "Wem verdankt {t} seine Entstehung?"], "Person"),
    (r"\bliegt\s+(?:in|im|am|an der)\s+(?:der\s+|dem\s+)?"
     r"([A-ZÄÖÜ][\wäöüß-]{2,}(?:\s+[A-ZÄÖÜ][\wäöüß-]{2,})?)",
     ["Wo liegt {t}?", "In welcher Gegend liegt {t}?",
      "Wo ist {t} zu finden?", "In welcher Umgebung befindet sich {t}?"], "Ort"),
    (r"\bbefindet sich\s+(?:in|im|am)\s+(?:der\s+|dem\s+)?"
     r"([A-ZÄÖÜ][\wäöüß-]{2,}(?:\s+[A-ZÄÖÜ][\wäöüß-]{2,})?)",
     ["Wo befindet sich {t}?", "An welchem Ort steht {t}?",
      "Wo genau ist {t}?", "Welchem Ort ist {t} zuzuordnen?"], "Ort"),
    (r"\bmündet\s+(?:in|bei)\s+(?:die\s+|den\s+|das\s+)?"
     r"([A-ZÄÖÜ][\wäöüß-]{2,})",
     ["Wo mündet {t}?", "Wohin fließt {t}?",
      "In was mündet {t}?", "Welches Gewässer nimmt {t} auf?"], "Ort"),
    (r"\bgehört\s+zu(?:m|r)?\s+([A-ZÄÖÜ][\wäöüß-]{2,}(?:\s+[A-ZÄÖÜ][\wäöüß-]{2,})?)",
     ["Wozu gehört {t}?", "Zu welcher Einheit zählt {t}?",
      "Wem ist {t} zugeordnet?", "In welchen Verband fällt {t}?"], "Ort"),
]

TRAIN_FORMEN = 3          # die ersten drei je Muster
PRUEF_FORM = 3            # Index der zurueckgehaltenen


def formen(muster, fuer):
    """Frageformen eines Musters holen: 'training' oder 'pruefung'."""
    return muster[1][:TRAIN_FORMEN] if fuer == "training" else [muster[1][PRUEF_FORM]]
