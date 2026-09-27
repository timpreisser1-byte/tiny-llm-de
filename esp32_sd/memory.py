"""Personal memory: the chatbot learns from what the user tells it.

Why memory instead of weight updates: training the ternary model needs
full-precision shadow weights (~140 MB for 35M parameters) plus backprop.
Neither fits 8 MB PSRAM, and a 240 MHz core would need minutes per sentence.
Retrieval does fit: whatever the user states is appended to a log on the SD
card and is answerable in the very next turn - the same idea as the rest of
the project (knowledge on SD, not in weights).

Three learning signals:
  - facts      "Merk dir: Mein Hund heißt Bello."        -> art "fakt"
  - corrections "Falsch, richtig ist 452 km."             -> art "korrektur"
  - confirmations "Stimmt." after an answer                -> art "bestaetigt"
Corrections and confirmations are bound to the question they answer and win
over everything else the next time that question comes.

Log format (JSON lines, append-only, FAT-friendly; a power cut can only lose
the last line):
  {"id": 7, "art": "fakt", "text": "Mein Hund heißt Bello."}
  {"id": 8, "art": "korrektur", "frage": "lang weser", "text": "452 km"}
  {"id": 7, "art": "weg"}                  # tombstone written by "vergiss ..."
At boot the log is replayed into a small in-RAM word index; thousands of
entries fit easily in PSRAM.
"""

import json
import os
import re
from pathlib import Path

WORD = re.compile(r"[a-zäöüß0-9]{2,30}")
# Ignored when matching: question words, articles, auxiliaries and the
# personal words themselves (every personal fact contains "mein").
STOP = {
    "was", "wer", "wie", "wo", "wann", "warum", "welche", "welcher", "welches",
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
    "ist", "sind", "war", "hat", "habe", "hast", "heißt", "heiße", "heißen",
    "und", "oder", "zu", "in", "im", "am", "an", "auf", "von", "mit", "für",
    "ich", "mich", "mir", "mein", "meine", "meinen", "meinem", "meiner", "meines",
    "du", "dich", "dir", "dein", "deine", "bitte", "noch", "mal", "denn", "es",
    "wurde", "wird", "viele", "gibt", "dass", "so", "nicht", "kein", "keine",
}
PERSONAL = re.compile(r"\b(ich|mich|mir|mein|meine|meinen|meinem|meiner|meines)\b",
                      re.IGNORECASE)
# "Wie heißt mein Hund?" can only be answered from memory; Wikipedia would
# match "Mein Katalonien". Plain "ich" is not enough ("Was kann ich in Berlin
# besichtigen?" is a knowledge question).
POSSESSIVE = re.compile(r"\b(mein|meine|meinen|meinem|meiner|meines)\b", re.IGNORECASE)
QUESTION_START = re.compile(
    r"^(was|wer|wie|wo|wann|warum|wieso|weshalb|welche[rsnm]?|wieviel|kannst|"
    r"weißt|ist|sind|hat|gibt)\b", re.IGNORECASE)

TEACH = re.compile(r"^(?:bitte\s+)?(?:merk(?:e)?\s+dir|speicher(?:e)?|notier(?:e)?)"
                   r"\s*[:,]?\s*(?:dass\s+)?(.+)$", re.IGNORECASE)
FORGET = re.compile(r"^(?:bitte\s+)?vergiss\s*[:,]?\s*(?:dass\s+)?(.+)$", re.IGNORECASE)
CORRECT = re.compile(
    r"^(?:nein|falsch|stimmt nicht|das stimmt nicht|das ist falsch|quatsch)\b[,.!]*\s*"
    r"(?:(?:richtig|korrekt) (?:ist|wäre|sind|lautet)|es (?:ist|sind|war|waren)|sondern)?"
    r"\s*[:,]?\s*(.*)$", re.IGNORECASE)
CONFIRM = {"stimmt", "richtig", "genau", "korrekt", "das stimmt", "ja genau",
           "ja stimmt", "ja richtig", "stimmt genau", "das ist richtig"}
# "Nein danke" is politeness, not a correction whose value is "danke".
NOT_A_VALUE = {"danke", "danke schön", "danke dir", "passt", "egal", "schon gut",
               "lass mal", "ok", "okay", "gut"}
LIST = {"was weißt du über mich", "was hast du dir gemerkt", "was hast du gespeichert",
        "was weisst du ueber mich", "was weisst du über mich"}


def _stem(word):
    for suffix in ("ern", "en", "er", "es", "e", "s", "n"):
        if len(word) - len(suffix) >= 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_words(text):
    return [_stem(w) for w in WORD.findall(text.lower()) if w not in STOP]


def question_key(text):
    """Order-free key: 'Wie lang ist die Weser?' == 'Die Weser, wie lang?'"""
    return " ".join(sorted(set(content_words(text))))


def _sentence(text):
    text = text.strip()
    text = text[0].upper() + text[1:] if text else text
    return text if text[-1:] in ".!?" else text + "."


def is_question(text):
    t = text.strip()
    return t.endswith("?") or bool(QUESTION_START.match(t))


def soll_merken(satz):
    """Should a plain statement (no question, no command) be stored on its own?

    Called only for inputs that are neither questions nor "Merk dir ..." commands
    nor smalltalk, e.g. "Ich wohne in Köln." or "Das Wetter ist heute schön."
    Returning True stores the sentence as a personal fact.
    """
    # TODO(human): decide which plain statements the device learns automatically.
    return False


class Memory:
    def __init__(self, path):
        self.path = Path(path)
        self.entries = {}          # id -> entry
        self.words = {}            # id -> content words, built once at load
        self.next_id = 1
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue           # torn last line after a power cut
                    self._apply(entry)

    # ------------------------------------------------------------ storage

    def _apply(self, entry):
        self.next_id = max(self.next_id, entry.get("id", 0) + 1)
        if entry.get("art") == "weg":
            self.entries.pop(entry.get("id"), None)
            self.words.pop(entry.get("id"), None)
        else:
            self.entries[entry["id"]] = entry
            self.words[entry["id"]] = set(content_words(entry["text"]))

    def _append(self, entry):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._apply(entry)

    def remember(self, text, art="fakt", frage=None):
        entry = {"id": self.next_id, "art": art, "text": text}
        if frage is not None:
            entry["frage"] = frage
        # A newer answer to the same question replaces the older one.
        if frage is not None:
            for old in list(self.entries.values()):
                if old.get("frage") == frage:
                    self._append({"id": old["id"], "art": "weg"})
        self._append(entry)
        return entry

    def compact(self):
        """Rewrite the log with live entries only.

        A tombstone alone leaves the forgotten sentence readable on a card
        that can be pulled out and read on any PC. Write a new file, flush it
        to the card, then rename: a power cut leaves either the old or the new
        log, never a half-written one.
        """
        tmp = self.path.with_suffix(".neu")
        with tmp.open("w", encoding="utf-8") as f:
            for entry in self.entries.values():
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def forget(self, text):
        words = set(content_words(text))
        gone = []
        for entry in list(self.entries.values()):
            have = self.words[entry["id"]] | set(entry.get("frage", "").split())
            if words and len(words & have) >= max(1, (len(words) + 1) // 2):
                self._apply({"id": entry["id"], "art": "weg"})
                gone.append(entry["text"])
        if gone:
            self.compact()
        return gone

    # ------------------------------------------------------------ lookup

    def learned_answer(self, question):
        """Answer that the user corrected or confirmed for this question."""
        key = question_key(question)
        if not key:
            return None
        want = set(key.split())
        best, best_overlap = None, 0.0
        for entry in self.entries.values():
            if entry["art"] not in ("korrektur", "bestaetigt"):
                continue
            have = set(entry["frage"].split())
            overlap = len(want & have) / len(want | have)
            # >= so that on a tie the newest entry wins (dicts keep insertion
            # order): after "Rex" was taught, "Bello" must not come back.
            if overlap >= best_overlap and overlap > 0:
                best, best_overlap = entry, overlap
        return best if best_overlap >= 0.75 else None

    def personal_fact(self, question):
        words = content_words(question)
        if not words:
            return None
        best, best_hits = None, 0
        for entry in self.entries.values():
            if entry["art"] != "fakt":
                continue
            hits = len(set(words) & self.words[entry["id"]])
            if hits >= best_hits and hits > 0:
                best, best_hits = entry, hits
        if best is None:
            return None
        # Personal questions ("Wie heißt mein Hund?") need one shared word;
        # impersonal ones must mostly match, so Wikipedia questions are not
        # hijacked by an unrelated note.
        if PERSONAL.search(question) or best_hits >= max(2, int(0.6 * len(set(words)))):
            return best
        return None

    # ------------------------------------------------------------ chat hook

    def handle(self, question, last=None):
        """Called before search. Returns a reply dict or None.

        `last` is the previous (question, answer) pair; corrections and
        confirmations refer to it.
        """
        text = question.strip()
        low = re.sub(r"\s+", " ", text.lower()).strip(" .!?")

        m = TEACH.match(text)
        if m:
            fact = _sentence(m.group(1))
            self.remember(fact)
            return {"answer": f"Okay, gemerkt: „{fact}“", "mode": "Gedächtnis: gemerkt"}

        m = FORGET.match(text)
        if m:
            gone = self.forget(m.group(1))
            if gone:
                return {"answer": "Okay, vergessen: " + " · ".join(gone), "mode": "Gedächtnis: gelöscht"}
            return {"answer": "Dazu habe ich mir nichts gemerkt.", "mode": "Gedächtnis"}

        if low in LIST:
            facts = [e["text"] for e in self.entries.values() if e["art"] == "fakt"]
            if not facts:
                return {"answer": "Ich habe mir noch nichts über dich gemerkt.", "mode": "Gedächtnis"}
            return {"answer": "Das hast du mir erzählt: " + " · ".join(facts[-10:]),
                    "mode": "Gedächtnis"}

        if last is not None:
            m = CORRECT.match(text)
            if m and m.group(1).strip(" .!").lower() not in NOT_A_VALUE:
                right = m.group(1).strip(" .!")
                if not right:
                    return {"answer": "Danke. Was ist richtig? Sag zum Beispiel: "
                                      "„Falsch, richtig ist …“", "mode": "Gedächtnis"}
                self.remember(right, art="korrektur", frage=question_key(last[0]))
                return {"answer": f"Danke, ich habe es korrigiert: {right}.",
                        "mode": "Gedächtnis: korrigiert"}
            # Only answers from the knowledge stages are worth caching;
            # confirming a memory answer would freeze it against later edits.
            if low in CONFIRM and last[1] and not (len(last) > 2 and last[2]):
                self.remember(last[1], art="bestaetigt", frage=question_key(last[0]))
                return {"answer": "Danke, dann merke ich mir die Antwort so.",
                        "mode": "Gedächtnis: bestätigt"}

        learned = self.learned_answer(text)
        if learned:
            prefix = "Du hast mir gesagt: " if learned["art"] == "korrektur" else ""
            return {"answer": prefix + learned["text"], "mode": "Gedächtnis: gelernt"}

        if is_question(text):
            fact = self.personal_fact(text)
            if fact:
                return {"answer": f"Das hast du mir erzählt: „{fact['text']}“",
                        "mode": "Gedächtnis: persönlich"}
            if POSSESSIVE.search(text):
                return {"answer": "Das hast du mir noch nicht erzählt. Sag zum Beispiel: "
                                  "„Merk dir: …“", "mode": "Gedächtnis"}
            return None

        if soll_merken(text):
            fact = _sentence(text)
            self.remember(fact)
            return {"answer": f"Okay, das merke ich mir: „{fact}“", "mode": "Gedächtnis: gemerkt"}
        return None
