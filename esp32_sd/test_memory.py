import tempfile
import unittest
from pathlib import Path

from .memory import Memory, question_key


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "gedaechtnis.jsonl"
        self.memory = Memory(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_teach_and_recall(self):
        reply = self.memory.handle("Merk dir: Mein Hund heißt Bello")
        self.assertIn("gemerkt", reply["answer"])
        answer = self.memory.handle("Wie heißt mein Hund?")
        self.assertIn("Bello", answer["answer"])

    def test_survives_restart(self):
        self.memory.handle("Merk dir, dass meine Schwester Anna heißt.")
        reloaded = Memory(self.path)
        self.assertIn("Anna", reloaded.handle("Wie heißt meine Schwester?")["answer"])

    def test_correction_wins_next_time(self):
        last = ("Wie lang ist die Weser?", "23,7")
        reply = self.memory.handle("Falsch, richtig ist 452 km", last)
        self.assertIn("korrigiert", reply["answer"])
        again = self.memory.handle("Wie lang ist die Weser?")
        self.assertIn("452 km", again["answer"])

    def test_correction_without_value_asks(self):
        reply = self.memory.handle("Das stimmt nicht.", ("Wie hoch ist X?", "3 m"))
        self.assertIn("Was ist richtig", reply["answer"])
        self.assertIsNone(self.memory.learned_answer("Wie hoch ist X?"))

    def test_confirmation_caches_answer(self):
        self.memory.handle("Stimmt.", ("Wie hoch ist der Brocken?", "1141 Meter"))
        self.assertEqual(self.memory.learned_answer("Wie hoch ist der Brocken")["text"],
                         "1141 Meter")

    def test_newer_correction_replaces_older(self):
        last = ("Wie lang ist die Weser?", "23,7")
        self.memory.handle("Falsch, richtig ist 451 km", last)
        self.memory.handle("Nein, es sind 452 km", last)
        self.assertEqual(self.memory.learned_answer("Wie lang ist die Weser?")["text"], "452 km")
        self.assertEqual(len(Memory(self.path).entries), 1)

    def test_forget(self):
        self.memory.handle("Merk dir: Mein Hund heißt Bello")
        reply = self.memory.handle("Vergiss meinen Hund")
        self.assertIn("vergessen", reply["answer"])
        self.assertIn("noch nicht erzählt",
                      Memory(self.path).handle("Wie heißt mein Hund?")["answer"])

    def test_wikipedia_question_not_hijacked(self):
        self.memory.handle("Merk dir: Mein Hund heißt Bello")
        self.assertIsNone(self.memory.handle("Wie hoch ist der Eiffelturm?"))

    def test_ich_question_still_goes_to_wikipedia(self):
        self.assertIsNone(self.memory.handle("Was kann ich in Berlin besichtigen?"))

    def test_newest_fact_wins_on_tie(self):
        self.memory.handle("Merk dir: Mein Hund heißt Bello")
        self.memory.handle("Merk dir: Mein Hund heißt Rex")
        self.assertIn("Rex", self.memory.handle("Wie heißt mein Hund?")["answer"])

    def test_no_thanks_is_not_a_correction(self):
        last = ("Wie lang ist die Weser?", "452 km")
        self.assertIsNone(self.memory.handle("Nein danke", last))
        self.assertIsNone(self.memory.learned_answer("Wie lang ist die Weser?"))

    def test_confirming_a_memory_answer_does_not_freeze_it(self):
        self.memory.handle("Merk dir: Mein Hund heißt Bello")
        last = ("Wie heißt mein Hund?", "Das hast du mir erzählt: „Mein Hund heißt Bello.“", True)
        self.memory.handle("Stimmt.", last)
        self.memory.handle("Merk dir: Mein Hund heißt Rex")
        self.assertIn("Rex", self.memory.handle("Wie heißt mein Hund?")["answer"])

    def test_forget_removes_text_from_card(self):
        self.memory.handle("Merk dir: Meine PIN ist 4711")
        self.memory.handle("Vergiss meine PIN")
        self.assertNotIn("4711", self.path.read_text(encoding="utf-8"))

    def test_question_key_ignores_order_and_filler(self):
        self.assertEqual(question_key("Wie lang ist die Weser?"),
                         question_key("Die Weser, wie lang ist die?"))

    def test_torn_last_line_is_ignored(self):
        self.memory.handle("Merk dir: Ich wohne in Köln")
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"id": 9, "art": "fak')       # power cut mid-write
        self.assertIn("Köln", Memory(self.path).handle("Wo wohne ich?")["answer"])


if __name__ == "__main__":
    unittest.main()
