import unittest

from .smalltalk import smalltalk_answer


class SmalltalkTests(unittest.TestCase):
    def test_greeting(self):
        self.assertIn("plaudern", smalltalk_answer("Hallo!"))

    def test_identity(self):
        self.assertIn("ESP32-S3", smalltalk_answer("Wer bist du?"))

    def test_topic(self):
        self.assertEqual(
            smalltalk_answer("Was war das Thema?", "Eiffelturm"),
            "Gerade ist das Thema: Eiffelturm.",
        )

    def test_unknown_returns_none(self):
        self.assertIsNone(smalltalk_answer("Wie hoch ist der Eiffelturm?"))


if __name__ == "__main__":
    unittest.main()
