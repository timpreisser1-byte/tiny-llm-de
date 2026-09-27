import unittest

from .chat import Chat, _pick_hit, _question_subject


class FakeIndex:
    def __init__(self):
        self.queries = []

    def suche(self, query, anzahl):
        self.queries.append(query)
        if "München" in query:
            return [
                (2, "Au (München): Die Au liegt rechts der Isar."),
                (1, "München: München liegt in Bayern."),
            ]
        return [(1, "Eiffelturm: Der Eiffelturm steht in Paris.")] if "Eiffelturm" in query else []


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, question, context):
        self.calls.append((question, context))
        return "Der Eiffelturm steht in Paris."


class ChatTests(unittest.TestCase):
    def test_greeting_does_not_search(self):
        index = FakeIndex()
        self.assertIsNone(Chat(index).answer("Hallo!")["source"])
        self.assertEqual(index.queries, [])

    def test_smalltalk_identity_does_not_search(self):
        index = FakeIndex()
        answer = Chat(index).answer("Wer bist du?")
        self.assertEqual(answer["mode"], "Smalltalk")
        self.assertIn("Offline-Wissens-KI", answer["answer"])
        self.assertEqual(index.queries, [])

    def test_smalltalk_topic_uses_state(self):
        index = FakeIndex()
        chat = Chat(index)
        chat.answer("Was ist der Eiffelturm?")
        answer = chat.answer("Was war das Thema?")
        self.assertEqual(answer["answer"], "Gerade ist das Thema: Eiffelturm.")

    def test_followup_and_reset(self):
        index = FakeIndex()
        chat = Chat(index)
        chat.answer("Was ist der Eiffelturm?")
        self.assertIn("Eiffelturm", chat.answer("Wo steht er?")["query"])
        chat.answer("/neu")
        self.assertIsNone(chat.answer("Wo steht er?")["source"])

    def test_new_topic_does_not_inherit(self):
        index = FakeIndex()
        chat = Chat(index)
        chat.answer("Eiffelturm")
        chat.answer("Was ist ein Transistor?")
        self.assertEqual(index.queries[-1], "Was ist ein Transistor?")

    def test_impersonal_pronoun_does_not_inherit(self):
        index = FakeIndex()
        chat = Chat(index)
        chat.answer("Eiffelturm")
        question = "Was bedeutet es, wenn Wasser gefriert?"
        chat.answer(question)
        self.assertEqual(index.queries[-1], question)

    def test_too_many_words_does_not_search(self):
        index = FakeIndex()
        answer = Chat(index).answer(" ".join(f"begriff{i}" for i in range(33)))
        self.assertIn("32", answer["answer"])
        self.assertEqual(index.queries, [])

    def test_model_answer_uses_retrieved_context(self):
        index = FakeIndex()
        model = FakeModel()
        answer = Chat(index, model=model).answer("Wo steht der Eiffelturm?")
        self.assertEqual(answer["answer"], "Der Eiffelturm steht in Paris.")
        self.assertEqual(answer["mode"], "Modellantwort mit Wikipedia-Kontext")
        self.assertIn("Eiffelturm steht in Paris", answer["context"])
        self.assertEqual(model.calls[0][0], "Wo steht der Eiffelturm?")

    def test_default_answer_is_extractive(self):
        answer = Chat(FakeIndex()).answer("Wo steht der Eiffelturm?")
        self.assertEqual(answer["answer"], "Der Eiffelturm steht in Paris.")
        self.assertEqual(answer["mode"], "Extraktive Antwort aus Wikipedia-Kontext")

    def test_prefers_exact_title_for_question_subject(self):
        answer = Chat(FakeIndex()).answer("Wo liegt München?")
        self.assertEqual(answer["source"], "München")
        self.assertEqual(answer["answer"], "München liegt in Bayern.")

    def test_question_subject_for_area_and_population(self):
        self.assertEqual(_question_subject("Wie groß ist die Fläche von Rohnert Park?"),
                         "Rohnert Park")
        self.assertEqual(_question_subject("Wie viele Einwohner hat Stockholm?"),
                         "Stockholm")

    def test_length_prefers_kilometres_over_local_metre_measurement(self):
        hit = _pick_hit("Wie lang ist Testort?", [
            (2, "Testort: Der Wanderweg ist 1200 Meter lang."),
            (1, "Testort: Testort ist 160 km lang."),
        ])
        self.assertIn("160 km", hit[1])


if __name__ == "__main__":
    unittest.main()
