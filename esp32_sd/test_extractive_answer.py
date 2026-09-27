import unittest

from .extractive_answer import answer_from_text, answer_matches_question_type


class ExtractiveAnswerTests(unittest.TestCase):
    def test_height_answer(self):
        text = "Eiffelturm: Der Eiffelturm ist ein 330 Meter hoher Eisenfachwerkturm in Paris."
        self.assertEqual(
            answer_from_text("Wie hoch ist der Eiffelturm?", text),
            "Eiffelturm: 330 Meter hoch.",
        )

    def test_location_answer(self):
        text = "München: München liegt in Bayern und ist die Hauptstadt des Freistaates."
        self.assertEqual(
            answer_from_text("Wo liegt München?", text),
            "München liegt in Bayern und ist die Hauptstadt des Freistaates.",
        )

    def test_foundation_year_answer(self):
        text = "Museum: Das Museum wurde 1892 gegründet. Es zeigt volkskundliche Sammlungen."
        self.assertEqual(
            answer_from_text("Wann wurde das Museum gegründet?", text),
            "Museum: 1892 gegründet.",
        )

    def test_foundation_year_with_inflected_word(self):
        text = "Preis: Im Auftrag der 1900 gegründeten Stiftung wird der Preis vergeben."
        self.assertEqual(
            answer_from_text("Wann wurde der Preis gegründet?", text),
            "Preis: 1900 gegründeten.",
        )

    def test_definition_answer(self):
        text = "Transistor: Ein Transistor ist ein elektronisches Halbleiterbauelement. Er schaltet Signale."
        self.assertEqual(
            answer_from_text("Was ist ein Transistor?", text),
            "Ein Transistor ist ein elektronisches Halbleiterbauelement.",
        )

    def test_date_with_day_month_does_not_split_mid_date(self):
        text = "Berliner Mauer: Die Berliner Mauer bestand vom 13. August 1961 bis 1989."
        self.assertEqual(
            answer_from_text("Wann wurde die Berliner Mauer gebaut?", text),
            "Berliner Mauer: vom 13. August 1961.",
        )

    def test_area_answer(self):
        text = "Oeschinensee: Der See hat eine Fläche von 1,1 km²."
        self.assertEqual(
            answer_from_text("Wie groß ist die Fläche von Oeschinensee?", text),
            "Oeschinensee: 1,1 km².",
        )

    def test_area_written_as_quadratkilometer(self):
        text = "County: Das County hat eine Fläche von 162 Quadratkilometern."
        self.assertEqual(
            answer_from_text("Wie groß ist die Fläche von County?", text),
            "County: 162 Quadratkilometern.",
        )

    def test_area_millions_are_expanded(self):
        text = "Tiefland: Die Fläche beträgt etwa 2,6 Millionen km²."
        self.assertEqual(
            answer_from_text("Wie groß ist die Fläche von Tiefland?", text),
            "Tiefland: 2.600.000 km².",
        )

    def test_grouped_large_area_keeps_full_number(self):
        text = "Tiefland: Es ist etwa 2.600.000 km² groß."
        self.assertEqual(
            answer_from_text("Wie groß ist die Fläche von Tiefland?", text),
            "Tiefland: 2.600.000 km².",
        )

    def test_count_answer(self):
        text = "Hamburg: Hamburg hat 1.900.000 Einwohner und liegt an der Elbe."
        self.assertEqual(
            answer_from_text("Wie viele Einwohner hat Hamburg?", text),
            "Hamburg: 1.900.000 Einwohner.",
        )

    def test_count_searches_body_if_best_sentence_has_no_number(self):
        text = "Maun: Maun ist eine Stadt. Es handelt sich um eine Streusiedlung mit 60.263 Einwohnern."
        self.assertEqual(
            answer_from_text("Wie viele Einwohner hat Maun?", text),
            "Maun: 60.263 Einwohnern.",
        )

    def test_height_of_answer(self):
        text = "Turm: Der Turm erreicht eine Höhe von 64,6 m."
        self.assertEqual(
            answer_from_text("Wie hoch ist der Turm?", text),
            "Turm: Höhe von 64,6 m hoch.",
        )

    def test_height_in_metres_without_adjective(self):
        text = "Turm: Die Konstruktion hatte eine Höhe von 147 Metern. Der Tank war 93 m hoch."
        self.assertEqual(
            answer_from_text("Wie hoch ist der Turm?", text),
            "Turm: Höhe von 147 Metern hoch.",
        )

    def test_death_year_answer(self):
        text = "Person: Die Person starb am 10. April 1931 in New York."
        self.assertEqual(
            answer_from_text("Wann starb die Person?", text),
            "Person: starb am 10. April 1931.",
        )

    def test_answer_type_check(self):
        self.assertTrue(answer_matches_question_type("Wie tief ist Loch Ness?", "Loch Ness: 230 m tief."))
        self.assertFalse(answer_matches_question_type("Wie tief ist Loch Ness?", "Loch Ness ist 37 km lang."))


if __name__ == "__main__":
    unittest.main()
