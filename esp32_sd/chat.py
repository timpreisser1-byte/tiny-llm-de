"""Desktop reference chatbot for the SD index; not ESP32 firmware."""

import argparse
import json
import re
from pathlib import Path

from .extractive_answer import answer_from_text, answer_matches_question_type
from .memory import Memory
from .sd_index import SDIndex, MAX_TERMS, tokenize
from .smalltalk import smalltalk_answer


def _question_subject(question):
    normalized = question.strip(" .!?")
    patterns = [
        r"^wie (?:groß|gross) ist die fläche von\s+(.+)$",
        r"^wie viele einwohner hat\s+(.+)$",
        r"^(?:wo (?:liegt|steht|ist)|wie (?:hoch|groß|alt|lang|tief) ist|was ist|wer ist)\s+(.+)$",
        r"^wann wurde\s+(.+?)\s+(?:gebaut|geboren|gegründet|eröffnet|errichtet)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, normalized, re.IGNORECASE)
        if match:
            return match.group(1).strip(" .!?")
    return ""


def _title_of(text):
    title, separator, _ = text.partition(":")
    return title.strip() if separator else ""


def _pick_hit(question, hits):
    subject = _question_subject(question).casefold()
    numeric = re.match(r"^(?:wie (?:hoch|groß|gross|lang|tief)|wie viele|wieviele|wann)",
                       question.strip(), re.IGNORECASE)
    if subject:
        exact = [(score, text) for score, text in hits if _title_of(text).casefold() == subject]
        if numeric:
            typed = []
            for score, text in exact:
                answer = answer_from_text(question, text) or ""
                if answer_matches_question_type(question, answer):
                    typed.append((score, text, answer))
            if typed:
                # A length in kilometres usually describes the extent of a
                # place better than a local measurement in metres.
                if question.lower().strip().startswith("wie lang"):
                    for score, text, answer in typed:
                        if re.search(r"\b(?:kilometer|km)\b", answer, re.IGNORECASE):
                            return score, text
                return typed[0][:2]
        if exact:
            return exact[0]
    return hits[0]


class Chat:
    def __init__(self, index, model=None, memory=None, facts=None):
        self.index = index
        self.model = model
        self.memory = memory
        self.facts = facts        # Wikidata tables (fakten.Fakten); on the device: sorted SD files
        self.topic = ""
        self.last = None          # (question, answer) for corrections

    def answer(self, question):
        # The memory goes first: a correction the user gave must beat the
        # Wikipedia search that produced the wrong answer in the first place.
        if self.memory is not None and question.strip():
            result = self.memory.handle(question, self.last)
            if result is not None:
                result.setdefault("source", None)
                if result["mode"] in ("Gedächtnis: gelernt", "Gedächtnis: persönlich"):
                    # third field: this answer came from memory itself
                    self.last = (question, result["answer"], True)
                return result
        result = self._answer(question)
        self.last = (question, result["answer"]) if result.get("source") else None
        return result

    def _answer(self, question):
        question = question.strip()
        if len(question) > 512:
            return {"answer": "Bitte beschränke die Frage auf 512 Zeichen.", "source": None}
        normalized = question.lower().strip(" .!?")
        if normalized in {"neu", "/neu"}:
            self.topic = ""
            return {"answer": "Neues Gespräch. Was möchtest du wissen?", "source": None}
        smalltalk = smalltalk_answer(question, self.topic)
        if smalltalk is not None:
            return {"answer": smalltalk, "source": None, "mode": "Smalltalk"}
        # Resolve only short, explicit followups, not impersonal German "es".
        query = question
        followup = re.fullmatch(
            r"(?:wo (?:liegt|steht|ist)|wie (?:hoch|groß|alt|lang|tief) ist|"
            r"wann wurde) (?:er|sie|es|dieser|diese|dieses)(?: (?:gebaut|geboren|gegründet))?",
            normalized)
        if self.topic and followup:
            query = re.sub(r"\b(er|sie|es|dieser|diese|dieses|dort)\b", self.topic,
                           question, flags=re.IGNORECASE)
        if len(set(tokenize(query))) > MAX_TERMS:
            return {"answer": "Bitte stelle eine kürzere Frage mit höchstens 32 Suchwörtern.",
                    "source": None}
        # Tables before search: the Wikipedia copy lost its infoboxes, so
        # heights, dates and authors are often missing from the text itself.
        if self.facts is not None:
            fact = self.facts.antwort(query)
            if fact:
                return {"answer": fact, "source": "Wikidata", "mode": "Tabelle (Wikidata)",
                        "query": query}
        hits = self.index.suche(query, anzahl=30)
        if not hits:
            return {"answer": "Dazu finde ich keinen passenden Text in meinem Index.", "source": None}
        _, text = _pick_hit(query, hits)
        title = _title_of(text)
        separator = bool(title)
        self.topic = title[:120] if separator else ""
        if self.model is not None:
            generated = self.model.generate(question, text)
            if generated:
                return {"answer": generated, "context": text, "source": self.topic or None,
                        "mode": "Modellantwort mit Wikipedia-Kontext",
                        "query": query}
        extracted = answer_from_text(question, text)
        if extracted:
            return {"answer": extracted, "context": text, "source": self.topic or None,
                    "mode": "Extraktive Antwort aus Wikipedia-Kontext",
                    "query": query}
        # Return the source verbatim: retrieval scores are not factual confidence.
        return {"answer": text, "source": self.topic or None,
                "mode": "Wikipedia-Textauszug, keine generierte Antwort",
                "query": query}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*")
    gestuft = Path(__file__).parent / "wiki_sd_gestuft"
    parser.add_argument("--index", type=Path,
                        default=gestuft if gestuft.exists() else Path(__file__).parent / "wiki_sd")
    parser.add_argument("--checkpoint", type=Path,
                        help="Optional: PyTorch-Checkpoint fuer generierte Antworten")
    parser.add_argument("--tokenizer", type=Path,
                        default=Path(__file__).resolve().parents[1] / "tokenizer/de_bpe.json")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--memory", type=Path,
                        default=Path(__file__).parent / "gedaechtnis.jsonl",
                        help="Gedaechtnis-Protokoll (auf dem Geraet: SD-Karte)")
    parser.add_argument("--ohne-gedaechtnis", action="store_true")
    parser.add_argument("--ohne-fakten", action="store_true",
                        help="Wikidata-Tabellen nicht laden (spart ~30 s Start am PC)")
    args = parser.parse_args()
    model = None
    if args.checkpoint:
        from .model_backend import ModelBackend
        model = ModelBackend(args.checkpoint, args.tokenizer, args.max_new_tokens)
    with SDIndex(args.index) as index:
        memory = None if args.ohne_gedaechtnis else Memory(args.memory)
        facts = None
        if not args.ohne_fakten:
            try:
                from knowledge.facts import Fakten      # project root, run as python -m esp32_sd.chat
                facts = Fakten()
            except ImportError:
                print("[Wikidata-Tabellen nicht gefunden - weiter ohne]")
        chat = Chat(index, model=model, memory=memory, facts=facts)
        if args.question:
            result = chat.answer(" ".join(args.question))
            print(json.dumps(result, ensure_ascii=False) if args.json else result["answer"])
            return
        print("Wikipedia-Chat · Desktop-Referenz · /neu setzt das Thema zurück · /ende beendet")
        print("Lernen: „Merk dir: …“ · „Falsch, richtig ist …“ · „Stimmt.“ · „Vergiss …“")
        while True:
            try:
                question = input("Du: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.strip().lower() == "/ende":
                break
            result = chat.answer(question)
            print("Bot:", result["answer"])
            if result.get("source"):
                print(f"[{result.get('mode')}; Suchtreffer kann unpassend sein]")


if __name__ == "__main__":
    main()
