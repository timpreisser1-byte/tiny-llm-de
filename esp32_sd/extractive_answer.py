"""Small extractive answer helper for the SD chatbot.

The ESP32-friendly path should prefer grounded snippets over free generation.
These rules intentionally answer only when the evidence is visible in the
retrieved paragraph; otherwise the caller can fall back to the paragraph.
"""

import re


def _body(text):
    title, sep, rest = text.partition(":")
    return (title.strip(), rest.strip() if sep else text.strip())


def _sentences(text):
    protected = re.sub(r"\b(\d{1,2})\.\s+(?=[A-ZÄÖÜ][a-zäöüß]+)", r"\1<DOT> ", text)
    return [s.replace("<DOT>", ".").strip()
            for s in re.split(r"(?<=[.!?])\s+", protected) if s.strip()]


def _first_match(patterns, text):
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return " ".join(match.group(0).split())
    return None


def _measure_answer(title, hit, adjective=None):
    hit = " ".join(hit.split())
    if adjective:
        value = re.sub(r"\s+(?:hoher|hohe|hohes|hoch|langer|lange|lang|tiefer|tiefe|tief)\b",
                       "", hit, flags=re.IGNORECASE)
        hit = f"{value} {adjective}"
    return f"{title}: {hit}." if title else f"{hit}."


def _expand_million_area(hit):
    match = re.search(r"\b(\d+)(?:[,.](\d+))?\s+millionen\s+(km²|km2|quadratkilometer)\b",
                      hit, re.IGNORECASE)
    if not match:
        return hit
    whole = int(match.group(1))
    frac = (match.group(2) or "").ljust(6, "0")[:6]
    value = whole * 1_000_000 + int(frac)
    return f"{value:,}".replace(",", ".") + " km²"


def has_typed_answer(question, text):
    title, body = _body(text)
    normalized = question.lower().strip(" .!?")
    if normalized.startswith("wie hoch"):
        return bool(_first_match([
            r"\b(?:höhe|hoehe)\s+von\s+\d{1,4}(?:[.,]\d+)?\s*(?:meter|m)\b",
            r"\b\d{1,4}(?:[.,]\d+)?\s*(?:meter|m)\s+(?:hoher|hohe|hohes|hoch)\b",
        ], body))
    if normalized.startswith("wie lang"):
        return bool(_first_match([
            r"\b\d{1,5}(?:[.,]\d+)?\s*(?:kilometer|km|meter|m)\s+(?:langer|lange|lang)\b",
        ], body))
    if normalized.startswith("wie groß") or normalized.startswith("wie gross"):
        return bool(_first_match([
            r"\b\d{1,3}(?:[.,]\d+)?\s+millionen\s+(?:km²|km2|quadratkilometer)\b",
            r"\b\d{1,3}(?:[.\s]\d{3})+\s*(?:km²|km2|m²|m2|quadratkilometer(?:n)?|hektar|ha)\b",
            r"\b\d{1,6}(?:[.,]\d+)?\s*(?:km²|km2|m²|m2|quadratkilometer(?:n)?|hektar|ha)\b",
        ], body))
    if normalized.startswith("wie tief"):
        return bool(_first_match([
            r"\b\d{1,5}(?:[.,]\d+)?\s*(?:meter|m)\s+(?:tiefer|tiefe|tief)\b",
        ], body))
    if normalized.startswith("wie viele") or normalized.startswith("wieviele"):
        return bool(_first_match([
            r"\b\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+)?\s*(?:einwohnern?|mitgliedern|mitglieder|mannschaften|plätze|plaetze|spieler|personen)\b",
        ], body))
    if normalized.startswith("wann"):
        return bool(_first_match([r"\b[12][0-9]{3}\b"], body))
    return False


def answer_matches_question_type(question, answer):
    normalized = question.lower().strip(" .!?")
    if not re.search(r"\d", answer):
        return False
    if normalized.startswith("wie hoch"):
        return bool(re.search(r"\b(?:\d|höhe|hoehe).*(?:m|meter).*hoch\b", answer, re.IGNORECASE))
    if normalized.startswith("wie lang"):
        return bool(re.search(r"\b\d.*(?:km|kilometer|m|meter).*lang\b", answer, re.IGNORECASE))
    if normalized.startswith("wie groß") or normalized.startswith("wie gross"):
        return bool(re.search(r"\b\d.*(?:km²|km2|m²|m2|quadratkilometer|hektar|ha)\b",
                              answer, re.IGNORECASE))
    if normalized.startswith("wie tief"):
        return bool(re.search(r"\b\d.*(?:m|meter).*tief\b", answer, re.IGNORECASE))
    if normalized.startswith("wie viele") or normalized.startswith("wieviele"):
        if "einwohner" in normalized:
            return bool(re.search(r"\b\d.*einwohner", answer, re.IGNORECASE))
        return True
    if normalized.startswith("wann"):
        return True
    return True


def _best_sentence(question, title, sentences):
    terms = {t for t in re.findall(r"[a-zäöüß0-9]{4,}", question.lower())}
    title_terms = {t for t in re.findall(r"[a-zäöüß0-9]{4,}", title.lower())}
    terms -= title_terms
    best = None
    best_score = -1
    for sentence in sentences:
        lower = sentence.lower()
        score = sum(1 for term in terms if term in lower)
        if title and title.lower() in lower:
            score += 1
        if score > best_score:
            best = sentence
            best_score = score
    return best


def answer_from_text(question, text):
    title, body = _body(text)
    sentences = _sentences(body)
    if not sentences:
        return None
    normalized = question.lower().strip(" .!?")

    if normalized.startswith("wie hoch"):
        hit = _first_match([
            r"\b(?:höhe|hoehe)\s+von\s+\d{1,4}(?:[.,]\d+)?\s*(?:meter|m)\b",
            r"\b(?:höhe|hoehe)\s+(?:von\s+)?\d{1,4}(?:[.,]\d+)?\s*(?:metern|meter|m)\b",
            r"\b\d{1,4}(?:[.,]\d+)?\s*(?:meter|m)\s+(?:hoher|hohe|hohes)\b",
            r"\b\d{1,4}(?:[.,]\d+)?\s*(?:meter|m)\s+hoch\b",
            r"\b\d{1,4}(?:[.,]\d+)?\s*(?:meter|m)\b",
        ], body)
        if hit:
            return _measure_answer(title, hit, "hoch")

    if normalized.startswith("wie lang"):
        hit = _first_match([
            r"\b\d{1,5}(?:[.,]\d+)?\s*(?:kilometer|km|meter|m)\s+(?:langer|lange)\b",
            r"\b\d{1,5}(?:[.,]\d+)?\s*(?:kilometer|km|meter|m)\s+lang\b",
            r"\b\d{1,5}(?:[.,]\d+)?\s*(?:kilometer|km|meter|m)\b",
        ], body)
        if hit:
            return _measure_answer(title, hit, "lang")

    if normalized.startswith("wie groß") or normalized.startswith("wie gross"):
        hit = _first_match([
            r"\b\d{1,3}(?:[.,]\d+)?\s+millionen\s+(?:km²|km2|quadratkilometer)\b",
            r"\b\d{1,3}(?:[.\s]\d{3})+\s*(?:km²|km2|m²|m2|quadratkilometer(?:n)?|hektar|ha)\b",
            r"\b\d{1,6}(?:[.,]\d+)?\s*(?:km²|km2|m²|m2|quadratkilometer(?:n)?|hektar|ha)\b",
        ], body)
        if hit:
            hit = _expand_million_area(hit)
            return f"{title}: {hit}." if title else f"{hit}."

    if normalized.startswith("wie tief"):
        hit = _first_match([
            r"\b\d{1,5}(?:[.,]\d+)?\s*(?:meter|m)\s+(?:tiefer|tiefe)\b",
            r"\b\d{1,5}(?:[.,]\d+)?\s*(?:meter|m)\s+tief\b",
            r"\b\d{1,5}(?:[.,]\d+)?\s*(?:meter|m)\b",
        ], body)
        if hit:
            return _measure_answer(title, hit, "tief")

    if normalized.startswith("wie viele") or normalized.startswith("wieviele"):
        sentence = _best_sentence(question, title, sentences)
        patterns = [
            r"\b\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+)?\s*(?:einwohnern?|mitgliedern|mitglieder|mannschaften|plätze|plaetze|spieler|personen|hektar|prozent|%)\b",
            r"\b\d{1,6}(?:[.,]\d+)?\b",
        ]
        hit = _first_match(patterns, sentence or "")
        if not hit:
            hit = _first_match(patterns, body)
        if hit:
            return f"{title}: {hit}." if title else f"{hit}."

    if normalized.startswith("wann starb"):
        hit = _first_match([
            r"(?:†|gestorben(?:e|er)?|starb)\s+(?:am\s+)?\d{1,2}\.\s+[A-ZÄÖÜ][a-zäöüß]+\s+[12][0-9]{3}\b",
            r"(?:†|gestorben(?:e|er)?|starb)[^.?!]*\b[12][0-9]{3}\b",
        ], body)
        if hit:
            return f"{title}: {hit}." if title else f"{hit}."

    if normalized.startswith("wann wurde"):
        sentence = _best_sentence(question, title, sentences)
        patterns = [
            r"\b(?:vom|am)\s+\d{1,2}\.\s+[A-ZÄÖÜ][a-zäöüß]+\s+[12][0-9]{3}\b",
            r"\b(?:im\s+Jahr\s+)?[12][0-9]{3}\b[^.?!]*(?:gegründ\w*|erbaut\w*|errichtet\w*|eröffnet\w*|gebaut\w*|fertiggestellt\w*|geboren)",
            r"\b(?:gegründ\w*|erbaut\w*|errichtet\w*|eröffnet\w*|gebaut\w*|fertiggestellt\w*|geboren)[^.?!]*\b[12][0-9]{3}\b",
        ]
        if not re.search(r"\b(?:gebaut|geboren|gegründet|eröffnet|errichtet)\b", normalized):
            patterns.append(r"\b[12][0-9]{3}\b")
        hit = _first_match(patterns, sentence or body)
        if not hit and sentence:
            hit = _first_match(patterns, body)
        if hit:
            return f"{title}: {hit}." if title else f"{hit}."

    if normalized.startswith("wo liegt") or normalized.startswith("wo steht") or normalized.startswith("wo ist"):
        sentence = _best_sentence(question, title, sentences) or sentences[0]
        return sentence

    if normalized.startswith("was ist") or normalized.startswith("wer ist"):
        return sentences[0]

    sentence = _best_sentence(question, title, sentences)
    return sentence or sentences[0]
