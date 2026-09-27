"""Run a broader black-box check against the desktop chatbot."""

import argparse
import json
from pathlib import Path

from .chat import Chat
from .sd_index import SDIndex


CASES = [
    {"q": "Hallo", "contains": "plaudern", "mode": "Smalltalk"},
    {"q": "Wer bist du?", "contains": "ESP32-S3", "mode": "Smalltalk"},
    {"q": "Was kannst du?", "contains": "Wikipedia", "mode": "Smalltalk"},
    {"q": "Wie geht es dir?", "contains": "SD-Karte", "mode": "Smalltalk"},
    {"q": "Danke", "contains": "Gern", "mode": "Smalltalk"},
    {"q": "Was ist ein Transistor?", "contains": "Halbleiter", "source": "Transistor"},
    {"q": "Wie hoch ist der Eiffelturm?", "contains": "330 Meter hoch", "source": "Eiffelturm"},
    {"q": "Wie lang ist der Rhein?", "contains": "1232,7 km lang", "source": "Rhein"},
    {"q": "Wann wurde die Berliner Mauer gebaut?", "contains": "1961", "source": "Berliner Mauer"},
    {"q": "Wo liegt München?", "contains": "Bayern", "source": "München"},
    {"q": "Was war das Thema?", "contains": "München", "mode": "Smalltalk"},
    {"q": "/neu", "contains": "Neues Gespräch"},
    {"q": "Was war das Thema?", "contains": "kein gespeichertes Thema", "mode": "Smalltalk"},
    {"q": "zzzxxy unbekannteswort", "contains": "keinen passenden Text"},
]


def _knowledge_cases(path, limit):
    samples = json.loads(path.read_text(encoding="utf-8"))[:limit]
    return [
        {"q": sample["frage"], "contains_any": sample["antwort"],
         "source": sample["titel"], "group": sample.get("gruppe")}
        for sample in samples
    ]


def _case_ok(case, result):
    expected = case.get("contains")
    if expected is not None and expected.casefold() not in result["answer"].casefold():
        return False
    expected_any = case.get("contains_any")
    if expected_any and not any(value.casefold() in result["answer"].casefold()
                               for value in expected_any):
        return False
    if "mode" in case and result.get("mode") != case["mode"]:
        return False
    if "source" in case and result.get("source") != case["source"]:
        return False
    return True


def run(index_path, cases):
    rows = []
    with SDIndex(index_path) as index:
        chat = Chat(index)
        for case in cases:
            result = chat.answer(case["q"])
            ok = _case_ok(case, result)
            rows.append({"ok": ok, "case": case, "result": result})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path(__file__).parent / "wiki_sd")
    parser.add_argument("--knowledge", type=Path,
                        default=Path(__file__).resolve().parents[1] / "chat_fragen.json")
    parser.add_argument("--total", type=int, default=len(CASES))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    cases = list(CASES)
    if args.total > len(cases):
        cases.extend(_knowledge_cases(args.knowledge, args.total - len(cases)))
    rows = run(args.index, cases)
    passed = sum(row["ok"] for row in rows)
    summary = {"passed": passed, "total": len(rows)}
    if args.json:
        print(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    else:
        print(f"{passed}/{len(rows)} bestanden")
        for row in rows:
            mark = "OK" if row["ok"] else "FAIL"
            print(f"{mark:4} {row['case']['q']} -> {row['result']['answer'][:100]}")
    if passed != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
