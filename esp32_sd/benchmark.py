"""Host retrieval measurements; these are not ESP32 timing or answer accuracy."""
import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path

from .sd_index import SDIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path(__file__).parent / "wiki_sd")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text())
    rows = []
    tracemalloc.start()
    with SDIndex(args.index) as index:
        for sample in questions:
            started = time.perf_counter()
            hits = index.suche(sample["frage"], anzahl=2)
            elapsed = time.perf_counter() - started
            titles = [text.partition(":")[0] for _, text in hits]
            rows.append({"question": sample["frage"], "seconds_host": elapsed,
                         "expected_title": sample["titel"], "titles": titles,
                         "title_found": sample["titel"] in titles})
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report = {"scope": "Host Python retrieval only, no generated-answer evaluation",
              "note": "Title match tests retrieval, not whether a paragraph answers the question. Python traced allocations exclude interpreter/native overhead.",
              "questions": len(rows), "title_hits_top2": sum(r["title_found"] for r in rows),
              "median_seconds_host": statistics.median(r["seconds_host"] for r in rows) if rows else None,
              "peak_python_traced_bytes": peak, "results": rows}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
