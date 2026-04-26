from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

from interp_experiment.io import read_jsonl, write_json, write_jsonl


FALLBACK_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _load_segmenter() -> tuple[Any, str]:
    try:
        import spacy

        try:
            return spacy.load("en_core_web_sm"), "spacy_en_core_web_sm"
        except OSError:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            return nlp, "spacy_blank_sentencizer_fallback"
    except ImportError:
        return None, "regex_fallback"


def _segments(text: str, segmenter: Any, segmenter_name: str) -> list[tuple[str, int, int]]:
    if segmenter is None:
        pieces = [piece.strip() for piece in FALLBACK_SENTENCE_RE.split(text) if piece.strip()]
    else:
        doc = segmenter(text)
        pieces = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    output: list[tuple[str, int, int]] = []
    cursor = 0
    for piece in pieces:
        start = text.find(piece, cursor)
        if start < 0:
            start = text.find(piece)
        if start < 0:
            continue
        end = start + len(piece)
        cursor = end
        output.append((piece, start, end))
    if not output and text.strip():
        stripped = text.strip()
        start = text.find(stripped)
        output.append((stripped, start, start + len(stripped)))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment matched FELM Llama answers into sentence-level claims.")
    parser.add_argument("--examples-jsonl", type=Path, default=Path("data/felm/felm_wk_matched_examples.jsonl"))
    parser.add_argument("--segments-jsonl", type=Path, default=Path("data/felm/felm_wk_matched_segments.jsonl"))
    parser.add_argument("--summary-json", type=Path, default=Path("artifacts/runs/felm_wk_matched_segmentation_summary.json"))
    args = parser.parse_args()

    segmenter, segmenter_name = _load_segmenter()
    examples = list(read_jsonl(args.examples_jsonl))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for example in examples:
        answer = example["llama_answer_text"]
        spans = _segments(answer, segmenter, segmenter_name)
        if not spans:
            failures.append({"example_id": example["example_id"], "reason": "no_segments"})
            continue
        for idx, (text, start, end) in enumerate(spans):
            rows.append(
                {
                    "claim_id": f"{example['example_id']}-seg-{idx:02d}",
                    "example_id": example["example_id"],
                    "claim_text": text,
                    "char_start": start,
                    "char_end": end,
                    "segment_index": idx,
                    "domain": example["contract_group"],
                    "source": "llama_matched_generation",
                    "annotation_version": "felm_matched_sentence_v1",
                }
            )
    write_jsonl(args.segments_jsonl, rows)
    summary = {
        "n_examples": len(examples),
        "n_segments": len(rows),
        "segments_by_example": dict(Counter(row["example_id"] for row in rows)),
        "segmenter": segmenter_name,
        "n_failures": len(failures),
        "failures": failures,
    }
    write_json(args.summary_json, summary)
    print(f"Wrote matched FELM segments to {args.segments_jsonl}")
    print(f"Wrote segmentation summary to {args.summary_json}")


if __name__ == "__main__":
    main()
