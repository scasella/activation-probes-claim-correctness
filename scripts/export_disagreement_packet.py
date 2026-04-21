from __future__ import annotations

import argparse
from pathlib import Path

from interp_experiment.io import read_json, write_csv, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export disagreement examples from an annotation agreement JSON for adjudication.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    payload = read_json(args.input_json)
    disagreements = payload.get("disagreements", [])
    flattened = []
    for item in disagreements:
        flattened.append(
            {
                "claim_id": item["claim_id"],
                "example_id": item["example_id"],
                "question_text": item["question_text"],
                "claim_text": item["claim_text"],
                "llama_answer_text": item["llama_answer_text"],
                "disagreement_kinds": ",".join(item["disagreement_kinds"]),
                "annotator_a_id": item["annotator_a"]["annotator_id"],
                "annotator_a_correctness": item["annotator_a"]["correctness_label"],
                "annotator_a_load_bearing": item["annotator_a"]["load_bearing_label"],
                "annotator_a_flip_evidence": item["annotator_a"]["flip_evidence_text"],
                "annotator_a_notes": item["annotator_a"]["notes"],
                "annotator_b_id": item["annotator_b"]["annotator_id"],
                "annotator_b_correctness": item["annotator_b"]["correctness_label"],
                "annotator_b_load_bearing": item["annotator_b"]["load_bearing_label"],
                "annotator_b_flip_evidence": item["annotator_b"]["flip_evidence_text"],
                "annotator_b_notes": item["annotator_b"]["notes"],
                "adjudicated_correctness_label": "",
                "adjudicated_load_bearing_label": "",
                "adjudicated_flip_evidence_text": "",
                "adjudicator_notes": "",
            }
        )
    write_jsonl(args.output_jsonl, flattened)
    if args.output_csv is not None:
        write_csv(args.output_csv, flattened)
    print(f"Wrote {len(flattened)} disagreement rows to {args.output_jsonl}")
    if args.output_csv is not None:
        print(f"Wrote disagreement CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
