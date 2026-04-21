from __future__ import annotations

import argparse
from pathlib import Path

from interp_experiment.data.annotation import compute_annotation_agreement, load_annotation_rows
from interp_experiment.io import write_json
from interp_experiment.utils import ensure_parent


def _markdown_report(payload: dict[str, object]) -> str:
    completion = payload["completion"]
    agreement = payload.get("agreement", {})
    gate = payload["gate"]
    disagreements = payload["disagreements"]
    lines = [
        "# Annotation Pilot Agreement Report",
        "",
        "## Completion",
        "",
        f"- Rows: {completion['n_rows']}",
        f"- Completed rows: {completion['n_completed_rows']}",
        f"- Incomplete rows: {completion['n_incomplete_rows']}",
        f"- Rows by annotator: {completion['rows_by_annotator']}",
        f"- Completed rows by annotator: {completion['completed_rows_by_annotator']}",
        "",
        "## Agreement",
        "",
    ]
    if agreement:
        lines.extend(
            [
                f"- Complete claim pairs: {payload['n_complete_pairs']}",
                f"- Correctness kappa: {agreement['correctness_kappa']:.4f}",
                f"- Correctness binary kappa: {agreement['correctness_binary_kappa']:.4f}",
                f"- Load-bearing kappa: {agreement['load_bearing_kappa']:.4f}",
            ]
        )
    else:
        lines.append("- Not enough completed annotation pairs to compute agreement.")
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- Status: {gate['status']}",
            f"- Next action: {gate['next_action']}",
            f"- Load-bearing threshold: {gate['load_bearing_threshold']}",
            f"- Attempt index: {gate['attempt_index']}",
            "",
            "## Disagreements",
            "",
        ]
    )
    if disagreements:
        for item in disagreements:
            lines.extend(
                [
                    f"- Claim `{item['claim_id']}` ({', '.join(item['disagreement_kinds'])})",
                    f"  Question: {item['question_text']}",
                    f"  Claim: {item['claim_text']}",
                    f"  Annotator {item['annotator_a']['annotator_id']}: correctness={item['annotator_a']['correctness_label']}, load={item['annotator_a']['load_bearing_label']}",
                    f"  Annotator {item['annotator_b']['annotator_id']}: correctness={item['annotator_b']['correctness_label']}, load={item['annotator_b']['load_bearing_label']}",
                ]
            )
    else:
        lines.append("- No disagreements found in completed claim pairs.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate completed annotation pilot files into JSON and Markdown reports.")
    parser.add_argument("--input-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--attempt-index", type=int, default=1)
    parser.add_argument("--load-bearing-threshold", type=float, default=0.6)
    args = parser.parse_args()

    rows = load_annotation_rows(args.input_jsonl)
    payload = compute_annotation_agreement(
        rows,
        load_bearing_threshold=args.load_bearing_threshold,
        attempt_index=args.attempt_index,
    )
    write_json(args.output_json, payload)
    ensure_parent(args.output_md)
    args.output_md.write_text(_markdown_report(payload), encoding="utf-8")
    print(f"Wrote agreement JSON to {args.output_json}")
    print(f"Wrote agreement report to {args.output_md}")


if __name__ == "__main__":
    main()
