from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from interp_experiment.io import read_jsonl, write_json, write_jsonl


RUBRIC = (
    "Assign correctness_label=true if the extracted claim is fully supported by the contract "
    "excerpt in context of the question and Llama answer; correctness_label=false if it is "
    "contradicted or unsupported; correctness_label=partially_true if it captures part of the "
    "truth but omits or distorts a material condition, carveout, timing requirement, or qualifier. "
    "Use only the excerpt, question, answer, and claim shown here. Add a brief justification."
)


def _load_by_claim(path: Path) -> dict[str, dict[str, Any]]:
    return {row["claim_id"]: row for row in read_jsonl(path)}


def _sample(pool: list[str], n: int, rng: random.Random, selected: set[str]) -> list[str]:
    candidates = [claim_id for claim_id in pool if claim_id not in selected]
    rng.shuffle(candidates)
    picks = candidates[:n]
    selected.update(picks)
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the blinded 30-claim MAUD human-audit packet.")
    parser.add_argument("--judge1-labels-jsonl", type=Path, default=Path("data/annotations/maud_full_judge_annotations.jsonl"))
    parser.add_argument("--judge2-labels-jsonl", type=Path, default=Path("data/annotations/maud_full_judge_annotations_v2.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/annotations/maud_human_audit_packet.jsonl"))
    parser.add_argument("--label-template-jsonl", type=Path, default=Path("data/annotations/maud_human_audit_labels.jsonl"))
    parser.add_argument("--manifest-json", type=Path, default=Path("artifacts/runs/maud_human_audit_sample_manifest.json"))
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=240424)
    args = parser.parse_args()

    judge1 = _load_by_claim(args.judge1_labels_jsonl)
    judge2 = _load_by_claim(args.judge2_labels_jsonl)
    shared_ids = sorted(set(judge1) & set(judge2))
    if len(shared_ids) < args.sample_size:
        raise ValueError("Not enough shared labels to build audit packet")

    agreement = [claim_id for claim_id in shared_ids if judge1[claim_id]["correctness_label"] == judge2[claim_id]["correctness_label"]]
    disagreement = [claim_id for claim_id in shared_ids if judge1[claim_id]["correctness_label"] != judge2[claim_id]["correctness_label"]]
    both_true = [claim_id for claim_id in agreement if judge1[claim_id]["correctness_label"] == "true"]
    both_false_or_partial = [
        claim_id
        for claim_id in agreement
        if judge1[claim_id]["correctness_label"] in {"false", "partially_true"}
    ]

    rng = random.Random(args.seed)
    selected: set[str] = set()
    selection_sources: dict[str, list[str]] = {}
    selection_sources["agreement"] = _sample(agreement, 10, rng, selected)
    selection_sources["disagreement"] = _sample(disagreement, 10, rng, selected)
    selection_sources["both_true"] = _sample(both_true, 5, rng, selected)
    selection_sources["both_false_or_partially_true"] = _sample(both_false_or_partial, 5, rng, selected)

    if len(selected) < args.sample_size:
        remainder = [claim_id for claim_id in shared_ids if claim_id not in selected]
        rng.shuffle(remainder)
        top_up = remainder[: args.sample_size - len(selected)]
        selected.update(top_up)
        selection_sources["top_up"] = top_up

    selected_ids = sorted(selected)
    rng.shuffle(selected_ids)
    packet_rows = []
    template_rows = []
    manifest_rows = []
    for idx, claim_id in enumerate(selected_ids, start=1):
        source = judge1[claim_id]
        audit_id = f"maud-human-audit-{idx:03d}"
        packet_rows.append(
            {
                "audit_id": audit_id,
                "claim_id": claim_id,
                "source_corpus": source["source_corpus"],
                "task_family": source["task_family"],
                "contract_id": source["contract_id"],
                "question_text": source["question_text"],
                "excerpt_text": source["excerpt_text"],
                "llama_answer_text": source["llama_answer_text"],
                "claim_text": source["claim_text"],
                "rubric": RUBRIC,
            }
        )
        template_rows.append(
            {
                "audit_id": audit_id,
                "claim_id": claim_id,
                "annotator_id": "",
                "annotator_background": "",
                "correctness_label": "",
                "justification": "",
                "annotation_seconds": None,
            }
        )
        manifest_rows.append(
            {
                "audit_id": audit_id,
                "claim_id": claim_id,
                "judge1_correctness_label": judge1[claim_id]["correctness_label"],
                "judge2_correctness_label": judge2[claim_id]["correctness_label"],
                "judge_relation": "agreement"
                if judge1[claim_id]["correctness_label"] == judge2[claim_id]["correctness_label"]
                else "disagreement",
                "agreement_label": judge1[claim_id]["correctness_label"]
                if judge1[claim_id]["correctness_label"] == judge2[claim_id]["correctness_label"]
                else None,
            }
        )

    write_jsonl(args.output_jsonl, packet_rows)
    if not args.label_template_jsonl.exists():
        write_jsonl(args.label_template_jsonl, template_rows)
    composition = {
        "sample_size": len(packet_rows),
        "seed": args.seed,
        "selection_sources": selection_sources,
        "final_composition": {
            "judge_relation": dict(Counter(row["judge_relation"] for row in manifest_rows)),
            "agreement_label": dict(Counter(row["agreement_label"] for row in manifest_rows if row["agreement_label"])),
            "judge_pair": dict(
                Counter(
                    f"{row['judge1_correctness_label']}__{row['judge2_correctness_label']}"
                    for row in manifest_rows
                )
            ),
        },
        "packet_is_blinded": True,
        "packet_excludes_judge_labels": True,
        "rows": manifest_rows,
    }
    write_json(args.manifest_json, composition)
    print(f"Wrote audit packet to {args.output_jsonl}")
    print(f"Wrote audit manifest to {args.manifest_json}")


if __name__ == "__main__":
    main()
