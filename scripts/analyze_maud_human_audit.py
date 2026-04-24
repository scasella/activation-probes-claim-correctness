from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from interp_experiment.io import read_jsonl, write_json


LABELS = ["true", "partially_true", "false"]


def _load_by_claim(path: Path) -> dict[str, dict[str, Any]]:
    return {row["claim_id"]: row for row in read_jsonl(path)}


def _cohen_kappa(pairs: list[tuple[str, str]]) -> dict[str, float | None]:
    n = len(pairs)
    if n == 0:
        return {"kappa": None, "observed_agreement": None, "expected_agreement": None}
    observed = sum(1 for a, b in pairs if a == b) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum((left[label] / n) * (right[label] / n) for label in LABELS)
    kappa = None if expected == 1.0 else (observed - expected) / (1.0 - expected)
    return {"kappa": kappa, "observed_agreement": observed, "expected_agreement": expected}


def _valid_human_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    for row in read_jsonl(path):
        label = row.get("correctness_label")
        if label in LABELS:
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MAUD human-audit labels against both LLM judges.")
    parser.add_argument("--packet-jsonl", type=Path, default=Path("data/annotations/maud_human_audit_packet.jsonl"))
    parser.add_argument("--human-labels-jsonl", type=Path, default=Path("data/annotations/maud_human_audit_labels.jsonl"))
    parser.add_argument("--judge1-labels-jsonl", type=Path, default=Path("data/annotations/maud_full_judge_annotations.jsonl"))
    parser.add_argument("--judge2-labels-jsonl", type=Path, default=Path("data/annotations/maud_full_judge_annotations_v2.jsonl"))
    parser.add_argument("--manifest-json", type=Path, default=Path("artifacts/runs/maud_human_audit_sample_manifest.json"))
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/runs/maud_human_audit_analysis.json"))
    args = parser.parse_args()

    packet = _load_by_claim(args.packet_jsonl)
    judge1 = _load_by_claim(args.judge1_labels_jsonl)
    judge2 = _load_by_claim(args.judge2_labels_jsonl)
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8")) if args.manifest_json.exists() else {}
    human_rows = _valid_human_rows(args.human_labels_jsonl)
    human_by_claim = {row["claim_id"]: row for row in human_rows}

    if not human_rows:
        write_json(
            args.output_json,
            {
                "status": "pending_human_labels",
                "n_packet_claims": len(packet),
                "n_valid_human_labels": 0,
                "message": "Human audit packet is built, but completed human labels have not been provided.",
                "sample_manifest": manifest.get("final_composition", {}),
            },
        )
        print(f"Wrote pending audit analysis to {args.output_json}")
        return

    claim_ids = sorted(set(packet) & set(human_by_claim) & set(judge1) & set(judge2))
    pairs_gpt = [(judge1[claim_id]["correctness_label"], human_by_claim[claim_id]["correctness_label"]) for claim_id in claim_ids]
    pairs_kimi = [(judge2[claim_id]["correctness_label"], human_by_claim[claim_id]["correctness_label"]) for claim_id in claim_ids]
    agreement_ids = [claim_id for claim_id in claim_ids if judge1[claim_id]["correctness_label"] == judge2[claim_id]["correctness_label"]]
    disagreement_ids = [claim_id for claim_id in claim_ids if judge1[claim_id]["correctness_label"] != judge2[claim_id]["correctness_label"]]
    disagreement_matches = Counter()
    comparison_rows = []
    for claim_id in claim_ids:
        h = human_by_claim[claim_id]["correctness_label"]
        g = judge1[claim_id]["correctness_label"]
        k = judge2[claim_id]["correctness_label"]
        if g != k:
            if h == g and h != k:
                disagreement_matches["human_matches_gpt54"] += 1
            elif h == k and h != g:
                disagreement_matches["human_matches_kimi"] += 1
            elif h == g == k:
                disagreement_matches["human_matches_both"] += 1
            else:
                disagreement_matches["human_matches_neither"] += 1
        comparison_rows.append(
            {
                "claim_id": claim_id,
                "audit_id": human_by_claim[claim_id].get("audit_id"),
                "gpt54_label": g,
                "kimi_label": k,
                "human_label": h,
                "judge_relation": "agreement" if g == k else "disagreement",
                "human_justification": human_by_claim[claim_id].get("justification", ""),
            }
        )

    def agreement_rate(ids: list[str], judge: dict[str, dict[str, Any]]) -> float | None:
        if not ids:
            return None
        return sum(judge[claim_id]["correctness_label"] == human_by_claim[claim_id]["correctness_label"] for claim_id in ids) / len(ids)

    annotator_ids = sorted({str(row.get("annotator_id", "")) for row in human_rows if row.get("annotator_id")})
    payload = {
        "status": "complete" if len(claim_ids) == len(packet) else "partial",
        "n_packet_claims": len(packet),
        "n_valid_human_labels": len(claim_ids),
        "annotator_ids": annotator_ids,
        "human_label_counts": dict(Counter(human_by_claim[claim_id]["correctness_label"] for claim_id in claim_ids)),
        "gpt54_vs_human": _cohen_kappa(pairs_gpt),
        "kimi_vs_human": _cohen_kappa(pairs_kimi),
        "agreement_stratum": {
            "n": len(agreement_ids),
            "gpt54_human_agreement_rate": agreement_rate(agreement_ids, judge1),
            "kimi_human_agreement_rate": agreement_rate(agreement_ids, judge2),
        },
        "disagreement_stratum": {
            "n": len(disagreement_ids),
            "human_arbitration_counts": dict(disagreement_matches),
            "gpt54_human_agreement_rate": agreement_rate(disagreement_ids, judge1),
            "kimi_human_agreement_rate": agreement_rate(disagreement_ids, judge2),
        },
        "sample_manifest": manifest.get("final_composition", {}),
        "comparison_rows": comparison_rows,
    }
    write_json(args.output_json, payload)
    print(f"Wrote audit analysis to {args.output_json}")


if __name__ == "__main__":
    main()
