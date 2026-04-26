from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from interp_experiment.codex_app_server_client import CodexAppServerClient
from interp_experiment.io import read_jsonl, write_json, write_jsonl
from interp_experiment.schemas import ExampleRow


def _judge_schema(expected_claim_ids: list[str]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "minItems": len(expected_claim_ids),
                "maxItems": len(expected_claim_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string", "enum": expected_claim_ids},
                        "correctness_label": {"type": "string", "enum": ["true", "false", "partially_true"]},
                        "evidence_text": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": ["claim_id", "correctness_label", "evidence_text", "notes"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    }


def _reference_block(reference_row: dict[str, Any], *, max_ref_chars: int, max_total_chars: int) -> str:
    chunks: list[str] = []
    total = 0
    for idx, ref in enumerate(reference_row.get("references", []), start=1):
        content = str(ref.get("content") or "").strip()
        if not content:
            continue
        excerpt = content[:max_ref_chars]
        chunk = f"[Reference {idx}] URL: {ref.get('url', '')}\n{excerpt}"
        if total + len(chunk) > max_total_chars:
            remaining = max_total_chars - total
            if remaining <= 200:
                break
            chunk = chunk[:remaining]
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_total_chars:
            break
    return "\n\n".join(chunks)


def _claims_block(segments: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {segment['claim_id']}: {segment['claim_text']}" for segment in segments)


def _judge_prompt(
    example: ExampleRow,
    segments: list[dict[str, Any]],
    reference_row: dict[str, Any],
    *,
    max_ref_chars: int,
    max_total_ref_chars: int,
) -> str:
    references = _reference_block(reference_row, max_ref_chars=max_ref_chars, max_total_chars=max_total_ref_chars)
    return (
        "You are a factuality judge for a research experiment. Use only the provided reference evidence. "
        "Do not use outside knowledge. Score each fixed answer segment independently.\n\n"
        "Labels:\n"
        "- true: the segment is fully supported by the references.\n"
        "- partially_true: the segment is directionally supported but overstates, omits an important qualifier, or mixes supported and unsupported content.\n"
        "- false: the segment is contradicted by the references or not supported by the provided evidence.\n\n"
        "If the references do not contain enough evidence for a segment, label it false rather than guessing.\n\n"
        f"Question:\n{example.question_text}\n\n"
        f"Llama-generated answer:\n{example.llama_answer_text}\n\n"
        f"Reference evidence:\n{references if references else '[No usable reference evidence]'}\n\n"
        f"Fixed answer segments:\n{_claims_block(segments)}\n\n"
        "Return strict JSON only with one entry per segment."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge matched FELM generated segments against cached FELM references.")
    parser.add_argument("--examples-jsonl", type=Path, default=Path("data/felm/felm_wk_matched_examples.jsonl"))
    parser.add_argument("--segments-jsonl", type=Path, default=Path("data/felm/felm_wk_matched_segments.jsonl"))
    parser.add_argument("--reference-cache-jsonl", type=Path, default=Path("data/felm/felm_wk_reference_cache.jsonl"))
    parser.add_argument("--labels-jsonl", type=Path, default=Path("data/annotations/felm_wk_matched_judge_labels.jsonl"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/annotations/judge_llm_raw/felm_wk_matched"))
    parser.add_argument("--summary-json", type=Path, default=Path("artifacts/runs/felm_wk_matched_judge_summary.json"))
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--turn-timeout-sec", type=float, default=180.0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-ref-chars", type=int, default=8000)
    parser.add_argument("--max-total-ref-chars", type=int, default=24000)
    args = parser.parse_args()

    examples = [ExampleRow.from_dict(row) for row in read_jsonl(args.examples_jsonl)]
    if args.max_examples:
        examples = examples[: args.max_examples]
    segments_by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.segments_jsonl):
        segments_by_example[row["example_id"]].append(row)
    references = {row["example_id"]: row for row in read_jsonl(args.reference_cache_jsonl)}

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    excluded_no_reference: list[str] = []
    processed = 0
    reused = 0
    with CodexAppServerClient(cwd=Path.cwd(), model=args.model) as client:
        for example in examples:
            segments = sorted(segments_by_example[example.example_id], key=lambda item: item["segment_index"])
            reference_row = references.get(example.example_id)
            if not segments:
                failures.append({"example_id": example.example_id, "stage": "precheck", "error": "no_segments"})
                continue
            if reference_row is None or not any(ref.get("content") for ref in reference_row.get("references", [])):
                excluded_no_reference.append(example.example_id)
                continue
            raw_path = args.raw_dir / f"{example.example_id}.json"
            if args.skip_existing and raw_path.exists():
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
                reused += 1
            else:
                try:
                    raw_text = client.run_prompt(
                        _judge_prompt(
                            example,
                            segments,
                            reference_row,
                            max_ref_chars=args.max_ref_chars,
                            max_total_ref_chars=args.max_total_ref_chars,
                        ),
                        output_schema=_judge_schema([segment["claim_id"] for segment in segments]),
                        turn_timeout_sec=args.turn_timeout_sec,
                    )
                    payload = json.loads(raw_text)
                    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    processed += 1
                except Exception as exc:
                    failures.append({"example_id": example.example_id, "stage": "judge", "error": f"{type(exc).__name__}: {exc}"})
                    continue
            by_claim_id = {segment["claim_id"]: segment for segment in segments}
            try:
                items = payload["claims"]
            except Exception as exc:
                failures.append({"example_id": example.example_id, "stage": "parse", "error": f"{type(exc).__name__}: {exc}"})
                continue
            for item in items:
                segment = by_claim_id.get(item.get("claim_id"))
                if segment is None:
                    failures.append({"example_id": example.example_id, "stage": "parse", "error": f"unknown_claim_id:{item.get('claim_id')}"})
                    continue
                rows.append(
                    {
                        "claim_id": segment["claim_id"],
                        "example_id": example.example_id,
                        "correctness_label": item["correctness_label"],
                        "load_bearing_label": "yes",
                        "flip_evidence_text": item.get("evidence_text", ""),
                        "notes": item.get("notes", ""),
                        "label_source": "judge_llm_reference_proxy",
                        "judge_model": args.model,
                        "annotation_version": "felm_matched_reference_judge_v1",
                    }
                )
            print(f"FELM_MATCHED_JUDGED {example.example_id} claims={len(items)}", flush=True)

    write_jsonl(args.labels_jsonl, rows)
    n_segments_requested = sum(len(segments_by_example[example.example_id]) for example in examples)
    summary = {
        "n_examples": len(examples),
        "n_segments_requested": n_segments_requested,
        "n_labeled_segments": len(rows),
        "n_examples_processed_this_run": processed,
        "n_examples_reused_existing": reused,
        "n_examples_excluded_no_reference": len(excluded_no_reference),
        "excluded_no_reference": excluded_no_reference,
        "reference_exclusion_rate_by_example": len(excluded_no_reference) / len(examples) if examples else None,
        "n_failures": len(failures),
        "failures": failures,
        "judge_model": args.model,
        "label_source": "judge_llm_reference_proxy",
        "annotation_version": "felm_matched_reference_judge_v1",
        "note": "Labels are produced by a judge LLM using cached FELM reference evidence; they are not FELM human labels.",
    }
    write_json(args.summary_json, summary)
    print(f"Wrote matched judge labels to {args.labels_jsonl}")
    print(f"Wrote matched judge summary to {args.summary_json}")


if __name__ == "__main__":
    main()
