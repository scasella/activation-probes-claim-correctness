from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from interp_experiment.codex_app_server_client import CodexAppServerClient
from interp_experiment.io import read_jsonl, write_json, write_jsonl
from interp_experiment.schemas import ExampleRow


LABELS = ["true", "partially_true", "false", "not_enough_evidence"]
STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "because",
    "been",
    "being",
    "between",
    "could",
    "does",
    "following",
    "from",
    "have",
    "into",
    "more",
    "only",
    "over",
    "same",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "under",
    "using",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _judge_schema(claim_id: str) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "claim_id": {"type": "string", "enum": [claim_id]},
            "correctness_label": {"type": "string", "enum": LABELS},
            "justification": {"type": "string", "minLength": 1},
            "evidence_text": {"type": "string"},
        },
        "required": ["claim_id", "correctness_label", "justification", "evidence_text"],
        "additionalProperties": False,
    }


def _parse_json_payload(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _validate_payload(payload: dict[str, Any], claim_id: str) -> None:
    if payload.get("claim_id") != claim_id:
        raise ValueError(f"claim_id mismatch: expected {claim_id!r}, got {payload.get('claim_id')!r}")
    if payload.get("correctness_label") not in LABELS:
        raise ValueError(f"invalid correctness_label: {payload.get('correctness_label')!r}")
    if not str(payload.get("justification") or "").strip():
        raise ValueError("missing justification")
    if "evidence_text" not in payload:
        raise ValueError("missing evidence_text")


def _query_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{3,}", text.lower())
        if term not in STOPWORDS and not term.isdigit()
    }


def _select_relevant_excerpt(content: str, query: str, *, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    terms = _query_terms(query)
    if not terms:
        return content[:max_chars]
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", content) if part.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", content) if part.strip()]
    scored: list[tuple[int, int, str]] = []
    for idx, paragraph in enumerate(paragraphs):
        lower = paragraph.lower()
        score = sum(lower.count(term) for term in terms)
        if score:
            scored.append((score, idx, paragraph))
    if not scored:
        return content[:max_chars]
    selected = sorted(sorted(scored, reverse=True)[:8], key=lambda item: item[1])
    excerpts: list[str] = []
    used = 0
    for _, _, paragraph in selected:
        if used >= max_chars:
            break
        remaining = max_chars - used
        excerpt = paragraph[:remaining]
        excerpts.append(excerpt)
        used += len(excerpt) + 2
    return "\n\n".join(excerpts)[:max_chars]


def _reference_block(reference_row: dict[str, Any], *, max_total_ref_chars: int, query: str) -> str:
    chunks: list[str] = []
    total = 0
    for idx, ref in enumerate(reference_row.get("references", []), start=1):
        content = str(ref.get("content") or "").strip()
        if not content:
            continue
        url = str(ref.get("url") or "")
        source = str(ref.get("source") or "")
        header = f"[Reference {idx}] URL: {url}\nSource: {source}\n"
        remaining = max_total_ref_chars - total - len(header)
        if remaining <= 200:
            break
        excerpt = _select_relevant_excerpt(content, query, max_chars=remaining)
        chunks.append(f"{header}{excerpt}")
        total += len(header) + len(excerpt)
        if total >= max_total_ref_chars:
            break
    return "\n\n".join(chunks)


def _judge_prompt(
    example: ExampleRow,
    segment: dict[str, Any],
    reference_row: dict[str, Any],
    *,
    max_total_ref_chars: int,
    strict_retry: bool,
) -> str:
    references = _reference_block(
        reference_row,
        max_total_ref_chars=max_total_ref_chars,
        query=f"{example.question_text}\n{segment['claim_text']}",
    )
    retry_line = (
        "\nThis is a retry after malformed output. Return exactly one JSON object matching the schema; "
        "do not include Markdown, extra prose, or labels outside the allowed set.\n"
        if strict_retry
        else ""
    )
    return (
        "You are a factuality judge for a reproducible research experiment. "
        "Use ONLY the provided reference text. Do not use outside knowledge or web search.\n\n"
        "Given the following reference text and a claim about it, label the claim's correctness based ONLY on what the reference text says.\n\n"
        f"Question:\n{example.question_text}\n\n"
        f"Llama-generated answer, for context only:\n{example.llama_answer_text}\n\n"
        f"Reference text:\n{references if references else '[No usable reference text]'}\n\n"
        f"Claim ID:\n{segment['claim_id']}\n\n"
        f"Claim:\n{segment['claim_text']}\n\n"
        "Output one of:\n"
        "- true: the reference clearly supports the claim\n"
        "- partially_true: the reference supports part of the claim but contradicts or omits another part that is presented as fact\n"
        "- false: the reference clearly contradicts the claim\n"
        "- not_enough_evidence: the reference does not contain enough information to determine whether the claim is true or false\n\n"
        "If the claim may be true in the world but the reference does not contain enough information to evaluate it, use not_enough_evidence. "
        "Do not convert missing evidence into false.\n\n"
        "Provide a one-sentence justification for your label and a short evidence_text quote or paraphrase from the reference when available. "
        "Return strict JSON only with keys: claim_id, correctness_label, justification, evidence_text."
        f"{retry_line}"
    )


def _load_existing_claim_ids(labels_path: Path) -> set[str]:
    if not labels_path.exists():
        return set()
    return {str(row.get("claim_id")) for row in read_jsonl(labels_path) if row.get("claim_id")}


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (str(item.get("example_id")), int(item.get("segment_index") or 0), str(item.get("claim_id"))),
    )


def _payload_from_raw(raw_path: Path, claim_id: str) -> dict[str, Any] | None:
    if not raw_path.exists():
        return None
    try:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for attempt in reversed(raw.get("attempts", [])):
        parsed = attempt.get("parsed")
        if not isinstance(parsed, dict):
            continue
        try:
            _validate_payload(parsed, claim_id)
        except Exception:
            continue
        return parsed
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge FELM matched pilot segments with a four-label evidence rubric.")
    parser.add_argument("--examples-jsonl", type=Path, default=Path("data/felm/felm_wk_matched_pilot_examples.jsonl"))
    parser.add_argument("--segments-jsonl", type=Path, default=Path("data/felm/felm_wk_matched_pilot_segments.jsonl"))
    parser.add_argument(
        "--reference-cache-jsonl",
        type=Path,
        default=Path("data/felm/felm_wk_matched_pilot_reference_cache_v2.jsonl"),
    )
    parser.add_argument(
        "--labels-jsonl",
        type=Path,
        default=Path("data/annotations/felm_wk_matched_pilot_judge_labels_v2.jsonl"),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/annotations/judge_llm_raw/felm_wk_matched_pilot_v2"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("artifacts/runs/felm_wk_matched_pilot_judge_summary_v2.json"),
    )
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--turn-timeout-sec", type=float, default=300.0)
    parser.add_argument("--max-total-ref-chars", type=int, default=16000)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--claim-id", action="append", default=[], help="Restrict to one claim id; can be repeated.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--reuse-raw", action="store_true", help="Reuse valid parsed raw outputs from raw-dir.")
    args = parser.parse_args()

    examples = {row["example_id"]: ExampleRow.from_dict(row) for row in read_jsonl(args.examples_jsonl)}
    segments = list(read_jsonl(args.segments_jsonl))
    if args.claim_id:
        wanted = set(args.claim_id)
        segments = [segment for segment in segments if segment.get("claim_id") in wanted]
    if args.max_segments:
        segments = segments[: args.max_segments]
    references = {row["example_id"]: row for row in read_jsonl(args.reference_cache_jsonl)}
    existing_claim_ids = _load_existing_claim_ids(args.labels_jsonl) if args.skip_existing else set()
    existing_rows = [row for row in read_jsonl(args.labels_jsonl)] if args.skip_existing and args.labels_jsonl.exists() else []

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = list(existing_rows)
    failures: list[dict[str, Any]] = []
    skipped_existing = 0
    processed = 0
    retries_used = 0
    reused_raw = 0

    with CodexAppServerClient(cwd=Path.cwd(), model=args.model) as client:
        for segment in segments:
            claim_id = str(segment["claim_id"])
            if claim_id in existing_claim_ids:
                skipped_existing += 1
                continue
            example = examples.get(str(segment["example_id"]))
            reference_row = references.get(str(segment["example_id"]))
            if example is None:
                failures.append({"claim_id": claim_id, "stage": "precheck", "error": "missing_example"})
                continue
            if reference_row is None or not any(ref.get("content") for ref in reference_row.get("references", [])):
                failures.append({"claim_id": claim_id, "stage": "precheck", "error": "missing_reference"})
                continue

            raw_path = args.raw_dir / f"{_safe_name(claim_id)}.json"
            payload = _payload_from_raw(raw_path, claim_id) if args.reuse_raw else None
            if payload is not None:
                reused_raw += 1
                attempts = json.loads(raw_path.read_text(encoding="utf-8")).get("attempts", [])
            else:
                attempts = []
            if payload is not None:
                rows.append(
                    {
                        "claim_id": claim_id,
                        "example_id": segment["example_id"],
                        "correctness_label": payload["correctness_label"],
                        "justification": payload["justification"],
                        "evidence_text": payload.get("evidence_text", ""),
                        "claim_text": segment["claim_text"],
                        "segment_index": segment.get("segment_index"),
                        "load_bearing_label": "yes",
                        "flip_evidence_text": payload.get("evidence_text", ""),
                        "notes": payload["justification"],
                        "label_source": "judge_llm_reference_proxy_v2",
                        "judge_model": args.model,
                        "annotation_version": "felm_matched_reference_judge_v2",
                    }
                )
                write_jsonl(args.labels_jsonl, _sort_rows(rows))
                print(f"FELM_MATCHED_V2_REUSED_RAW {claim_id} label={payload['correctness_label']}", flush=True)
                continue

            attempts: list[dict[str, Any]] = []
            payload = None
            last_error: str | None = None
            for attempt_idx in range(2):
                strict_retry = attempt_idx == 1
                prompt = _judge_prompt(
                    example,
                    segment,
                    reference_row,
                    max_total_ref_chars=args.max_total_ref_chars,
                    strict_retry=strict_retry,
                )
                try:
                    raw_text = ""
                    raw_text = client.run_prompt(
                        prompt,
                        output_schema=_judge_schema(claim_id),
                        turn_timeout_sec=args.turn_timeout_sec,
                    )
                    parsed = _parse_json_payload(raw_text)
                    _validate_payload(parsed, claim_id)
                    payload = parsed
                    attempts.append(
                        {
                            "attempt": attempt_idx + 1,
                            "strict_retry": strict_retry,
                            "prompt": prompt,
                            "raw_response": raw_text,
                            "parsed": parsed,
                            "error": None,
                        }
                    )
                    if attempt_idx == 1:
                        retries_used += 1
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    attempts.append(
                        {
                            "attempt": attempt_idx + 1,
                            "strict_retry": strict_retry,
                            "prompt": prompt,
                            "raw_response": locals().get("raw_text", ""),
                            "parsed": None,
                            "error": last_error,
                        }
                    )

            raw_path.write_text(
                json.dumps(
                    {
                        "claim_id": claim_id,
                        "example_id": segment["example_id"],
                        "judge_model": args.model,
                        "codex_app_server": {
                            "transport": "stdio://",
                            "approval_policy": "never",
                            "cwd": str(Path.cwd()),
                        },
                        "attempts": attempts,
                    },
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if payload is None:
                failures.append({"claim_id": claim_id, "stage": "judge", "error": last_error})
                print(f"FELM_MATCHED_V2_FAILED {claim_id} error={last_error}", flush=True)
                continue

            rows.append(
                {
                    "claim_id": claim_id,
                    "example_id": segment["example_id"],
                    "correctness_label": payload["correctness_label"],
                    "justification": payload["justification"],
                    "evidence_text": payload.get("evidence_text", ""),
                    "claim_text": segment["claim_text"],
                    "segment_index": segment.get("segment_index"),
                    "load_bearing_label": "yes",
                    "flip_evidence_text": payload.get("evidence_text", ""),
                    "notes": payload["justification"],
                    "label_source": "judge_llm_reference_proxy_v2",
                    "judge_model": args.model,
                    "annotation_version": "felm_matched_reference_judge_v2",
                }
            )
            write_jsonl(args.labels_jsonl, _sort_rows(rows))
            processed += 1
            print(f"FELM_MATCHED_V2_JUDGED {claim_id} label={payload['correctness_label']}", flush=True)

    rows = _sort_rows(rows)
    distribution = dict(Counter(str(row.get("correctness_label")) for row in rows))
    write_jsonl(args.labels_jsonl, rows)
    summary = {
        "n_segments_requested": len(segments),
        "n_labeled_segments": len(rows),
        "n_segments_available_in_labels": len(rows),
        "n_segments_processed_this_run": processed,
        "n_segments_skipped_existing": skipped_existing,
        "n_segments_reused_raw": reused_raw,
        "n_failures": len(failures),
        "failures": failures,
        "label_distribution": distribution,
        "n_retries_used_successfully": retries_used,
        "judge_model": args.model,
        "label_source": "judge_llm_reference_proxy_v2",
        "annotation_version": "felm_matched_reference_judge_v2",
        "allowed_labels": LABELS,
        "codex_app_server": {
            "transport": "stdio://",
            "approval_policy": "never",
            "cwd": str(Path.cwd()),
            "turn_timeout_sec": args.turn_timeout_sec,
        },
        "reference_excerpt_max_chars": args.max_total_ref_chars,
        "prompt_reference_policy": "Use the v2 readability cache and pass claim-relevant excerpts from the extracted reference text.",
        "note": (
            "Labels use fuller v2 FELM reference text and preserve not_enough_evidence separately from false. "
            "For roll-up runs with --skip-existing, n_segments_processed_this_run can be 0 even when n_labeled_segments is complete."
        ),
    }
    write_json(args.summary_json, summary)
    print(f"Wrote v2 matched judge labels to {args.labels_jsonl}", flush=True)
    print(f"Wrote v2 matched judge summary to {args.summary_json}", flush=True)


if __name__ == "__main__":
    main()
