from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from interp_experiment.io import read_jsonl, write_json, write_jsonl
from interp_experiment.schemas import ClaimRow, ExampleRow


PRIME_BASE_URL = "https://api.pinference.ai/api/v1"
DEFAULT_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_INPUT_USD_PER_MTOK = 0.95
DEFAULT_OUTPUT_USD_PER_MTOK = 4.0


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _judge_schema(expected_claim_ids: list[str]) -> dict[str, object]:
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
                        "load_bearing_label": {"type": "string", "enum": ["yes", "no"]},
                        "flip_evidence_text": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": [
                        "claim_id",
                        "correctness_label",
                        "load_bearing_label",
                        "flip_evidence_text",
                        "notes",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    }


def _judge_prompt(example: ExampleRow, claims: list[ClaimRow]) -> str:
    claim_block = "\n".join(f"- {claim.claim_id}: {claim.claim_text}" for claim in claims)
    return (
        "You are a proxy legal annotator for a research pilot. "
        "Use only the contract excerpt, the question, the model answer, and the fixed claim list.\n\n"
        "For each claim return:\n"
        "- correctness_label: true, false, or partially_true\n"
        "- load_bearing_label: yes or no\n"
        "- flip_evidence_text: required if load_bearing_label=yes, else empty string\n"
        "- notes: optional short note\n\n"
        f"Contract excerpt:\n{example.excerpt_text}\n\n"
        f"Question:\n{example.question_text}\n\n"
        f"Llama answer:\n{example.llama_answer_text}\n\n"
        f"Canonical claims:\n{claim_block}\n"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response did not contain a JSON object")
    return json.loads(text[start : end + 1])


class PrimeInferenceClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_sec: float,
        input_usd_per_mtok: float,
        output_usd_per_mtok: float,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec
        self.input_usd_per_mtok = input_usd_per_mtok
        self.output_usd_per_mtok = output_usd_per_mtok

    def run_prompt(self, prompt: str, schema: dict[str, object]) -> dict[str, Any]:
        request_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "usage": {"include": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_annotations",
                    "schema": schema,
                    "strict": True,
                },
            },
        }
        body = json.dumps(request_payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "curl/8.7.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Prime HTTP {exc.code}: {body[:1000]}") from exc
        message = response_payload["choices"][0]["message"]
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        if not isinstance(content, str):
            raise ValueError("Prime response did not include string message content")
        parsed = _extract_json_object(content)
        return {
            "request": request_payload,
            "response": response_payload,
            "content": content,
            "parsed": parsed,
        }


def _validate_payload(payload: dict[str, Any], expected_claim_ids: list[str]) -> list[dict[str, Any]]:
    items = payload.get("claims")
    if not isinstance(items, list):
        raise ValueError("parsed payload lacks a claims array")
    if len(items) != len(expected_claim_ids):
        raise ValueError(f"expected {len(expected_claim_ids)} claims, received {len(items)}")
    expected = set(expected_claim_ids)
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("claim item is not an object")
        claim_id = item.get("claim_id")
        if claim_id not in expected:
            raise ValueError(f"unexpected claim_id: {claim_id}")
        if claim_id in seen:
            raise ValueError(f"duplicate claim_id: {claim_id}")
        seen.add(claim_id)
        if item.get("correctness_label") not in {"true", "false", "partially_true"}:
            raise ValueError(f"invalid correctness_label for {claim_id}")
        if item.get("load_bearing_label") not in {"yes", "no"}:
            raise ValueError(f"invalid load_bearing_label for {claim_id}")
        if not isinstance(item.get("flip_evidence_text"), str):
            raise ValueError(f"invalid flip_evidence_text for {claim_id}")
        if not isinstance(item.get("notes"), str):
            raise ValueError(f"invalid notes for {claim_id}")
    return items


def _usage_cost(raw_rows: list[dict[str, Any]], input_usd_per_mtok: float, output_usd_per_mtok: float) -> dict[str, object]:
    prompt_tokens = 0
    completion_tokens = 0
    for row in raw_rows:
        usage = row.get("response", {}).get("usage", {})
        prompt_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    estimated_cost = (prompt_tokens / 1_000_000 * input_usd_per_mtok) + (
        completion_tokens / 1_000_000 * output_usd_per_mtok
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "input_usd_per_mtok": input_usd_per_mtok,
        "output_usd_per_mtok": output_usd_per_mtok,
        "estimated_cost_usd": estimated_cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize second-judge MAUD annotations with Prime Inference.")
    parser.add_argument("--examples-jsonl", type=Path, default=Path("artifacts/runs/maud_full_examples_with_answers.jsonl"))
    parser.add_argument("--claims-jsonl", type=Path, default=Path("artifacts/runs/maud_full_claims.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/annotations/maud_full_judge_annotations_v2.jsonl"))
    parser.add_argument("--summary-json", type=Path, default=Path("artifacts/runs/maud_full_judge_annotation_summary_v2.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/annotations/judge_llm_raw_v2/maud_full"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--annotator-id", default="judge_kimi_k2_6")
    parser.add_argument("--annotation-version", default="proxy_v2")
    parser.add_argument("--base-url", default=PRIME_BASE_URL)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--retry-once", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    parser.add_argument("--input-usd-per-mtok", type=float, default=DEFAULT_INPUT_USD_PER_MTOK)
    parser.add_argument("--output-usd-per-mtok", type=float, default=DEFAULT_OUTPUT_USD_PER_MTOK)
    args = parser.parse_args()

    _load_dotenv(args.dotenv)
    api_key = os.environ.get("PRIME_API_KEY")
    if not api_key:
        raise RuntimeError("PRIME_API_KEY is not set")

    examples = {row.example_id: row for row in (ExampleRow.from_dict(item) for item in read_jsonl(args.examples_jsonl))}
    claims_by_example: dict[str, list[ClaimRow]] = defaultdict(list)
    for claim in (ClaimRow.from_dict(item) for item in read_jsonl(args.claims_jsonl)):
        claims_by_example[claim.example_id].append(claim)

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    client = PrimeInferenceClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout_sec=args.timeout_sec,
        input_usd_per_mtok=args.input_usd_per_mtok,
        output_usd_per_mtok=args.output_usd_per_mtok,
    )

    annotation_rows: list[dict[str, str]] = []
    failures: list[dict[str, object]] = []
    raw_rows_for_cost: list[dict[str, Any]] = []
    processed = 0
    reused_existing = 0
    selected_examples = list(examples.items())
    if args.max_examples:
        selected_examples = selected_examples[: args.max_examples]

    def process_example(example_id: str, example: ExampleRow) -> dict[str, Any]:
        claims = claims_by_example[example_id]
        expected_claim_ids = [claim.claim_id for claim in claims]
        prompt = _judge_prompt(example, claims)
        raw_path = args.raw_dir / f"{example_id}.json"
        if args.skip_existing and raw_path.exists():
            raw_record = json.loads(raw_path.read_text(encoding="utf-8"))
            processed_this = 0
            reused_this = 1
        else:
            attempts = 2 if args.retry_once else 1
            last_error = ""
            raw_record = None
            for attempt in range(1, attempts + 1):
                started_at = time.time()
                try:
                    result = client.run_prompt(prompt, _judge_schema(expected_claim_ids))
                    parsed_items = _validate_payload(result["parsed"], expected_claim_ids)
                    raw_record = {
                        "example_id": example_id,
                        "model": args.model,
                        "annotator_id": args.annotator_id,
                        "annotation_version": args.annotation_version,
                        "attempt": attempt,
                        "ok": True,
                        "elapsed_sec": time.time() - started_at,
                        "prompt": prompt,
                        "schema": _judge_schema(expected_claim_ids),
                        **result,
                    }
                    raw_record["parsed"]["claims"] = parsed_items
                    break
                except (
                    RuntimeError,
                    urllib.error.HTTPError,
                    urllib.error.URLError,
                    TimeoutError,
                    ValueError,
                    KeyError,
                    json.JSONDecodeError,
                ) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < attempts:
                        time.sleep(2.0)
            if raw_record is None:
                raw_record = {
                    "example_id": example_id,
                    "model": args.model,
                    "annotator_id": args.annotator_id,
                    "annotation_version": args.annotation_version,
                    "ok": False,
                    "error": last_error,
                    "prompt": prompt,
                    "schema": _judge_schema(expected_claim_ids),
                }
            raw_path.write_text(json.dumps(raw_record, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
            processed_this = 1
            reused_this = 0
        rows: list[dict[str, str]] = []
        failure = None
        if raw_record.get("ok"):
            claim_text_by_id = {claim.claim_id: claim.claim_text for claim in claims}
            for item in raw_record["parsed"]["claims"]:
                rows.append(
                    {
                        "annotator_id": args.annotator_id,
                        "annotation_version": args.annotation_version,
                        "example_id": example.example_id,
                        "source_corpus": example.source_corpus,
                        "task_family": example.task_family,
                        "contract_id": example.contract_id,
                        "question_text": example.question_text,
                        "excerpt_text": example.excerpt_text,
                        "llama_answer_text": example.llama_answer_text,
                        "claim_id": item["claim_id"],
                        "claim_text": claim_text_by_id[item["claim_id"]],
                        "correctness_label": item["correctness_label"],
                        "load_bearing_label": item["load_bearing_label"],
                        "flip_evidence_text": item["flip_evidence_text"],
                        "notes": item["notes"],
                    }
                )
        else:
            failure = {"example_id": example_id, "error": raw_record.get("error", "unknown_error"), "category": "judge_failed"}
        return {
            "example_id": example_id,
            "rows": rows,
            "failure": failure,
            "raw_record": raw_record,
            "processed_this": processed_this,
            "reused_this": reused_this,
        }

    if args.concurrency <= 1:
        results = [process_example(example_id, example) for example_id, example in selected_examples]
    else:
        results_by_id: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(process_example, example_id, example): example_id
                for example_id, example in selected_examples
            }
            for future in as_completed(futures):
                example_id = futures[future]
                result = future.result()
                results_by_id[example_id] = result
                if result["failure"]:
                    print(f"MATERIALIZED_JUDGE_V2_FAILED {example_id}")
                else:
                    print(f"MATERIALIZED_JUDGE_V2 {example_id} claims={len(result['rows'])}")
        results = [results_by_id[example_id] for example_id, _ in selected_examples]

    for result in results:
        annotation_rows.extend(result["rows"])
        if result["failure"]:
            failures.append(result["failure"])
        elif args.concurrency <= 1:
            print(f"MATERIALIZED_JUDGE_V2 {result['example_id']} claims={len(result['rows'])}")
        raw_record = result["raw_record"]
        if raw_record.get("ok"):
            raw_rows_for_cost.append(raw_record)
        processed += result["processed_this"]
        reused_existing += result["reused_this"]

    write_jsonl(args.output_jsonl, annotation_rows)
    succeeded_example_ids = {row["example_id"] for row in annotation_rows}
    summary = {
        "n_examples": len(examples),
        "n_examples_selected": len(selected_examples),
        "n_examples_succeeded": len(succeeded_example_ids),
        "n_examples_failed": len(failures),
        "n_rows": len(annotation_rows),
        "n_examples_processed_this_run": processed,
        "n_examples_reused_existing": reused_existing,
        "annotator_id": args.annotator_id,
        "annotation_version": args.annotation_version,
        "model": args.model,
        "provider": "prime_intellect_inference",
        "base_url": args.base_url,
        "proxy_only": True,
        "same_prompt_as": "scripts/materialize_judge_annotations.py",
        "usage": _usage_cost(raw_rows_for_cost, args.input_usd_per_mtok, args.output_usd_per_mtok),
        "failures": failures,
    }
    write_json(args.summary_json, summary)
    print(f"Wrote judge-v2 annotations to {args.output_jsonl}")
    print(f"Wrote judge-v2 summary to {args.summary_json}")


if __name__ == "__main__":
    main()
