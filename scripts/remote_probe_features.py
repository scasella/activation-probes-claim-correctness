from __future__ import annotations

import argparse
import base64
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from interp_experiment.activations.claim_pooling import mean_pool_claim_features
from interp_experiment.activations.extractors import build_extractor_with_info
from interp_experiment.activations.sae_features import encode_with_sae, load_sae
from interp_experiment.env import load_repo_env
from interp_experiment.io import read_jsonl
from interp_experiment.schemas import AnswerRunRow, ClaimFeatureRow, ClaimRow


def _emit_jsonl_section(label: str, rows: list[dict[str, object]]) -> None:
    print(f"===BEGIN_{label}===")
    for row in rows:
        print(json.dumps(row, ensure_ascii=True))
    print(f"===END_{label}===")


def _emit_json_section(label: str, payload: dict[str, object]) -> None:
    print(f"===BEGIN_{label}===")
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    print(f"===END_{label}===")


def _emit_text_section(label: str, text: str) -> None:
    print(f"===BEGIN_{label}===")
    print(text)
    print(f"===END_{label}===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract probe features inside a remote sandbox and print them as JSONL.")
    parser.add_argument("--answer-runs-jsonl", type=Path, required=True)
    parser.add_argument("--claims-jsonl", type=Path, required=True)
    parser.add_argument("--feature-source", choices=["residual", "sae"], required=True)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer-index", type=int, default=19)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sae-release", default="goodfire-llama-3.1-8b-instruct")
    parser.add_argument("--sae-id", default="layer_19")
    args = parser.parse_args()

    load_repo_env()
    build_info = build_extractor_with_info(args.model_name, layer_index=args.layer_index, device=args.device)
    extractor = build_info.extractor
    answer_runs = {
        row.example_id: row
        for row in (AnswerRunRow.from_dict(item) for item in read_jsonl(args.answer_runs_jsonl))
    }
    claims_by_example: dict[str, list[ClaimRow]] = defaultdict(list)
    for claim in (ClaimRow.from_dict(item) for item in read_jsonl(args.claims_jsonl)):
        claims_by_example[claim.example_id].append(claim)

    sae = None
    if args.feature_source == "sae":
        sae, _, _ = load_sae(args.sae_release, args.sae_id, device=args.device)

    rows: list[dict[str, object]] = []
    sae_claim_ids: list[str] = []
    sae_example_ids: list[str] = []
    sae_vectors: list[list[float]] = []
    for example_id, example_claims in claims_by_example.items():
        answer_run = answer_runs[example_id]
        encoded = extractor.encode_answer_with_activations(
            prompt_text=answer_run.prompt_text,
            answer_text=answer_run.answer_text,
        )
        if encoded.token_ids != answer_run.token_ids:
            raise SystemExit(f"Token-id mismatch for example_id={example_id}")
        feature_tensor = encoded.residual_stream
        if sae is not None:
            feature_tensor = encode_with_sae(sae, feature_tensor)
        for claim in example_claims:
            vector = mean_pool_claim_features(feature_tensor, claim)
            if args.feature_source == "sae":
                sae_claim_ids.append(claim.claim_id)
                sae_example_ids.append(claim.example_id)
                sae_vectors.append(vector)
            else:
                row = ClaimFeatureRow(
                    claim_id=claim.claim_id,
                    example_id=claim.example_id,
                    feature_source=args.feature_source,
                    vector=vector,
                    correctness_target=None,
                    load_bearing_target=None,
                    stability_target=None,
                ).validate()
                rows.append(row.as_dict())
        print(f"REMOTE_FEATURES_OK {example_id} claims={len(example_claims)} source={args.feature_source}")

    summary = {
        "n_examples": len(claims_by_example),
        "n_rows": len(rows) if args.feature_source != "sae" else len(sae_claim_ids),
        "feature_source": args.feature_source,
        "extractor_name": build_info.extractor.__class__.__name__,
        "transformer_lens_primary_succeeded": build_info.primary_succeeded,
        "transformer_lens_primary_error": build_info.primary_error,
        "model_name": args.model_name,
    }
    _emit_json_section("SUMMARY_JSON", summary)
    if args.feature_source == "sae":
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            claim_ids=np.asarray(sae_claim_ids),
            example_ids=np.asarray(sae_example_ids),
            matrix=np.asarray(sae_vectors, dtype=np.float32),
        )
        _emit_text_section("FEATURES_NPZ_B64", base64.b64encode(buffer.getvalue()).decode("ascii"))
    else:
        _emit_jsonl_section("FEATURES_JSONL", rows)


if __name__ == "__main__":
    main()
