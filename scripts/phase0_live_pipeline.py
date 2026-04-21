from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from interp_experiment.activations.claim_pooling import mean_pool_claim_features
from interp_experiment.activations.extractors import build_extractor_with_info
from interp_experiment.activations.sae_features import encode_with_sae, load_sae
from interp_experiment.baselines.codex_app_server import build_request_packet
from interp_experiment.baselines.llama_self_report import parse_baseline_claims
from interp_experiment.baselines.prompts import render_gpt54_prompt, render_self_report_prompt
from interp_experiment.data.claims import build_canonical_claims
from interp_experiment.env import load_repo_env
from interp_experiment.io import write_json, write_jsonl
from interp_experiment.paths import ROOT
from interp_experiment.schemas import ClaimRow, ExampleRow


DEFAULT_EXCERPT = (
    "Section 8.2 Termination Rights. Buyer may terminate this Agreement prior to Closing if a "
    "Company Material Adverse Effect has occurred and is continuing, provided Buyer has not "
    "materially breached this Agreement. Section 10.7 Governing Law. This Agreement is governed "
    "by the laws of the State of Delaware."
)
DEFAULT_QUESTION = "Can the buyer terminate the agreement for a Company Material Adverse Effect?"
DEFAULT_PUBLIC_SEED_ANSWER = (
    "Yes. The agreement allows the buyer to terminate before closing if a Company Material Adverse "
    "Effect has occurred and is continuing, assuming the buyer is not in material breach."
)


def _example_prompt(example: ExampleRow) -> str:
    return (
        "You are a careful legal contract analyst. Read the contract excerpt and answer the question "
        "briefly and directly.\n\n"
        f"Contract excerpt:\n{example.excerpt_text}\n\n"
        f"Question:\n{example.question_text}\n"
    )


def _default_example() -> ExampleRow:
    return ExampleRow(
        example_id="phase0-live-smoke",
        source_corpus="phase0_smoke",
        contract_id="phase0-contract",
        contract_group="merger_agreement",
        excerpt_text=DEFAULT_EXCERPT,
        question_text=DEFAULT_QUESTION,
        public_seed_answer=DEFAULT_PUBLIC_SEED_ANSWER,
        llama_answer_text="",
        split="pilot",
        cross_dist_group="maud_merger",
    ).validate()


def _synthetic_baseline_payload(claims: list[ClaimRow]) -> dict[str, object]:
    payload = {
        "claims": [
            {
                "claim_id": claim.claim_id,
                "correctness_confidence": 0.72,
                "load_bearing_label": "yes" if idx == 0 else "no",
                "load_bearing_confidence": 0.67 if idx == 0 else 0.31,
                "flip_evidence_text": (
                    "Contrary termination language or an explicit carve-out for Company Material Adverse Effect."
                    if idx == 0
                    else ""
                ),
            }
            for idx, claim in enumerate(claims)
        ]
    }
    return payload


def _to_numpy(array_like: object) -> np.ndarray:
    if hasattr(array_like, "detach"):
        array_like = array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live phase-0 end-to-end smoke pipeline.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/phase0_live"))
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer-index", type=int, default=19)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sae-release", default="goodfire-llama-3.1-8b-instruct")
    parser.add_argument("--sae-id", default="layer_19")
    args = parser.parse_args()

    load_repo_env()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    example = _default_example()
    build_info = build_extractor_with_info(args.model_name, layer_index=args.layer_index, device=args.device)
    generated = build_info.extractor.generate_with_activations(_example_prompt(example))
    example.llama_answer_text = generated.answer_text
    example.validate()

    claims = build_canonical_claims(example)
    residual_array = _to_numpy(generated.residual_stream)
    residual_path = output_dir / "raw_residual.npz"
    np.savez_compressed(residual_path, residual=residual_array)

    sae, cfg_dict, sparsity = load_sae(args.sae_release, args.sae_id, device=args.device)
    sae_encoded = encode_with_sae(sae, generated.residual_stream)
    sae_array = _to_numpy(sae_encoded)
    np.savez_compressed(output_dir / "sae_encoded.npz", sae=sae_array)

    first_claim = claims[0]
    pooled_sae = mean_pool_claim_features(sae_array, first_claim)
    pooled_residual = mean_pool_claim_features(residual_array, first_claim)
    pooled_sae_np = np.asarray(pooled_sae, dtype=float)
    topk = min(10, pooled_sae_np.shape[0])
    topk_idx = np.argsort(np.abs(pooled_sae_np))[-topk:][::-1]

    synthetic_raw = _synthetic_baseline_payload(claims)
    parsed_baseline = parse_baseline_claims(
        synthetic_raw,
        prompt_version="v1",
        model_name="phase0-synthetic-baseline",
    )

    self_report_prompt = render_self_report_prompt(example, claims)
    gpt54_prompt = render_gpt54_prompt(example, claims)
    gpt54_packet = build_request_packet(
        example_id=example.example_id,
        system_prompt=gpt54_prompt["system"],
        user_prompt=gpt54_prompt["user"],
        prompt_version=gpt54_prompt["prompt_version"],
    )

    write_jsonl(output_dir / "example.jsonl", [example.as_dict()])
    write_jsonl(output_dir / "claims.jsonl", [claim.as_dict() for claim in claims])
    write_json(output_dir / "synthetic_baseline_raw.json", synthetic_raw)
    write_jsonl(output_dir / "synthetic_baseline_parsed.jsonl", [row.as_dict() for row in parsed_baseline])
    write_json(output_dir / "self_report_prompt.json", self_report_prompt)
    write_json(output_dir / "gpt54_request_packet.json", gpt54_packet)

    summary = {
        "repo_root": str(ROOT),
        "model_name": generated.model_name,
        "extractor_name": generated.extractor_name,
        "transformer_lens_primary_succeeded": build_info.primary_succeeded,
        "transformer_lens_primary_error": build_info.primary_error,
        "answer_text": generated.answer_text,
        "n_answer_tokens": len(generated.token_ids),
        "n_claims": len(claims),
        "claim_ids": [claim.claim_id for claim in claims],
        "residual_shape": list(residual_array.shape),
        "sae_shape": list(sae_array.shape),
        "sae_cfg_keys": sorted(cfg_dict.keys())[:20],
        "sparsity_object_type": type(sparsity).__name__,
        "first_claim_id": first_claim.claim_id,
        "first_claim_text": first_claim.claim_text,
        "first_claim_token_span": [first_claim.token_start, first_claim.token_end],
        "first_claim_residual_dim": len(pooled_residual),
        "first_claim_sae_dim": len(pooled_sae),
        "first_claim_sae_top_features": [
            {"feature_index": int(idx), "value": float(pooled_sae_np[idx])}
            for idx in topk_idx
        ],
        "artifacts": {
            "example": str(output_dir / "example.jsonl"),
            "claims": str(output_dir / "claims.jsonl"),
            "raw_residual": str(residual_path),
            "sae_encoded": str(output_dir / "sae_encoded.npz"),
            "synthetic_baseline_raw": str(output_dir / "synthetic_baseline_raw.json"),
            "synthetic_baseline_parsed": str(output_dir / "synthetic_baseline_parsed.jsonl"),
            "self_report_prompt": str(output_dir / "self_report_prompt.json"),
            "gpt54_request_packet": str(output_dir / "gpt54_request_packet.json"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Wrote live phase-0 smoke artifacts to {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
