from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from interp_experiment.activations.claim_pooling import mean_pool_claim_features
from interp_experiment.activations.extractors import build_extractor
from interp_experiment.activations.sae_features import encode_with_sae, load_sae
from interp_experiment.io import read_jsonl, write_jsonl
from interp_experiment.schemas import ClaimFeatureRow, ClaimRow, ExampleRow


def build_prompt(example: ExampleRow) -> str:
    return (
        "Read the contract excerpt and answer the legal question.\n\n"
        f"Contract excerpt:\n{example.excerpt_text}\n\n"
        f"Question:\n{example.question_text}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pooled claim features from residual or SAE activations.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--claims-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer-index", type=int, default=19)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-source", choices=["residual", "sae"], default="residual")
    parser.add_argument("--sae-release", default="goodfire-llama-3.1-8b-instruct")
    parser.add_argument("--sae-id", default="llama3.1-8b-it/19-resid-post-gf")
    args = parser.parse_args()

    extractor = build_extractor(args.model_name, layer_index=args.layer_index, device=args.device)
    examples = {row.example_id: row for row in (ExampleRow.from_dict(item) for item in read_jsonl(args.examples_jsonl))}
    claims_by_example: dict[str, list[ClaimRow]] = defaultdict(list)
    for claim in (ClaimRow.from_dict(item) for item in read_jsonl(args.claims_jsonl)):
        claims_by_example[claim.example_id].append(claim)

    sae = None
    if args.feature_source == "sae":
        sae, _, _ = load_sae(args.sae_release, args.sae_id, device=args.device)

    rows: list[dict[str, object]] = []
    for example_id, example_claims in claims_by_example.items():
        generated = extractor.generate_with_activations(build_prompt(examples[example_id]))
        feature_tensor = generated.residual_stream
        if sae is not None:
            feature_tensor = encode_with_sae(sae, feature_tensor)
        for claim in example_claims:
            row = ClaimFeatureRow(
                claim_id=claim.claim_id,
                example_id=claim.example_id,
                feature_source=args.feature_source,
                vector=mean_pool_claim_features(feature_tensor, claim),
                correctness_target=None,
                load_bearing_target=None,
                stability_target=None,
            ).validate()
            rows.append(row.as_dict())
    write_jsonl(args.output_jsonl, rows)
    print(f"Wrote {len(rows)} claim feature rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
