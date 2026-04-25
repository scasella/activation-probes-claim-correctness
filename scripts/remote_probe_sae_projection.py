from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from interp_experiment.activations.sae_features import load_sae
from interp_experiment.env import load_repo_env


def _emit_json_section(label: str, payload: dict[str, Any]) -> None:
    print(f"===BEGIN_{label}===")
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    print(f"===END_{label}===")


def _as_numpy(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().float().cpu()
    return np.asarray(tensor, dtype=np.float64)


def _orient_feature_matrix(matrix: np.ndarray, d_model: int, *, name: str) -> np.ndarray:
    """Return matrix as (n_features, d_model)."""
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape={matrix.shape}")
    if matrix.shape[1] == d_model:
        return matrix
    if matrix.shape[0] == d_model:
        return matrix.T
    raise ValueError(f"{name} shape={matrix.shape} does not include d_model={d_model}")


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    values = np.sort(np.abs(values))
    total = values.sum()
    if total == 0.0:
        return 0.0
    index = np.arange(1, values.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * values) / (values.size * total)) - ((values.size + 1.0) / values.size))


def _topk_l2_coverage(values: np.ndarray, ks: list[int]) -> dict[str, float]:
    squared = np.sort(np.square(np.asarray(values, dtype=np.float64)))[::-1]
    denom = float(squared.sum())
    if denom == 0.0:
        return {str(k): 0.0 for k in ks}
    return {str(k): float(squared[:k].sum() / denom) for k in ks}


def _ranked_features(values: np.ndarray, limit: int = 50) -> dict[str, list[dict[str, float | int]]]:
    values = np.asarray(values, dtype=np.float64)

    def rows(indices: np.ndarray) -> list[dict[str, float | int]]:
        return [
            {
                "feature_id": int(idx),
                "value": float(values[idx]),
                "abs_value": float(abs(values[idx])),
                "rank": int(rank),
            }
            for rank, idx in enumerate(indices[:limit], start=1)
        ]

    return {
        "top_abs": rows(np.argsort(np.abs(values))[::-1]),
        "top_positive": rows(np.argsort(values)[::-1]),
        "top_negative": rows(np.argsort(values)),
    }


def _summary_stats(samples: list[dict[str, Any]], key: str, *, nested_key: str | None = None) -> dict[str, float]:
    if nested_key is None:
        values = np.asarray([float(row[key]) for row in samples], dtype=np.float64)
    else:
        values = np.asarray([float(row[key][nested_key]) for row in samples], dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _diagnostics(values: np.ndarray, ks: list[int]) -> dict[str, Any]:
    return {
        "gini_abs": _gini(values),
        "topk_l2_coverage": _topk_l2_coverage(values, ks),
        "l2_norm": float(np.linalg.norm(values)),
        "l1_norm": float(np.sum(np.abs(values))),
        "max_abs": float(np.max(np.abs(values))),
        "median_abs": float(np.median(np.abs(values))),
    }


def _null_distribution(
    *,
    normalized_decoder: np.ndarray,
    encoder_feature_by_residual: np.ndarray,
    d_model: int,
    ks: list[int],
    n_random: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    decoder_samples: list[dict[str, Any]] = []
    encoder_samples: list[dict[str, Any]] = []
    for _ in range(n_random):
        vector = rng.normal(size=d_model)
        vector = vector / np.linalg.norm(vector)
        decoder_values = normalized_decoder @ vector
        encoder_values = encoder_feature_by_residual @ vector
        decoder_samples.append(_diagnostics(decoder_values, ks))
        encoder_samples.append(_diagnostics(encoder_values, ks))

    def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "gini_abs": _summary_stats(samples, "gini_abs"),
            "l2_norm": _summary_stats(samples, "l2_norm"),
            "max_abs": _summary_stats(samples, "max_abs"),
            "topk_l2_coverage": {
                str(k): _summary_stats(samples, "topk_l2_coverage", nested_key=str(k)) for k in ks
            },
        }

    return {
        "n_random": n_random,
        "seed": seed,
        "decoder_cosine": summarize(decoder_samples),
        "encoder_projection": summarize(encoder_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Project a residual probe direction into the Goodfire SAE basis.")
    parser.add_argument("--probe-npz", type=Path, default=Path("artifacts/runs/residual_correctness_probe_direction.npz"))
    parser.add_argument("--sae-release", default="goodfire-llama-3.1-8b-instruct")
    parser.add_argument("--sae-id", default="layer_19")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260424)
    args = parser.parse_args()

    load_repo_env()
    payload = np.load(args.probe_npz, allow_pickle=False)
    probe_direction = np.asarray(payload["residual_direction"], dtype=np.float64)
    d_model = int(probe_direction.shape[0])
    probe_unit = probe_direction / np.linalg.norm(probe_direction)

    sae, cfg_dict, sparsity = load_sae(args.sae_release, args.sae_id, device=args.device)
    w_dec = _orient_feature_matrix(_as_numpy(sae.W_dec), d_model, name="W_dec")
    w_enc = _orient_feature_matrix(_as_numpy(sae.W_enc), d_model, name="W_enc")
    decoder_norms = np.linalg.norm(w_dec, axis=1)
    safe_decoder_norms = np.where(decoder_norms == 0.0, 1.0, decoder_norms)
    normalized_decoder = w_dec / safe_decoder_norms[:, None]

    decoder_cosine = normalized_decoder @ probe_unit
    encoder_projection = w_enc @ probe_unit
    ks = [1, 5, 10, 25, 50]

    alignment = {
        "probe_source": str(args.probe_npz),
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "d_model": d_model,
        "n_features": int(w_dec.shape[0]),
        "w_dec_shape_oriented_feature_by_residual": list(w_dec.shape),
        "w_enc_shape_oriented_feature_by_residual": list(w_enc.shape),
        "decoder_cosine": _ranked_features(decoder_cosine, limit=args.top_n),
        "encoder_projection": _ranked_features(encoder_projection, limit=args.top_n),
        "cfg": cfg_dict if isinstance(cfg_dict, dict) else {},
        "sparsity_available": sparsity is not None,
    }
    diagnostics = {
        "probe_source": str(args.probe_npz),
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "ks": ks,
        "decoder_cosine": _diagnostics(decoder_cosine, ks),
        "encoder_projection": _diagnostics(encoder_projection, ks),
        "null_distribution": _null_distribution(
            normalized_decoder=normalized_decoder,
            encoder_feature_by_residual=w_enc,
            d_model=d_model,
            ks=ks,
            n_random=args.n_random,
            seed=args.seed,
        ),
    }
    _emit_json_section("PROBE_FEATURE_ALIGNMENT_JSON", alignment)
    _emit_json_section("PROBE_SPARSITY_DIAGNOSTICS_JSON", diagnostics)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
