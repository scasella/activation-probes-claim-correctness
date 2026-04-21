from __future__ import annotations

from typing import Any


def load_sae(release: str, sae_id: str, device: str = "cuda") -> tuple[Any, dict[str, Any], Any]:
    try:
        from sae_lens import SAE
    except ImportError as exc:
        raise RuntimeError(
            "sae-lens is required for SAE feature extraction. Install with `uv sync --extra interp`."
        ) from exc
    return SAE.from_pretrained_with_cfg_and_sparsity(release=release, sae_id=sae_id, device=device)


def encode_with_sae(sae: Any, residual_stream: Any) -> Any:
    sae_device = None
    if hasattr(sae, "parameters"):
        try:
            sae_device = next(sae.parameters()).device
        except StopIteration:
            sae_device = None
    sae_input = residual_stream
    if sae_device is not None and hasattr(residual_stream, "to"):
        sae_input = residual_stream.to(sae_device)
    encoded = sae.encode(sae_input)
    return encoded.detach().cpu() if hasattr(encoded, "detach") else encoded
