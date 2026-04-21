from __future__ import annotations

from typing import Any


def load_sae(release: str, sae_id: str, device: str = "cuda") -> tuple[Any, dict[str, Any], Any]:
    try:
        from sae_lens import SAE
    except ImportError as exc:
        raise RuntimeError(
            "sae-lens is required for SAE feature extraction. Install with `uv sync --extra interp`."
        ) from exc
    return SAE.from_pretrained(release=release, sae_id=sae_id, device=device)


def encode_with_sae(sae: Any, residual_stream: Any) -> Any:
    encoded = sae.encode(residual_stream)
    return encoded.detach().cpu() if hasattr(encoded, "detach") else encoded
