from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class GenerationWithActivations:
    answer_text: str
    token_ids: list[int]
    token_offsets: list[tuple[int, int]]
    residual_stream: Any
    extractor_name: str


class ActivationExtractor:
    def generate_with_activations(self, prompt_text: str) -> GenerationWithActivations:  # pragma: no cover - interface
        raise NotImplementedError


class TransformerLensActivationExtractor(ActivationExtractor):
    def __init__(self, model_name: str, layer_index: int = 19, device: str = "cuda") -> None:
        self.model_name = model_name
        self.layer_index = layer_index
        self.device = device
        try:
            import torch
            from transformer_lens import HookedTransformer
        except ImportError as exc:
            raise RuntimeError(
                "transformer-lens and torch are required for the primary activation path. "
                "Install with `uv sync --extra inference --extra interp`."
            ) from exc
        self._torch = torch
        self._model = HookedTransformer.from_pretrained(model_name, device=device)

    def generate_with_activations(self, prompt_text: str) -> GenerationWithActivations:
        tokens = self._model.to_tokens(prompt_text, prepend_bos=True)
        answer_tokens = self._model.generate(tokens, max_new_tokens=256, temperature=0.0)
        full_text = self._model.to_string(answer_tokens[0])
        _, cache = self._model.run_with_cache(answer_tokens, names_filter=lambda name: name == f"blocks.{self.layer_index}.hook_resid_post")
        residual = cache[f"blocks.{self.layer_index}.hook_resid_post"][0].detach().cpu()
        decoded_tokens = answer_tokens[0].tolist()
        offsets = _naive_offsets_from_tokens(self._model.to_str_tokens(answer_tokens[0]))
        return GenerationWithActivations(
            answer_text=full_text,
            token_ids=decoded_tokens,
            token_offsets=offsets,
            residual_stream=residual,
            extractor_name="transformer_lens",
        )


class HuggingFaceActivationExtractor(ActivationExtractor):
    def __init__(self, model_name: str, layer_index: int = 19, device: str = "cuda") -> None:
        self.model_name = model_name
        self.layer_index = layer_index
        self.device = device
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for the fallback activation path. "
                "Install with `uv sync --extra inference`."
            ) from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self._model.to(device)

    def generate_with_activations(self, prompt_text: str) -> GenerationWithActivations:
        batch = self._tokenizer(prompt_text, return_tensors="pt", return_offsets_mapping=True)
        input_ids = batch["input_ids"].to(self.device)
        outputs = self._model.generate(
            input_ids,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        sequences = outputs.sequences[0]
        answer_text = self._tokenizer.decode(sequences, skip_special_tokens=True)
        hidden_states = outputs.hidden_states[-1][self.layer_index][0].detach().cpu()
        offsets = [
            (int(start), int(end))
            for start, end in batch["offset_mapping"][0].tolist()
        ]
        return GenerationWithActivations(
            answer_text=answer_text,
            token_ids=sequences.tolist(),
            token_offsets=offsets,
            residual_stream=hidden_states,
            extractor_name="huggingface_transformers",
        )


def _naive_offsets_from_tokens(tokens: list[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = cursor
        cursor += len(token)
        offsets.append((start, cursor))
    return offsets


def build_extractor(model_name: str, layer_index: int = 19, device: str = "cuda") -> ActivationExtractor:
    try:
        return TransformerLensActivationExtractor(model_name=model_name, layer_index=layer_index, device=device)
    except Exception:
        return HuggingFaceActivationExtractor(model_name=model_name, layer_index=layer_index, device=device)
