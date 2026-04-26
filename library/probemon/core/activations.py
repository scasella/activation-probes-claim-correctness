from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class GenerationWithActivations:
    answer_text: str
    residual_stream: np.ndarray
    token_offsets: list[tuple[int, int]]
    model_name: str
    layer: int
    token_ids: list[int] | None = None


def mean_pool_char_span(
    residual_stream: Any,
    token_offsets: list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> np.ndarray:
    residual = np.asarray(residual_stream, dtype=float)
    if residual.ndim != 2:
        raise ValueError("residual_stream must be a 2D array of shape [tokens, residual_dim]")
    if len(token_offsets) != residual.shape[0]:
        raise ValueError("token_offsets length must match residual_stream token dimension")
    indices = [
        idx
        for idx, (start, end) in enumerate(token_offsets)
        if end > char_start and start < char_end
    ]
    if not indices:
        raise ValueError(f"No tokens overlap character span [{char_start}, {char_end})")
    pooled = residual[indices].mean(axis=0)
    if pooled.size == 0 or not np.isfinite(pooled).all():
        raise ValueError("Pooled residual vector is empty or non-finite")
    return pooled


class HuggingFaceActivationExtractor:
    """Lazy HuggingFace residual extractor.

    This class deliberately imports torch/transformers only at construction
    time so importing probemon remains lightweight.
    """

    def __init__(self, model_name: str, layer: int = 19, device: str = "cuda") -> None:
        self.model_name = model_name
        self.layer = layer
        self.device = device
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "HuggingFace activation extraction requires torch and transformers. "
                "Install with `pip install probemon[inference]`."
            ) from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self._model.to(device)
        self._model.eval()

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        batch = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self._model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        answer_ids = outputs[0][batch["input_ids"].shape[1] :]
        return self._tokenizer.decode(answer_ids, skip_special_tokens=True)

    def encode_answer_with_activations(self, prompt: str, generation: str) -> GenerationWithActivations:
        prompt_batch = self._tokenizer(prompt, return_tensors="pt")
        full_batch = self._tokenizer(prompt + generation, return_tensors="pt")
        prompt_len = int(prompt_batch["input_ids"].shape[1])
        input_ids = full_batch["input_ids"].to(self.device)
        with self._torch.no_grad():
            outputs = self._model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[self.layer + 1][0].detach().float().cpu().numpy()
        residual = hidden[prompt_len:]
        answer_offsets = self._tokenizer(
            generation,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )["offset_mapping"]
        offsets = [(int(start), int(end)) for start, end in answer_offsets]
        if len(offsets) != residual.shape[0]:
            limit = min(len(offsets), residual.shape[0])
            offsets = offsets[:limit]
            residual = residual[:limit]
        return GenerationWithActivations(
            answer_text=generation,
            residual_stream=residual,
            token_offsets=offsets,
            model_name=self.model_name,
            layer=self.layer,
            token_ids=input_ids[0][prompt_len:].detach().cpu().tolist(),
        )
