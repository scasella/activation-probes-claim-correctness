from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..schemas import AnswerRunRow


@dataclass(slots=True)
class GenerationWithActivations:
    answer_text: str
    token_ids: list[int]
    token_offsets: list[tuple[int, int]]
    residual_stream: Any
    extractor_name: str
    model_name: str


@dataclass(slots=True)
class ExtractorBuildInfo:
    extractor: "ActivationExtractor"
    primary_attempted: bool
    primary_succeeded: bool
    primary_error: str | None


class ActivationExtractor:
    def generate_answer_run(
        self,
        example_id: str,
        source_corpus: str,
        task_family: str,
        prompt_text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> AnswerRunRow:  # pragma: no cover - interface
        raise NotImplementedError

    def retokenize_answer_run(
        self,
        example_id: str,
        source_corpus: str,
        task_family: str,
        prompt_text: str,
        answer_text: str,
    ) -> AnswerRunRow:  # pragma: no cover - interface
        raise NotImplementedError

    def generate_text(
        self,
        prompt_text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def generate_with_activations(self, prompt_text: str) -> GenerationWithActivations:  # pragma: no cover - interface
        raise NotImplementedError

    def encode_answer_with_activations(
        self,
        prompt_text: str,
        answer_text: str,
    ) -> GenerationWithActivations:  # pragma: no cover - interface
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

    def _answer_run_from_answer_sequence(
        self,
        example_id: str,
        source_corpus: str,
        task_family: str,
        prompt_text: str,
        answer_sequence: Any,
    ) -> AnswerRunRow:
        decoded_answer = self._model.to_string(answer_sequence)
        return AnswerRunRow(
            example_id=example_id,
            source_corpus=source_corpus,
            task_family=task_family,
            prompt_text=prompt_text,
            answer_text=decoded_answer,
            model_name=self.model_name,
            extractor_name="transformer_lens",
            token_ids=answer_sequence.tolist(),
            token_offsets=_naive_offsets_from_tokens(self._model.to_str_tokens(answer_sequence)),
        ).validate()

    def generate_answer_run(
        self,
        example_id: str,
        source_corpus: str,
        task_family: str,
        prompt_text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> AnswerRunRow:
        prompt_tokens = self._model.to_tokens(prompt_text, prepend_bos=True)
        generated = self._model.generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        prompt_len = prompt_tokens.shape[-1]
        answer_sequence = generated[0][prompt_len:]
        return self._answer_run_from_answer_sequence(
            example_id=example_id,
            source_corpus=source_corpus,
            task_family=task_family,
            prompt_text=prompt_text,
            answer_sequence=answer_sequence,
        )

    def retokenize_answer_run(
        self,
        example_id: str,
        source_corpus: str,
        task_family: str,
        prompt_text: str,
        answer_text: str,
    ) -> AnswerRunRow:
        prompt_tokens = self._model.to_tokens(prompt_text, prepend_bos=True)
        full_tokens = self._model.to_tokens(prompt_text + answer_text, prepend_bos=True)
        prompt_len = prompt_tokens.shape[-1]
        answer_sequence = full_tokens[0][prompt_len:]
        if answer_sequence.numel() == 0:
            answer_sequence = self._model.to_tokens(answer_text, prepend_bos=False)[0]
        return self._answer_run_from_answer_sequence(
            example_id=example_id,
            source_corpus=source_corpus,
            task_family=task_family,
            prompt_text=prompt_text,
            answer_sequence=answer_sequence,
        )

    def generate_text(
        self,
        prompt_text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        prompt_tokens = self._model.to_tokens(prompt_text, prepend_bos=True)
        generated = self._model.generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        prompt_len = prompt_tokens.shape[-1]
        answer_sequence = generated[0][prompt_len:]
        return self._model.to_string(answer_sequence)

    def generate_with_activations(self, prompt_text: str) -> GenerationWithActivations:
        prompt_tokens = self._model.to_tokens(prompt_text, prepend_bos=True)
        generated = self._model.generate(prompt_tokens, max_new_tokens=256, temperature=0.0)
        full_sequence = generated[0].detach().clone()
        prompt_len = prompt_tokens.shape[-1]
        answer_sequence = full_sequence[prompt_len:]
        answer_text = self._model.to_string(answer_sequence)
        with self._torch.no_grad():
            _, cache = self._model.run_with_cache(
                full_sequence.unsqueeze(0),
                names_filter=lambda name: name == f"blocks.{self.layer_index}.hook_resid_post",
            )
        residual_full = cache[f"blocks.{self.layer_index}.hook_resid_post"][0].detach().cpu()
        residual = residual_full[prompt_len:]
        decoded_tokens = answer_sequence.tolist()
        offsets = _naive_offsets_from_tokens(self._model.to_str_tokens(answer_sequence))
        return GenerationWithActivations(
            answer_text=answer_text,
            token_ids=decoded_tokens,
            token_offsets=offsets,
            residual_stream=residual,
            extractor_name="transformer_lens",
            model_name=self.model_name,
        )

    def encode_answer_with_activations(
        self,
        prompt_text: str,
        answer_text: str,
    ) -> GenerationWithActivations:
        answer_run = self.retokenize_answer_run(
            example_id="adhoc",
            source_corpus="adhoc",
            task_family="generative_qa",
            prompt_text=prompt_text,
            answer_text=answer_text,
        )
        prompt_tokens = self._model.to_tokens(prompt_text, prepend_bos=True)
        full_tokens = self._model.to_tokens(prompt_text + answer_run.answer_text, prepend_bos=True)
        prompt_len = prompt_tokens.shape[-1]
        full_sequence = full_tokens[0].detach().clone()
        with self._torch.no_grad():
            _, cache = self._model.run_with_cache(
                full_sequence.unsqueeze(0),
                names_filter=lambda name: name == f"blocks.{self.layer_index}.hook_resid_post",
            )
        residual_full = cache[f"blocks.{self.layer_index}.hook_resid_post"][0].detach().cpu()
        residual = residual_full[prompt_len:]
        return GenerationWithActivations(
            answer_text=answer_run.answer_text,
            token_ids=answer_run.token_ids,
            token_offsets=answer_run.token_offsets,
            residual_stream=residual,
            extractor_name="transformer_lens",
            model_name=self.model_name,
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

    def _decode_answer_sequence(self, answer_ids: Any) -> str:
        return self._tokenizer.decode(answer_ids, skip_special_tokens=True)

    def _answer_run_from_answer_ids(
        self,
        example_id: str,
        source_corpus: str,
        task_family: str,
        prompt_text: str,
        answer_ids: Any,
    ) -> AnswerRunRow:
        decoded_answer = self._decode_answer_sequence(answer_ids)
        answer_offsets_batch = self._tokenizer(
            decoded_answer,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = [
            (int(start), int(end))
            for start, end in answer_offsets_batch["offset_mapping"]
        ]
        return AnswerRunRow(
            example_id=example_id,
            source_corpus=source_corpus,
            task_family=task_family,
            prompt_text=prompt_text,
            answer_text=decoded_answer,
            model_name=self.model_name,
            extractor_name="huggingface_transformers",
            token_ids=answer_ids.tolist(),
            token_offsets=offsets,
        ).validate()

    def generate_answer_run(
        self,
        example_id: str,
        source_corpus: str,
        task_family: str,
        prompt_text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> AnswerRunRow:
        batch = self._tokenizer(prompt_text, return_tensors="pt")
        input_ids = batch["input_ids"].to(self.device)
        sequences = self._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )[0]
        prompt_len = int(input_ids.shape[1])
        answer_ids = sequences[prompt_len:]
        return self._answer_run_from_answer_ids(
            example_id=example_id,
            source_corpus=source_corpus,
            task_family=task_family,
            prompt_text=prompt_text,
            answer_ids=answer_ids,
        )

    def retokenize_answer_run(
        self,
        example_id: str,
        source_corpus: str,
        task_family: str,
        prompt_text: str,
        answer_text: str,
    ) -> AnswerRunRow:
        prompt_batch = self._tokenizer(prompt_text, return_tensors="pt")
        full_batch = self._tokenizer(prompt_text + answer_text, return_tensors="pt")
        prompt_len = int(prompt_batch["input_ids"].shape[1])
        answer_ids = full_batch["input_ids"][0][prompt_len:]
        if answer_ids.numel() == 0:
            answer_ids = self._tokenizer(
                answer_text,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0]
        return self._answer_run_from_answer_ids(
            example_id=example_id,
            source_corpus=source_corpus,
            task_family=task_family,
            prompt_text=prompt_text,
            answer_ids=answer_ids,
        )

    def generate_text(
        self,
        prompt_text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        batch = self._tokenizer(prompt_text, return_tensors="pt")
        input_ids = batch["input_ids"].to(self.device)
        sequences = self._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )[0]
        prompt_len = int(input_ids.shape[1])
        answer_ids = sequences[prompt_len:]
        return self._tokenizer.decode(answer_ids, skip_special_tokens=True)

    def generate_with_activations(self, prompt_text: str) -> GenerationWithActivations:
        batch = self._tokenizer(prompt_text, return_tensors="pt")
        input_ids = batch["input_ids"].to(self.device)
        outputs = self._model.generate(
            input_ids,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            return_dict_in_generate=True,
        )
        sequences = outputs.sequences[0]
        prompt_len = int(input_ids.shape[1])
        answer_ids = sequences[prompt_len:]
        answer_text = self._tokenizer.decode(answer_ids, skip_special_tokens=True)
        forward = self._model(sequences.unsqueeze(0), output_hidden_states=True)
        hidden_states = forward.hidden_states[self.layer_index + 1][0].detach().cpu()
        residual = hidden_states[prompt_len:]
        answer_offsets_batch = self._tokenizer(
            answer_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = [
            (int(start), int(end))
            for start, end in answer_offsets_batch["offset_mapping"]
        ]
        return GenerationWithActivations(
            answer_text=answer_text,
            token_ids=answer_ids.tolist(),
            token_offsets=offsets,
            residual_stream=residual,
            extractor_name="huggingface_transformers",
            model_name=self.model_name,
        )

    def encode_answer_with_activations(
        self,
        prompt_text: str,
        answer_text: str,
    ) -> GenerationWithActivations:
        answer_run = self.retokenize_answer_run(
            example_id="adhoc",
            source_corpus="adhoc",
            task_family="generative_qa",
            prompt_text=prompt_text,
            answer_text=answer_text,
        )
        prompt_batch = self._tokenizer(prompt_text, return_tensors="pt")
        full_batch = self._tokenizer(prompt_text + answer_run.answer_text, return_tensors="pt")
        input_ids = full_batch["input_ids"].to(self.device)
        prompt_len = int(prompt_batch["input_ids"].shape[1])
        forward = self._model(input_ids, output_hidden_states=True)
        hidden_states = forward.hidden_states[self.layer_index + 1][0].detach().cpu()
        residual = hidden_states[prompt_len:]
        return GenerationWithActivations(
            answer_text=answer_run.answer_text,
            token_ids=answer_run.token_ids,
            token_offsets=answer_run.token_offsets,
            residual_stream=residual,
            extractor_name="huggingface_transformers",
            model_name=self.model_name,
        )


def _naive_offsets_from_tokens(tokens: list[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = cursor
        cursor += len(token)
        offsets.append((start, cursor))
    return offsets


def build_extractor_with_info(model_name: str, layer_index: int = 19, device: str = "cuda") -> ExtractorBuildInfo:
    try:
        extractor = TransformerLensActivationExtractor(model_name=model_name, layer_index=layer_index, device=device)
        return ExtractorBuildInfo(
            extractor=extractor,
            primary_attempted=True,
            primary_succeeded=True,
            primary_error=None,
        )
    except Exception as exc:
        fallback = HuggingFaceActivationExtractor(model_name=model_name, layer_index=layer_index, device=device)
        return ExtractorBuildInfo(
            extractor=fallback,
            primary_attempted=True,
            primary_succeeded=False,
            primary_error=f"{type(exc).__name__}: {exc}",
        )


def build_extractor(model_name: str, layer_index: int = 19, device: str = "cuda") -> ActivationExtractor:
    return build_extractor_with_info(model_name=model_name, layer_index=layer_index, device=device).extractor
