# ProbeMon

ProbeMon is a research-grade Python library for claim-level uncertainty monitoring with linear probes over Llama 3.1 8B residual activations. It ships two pretrained probes from the accompanying methods paper: MAUD legal QA and FActScore biographies. The core result behind the library is bounded: in those settings, residual activations recover correctness signal that structured self-report misses. Probe scores are not legal advice, factual ground truth, or a universal hallucination detector.

## Quickstart

```python
from probemon import load_probe, score_generation

probe = load_probe("legal-qa-llama-3.1-8b-v1")
result = score_generation(
    model="meta-llama/Llama-3.1-8B-Instruct",
    probe=probe,
    prompt="What does the merger agreement say about fiduciary exceptions?",
    generation="The board may change its recommendation if the agreement permits it.",
)

for item in result.scores:
    print(item.text, item.calibrated_score)
```

The package also includes a biography probe:

```python
probe = load_probe("biography-llama-3.1-8b-v1")
```

Both probes require Llama 3.1 8B Instruct residual activations at layer 19. If the loaded model name does not match the probe metadata, ProbeMon raises a clear error instead of silently scoring the wrong model.

## Train Your Own Probe

```python
from probemon.training import fit_probe, load_canonical_dataset

dataset = load_canonical_dataset("labeled_claims.jsonl")
fit_probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dataset=dataset,
    layer=19,
    output_path="my_probe.npz",
)
```

Canonical JSONL rows look like:

```json
{
  "prompt": "...",
  "generation": "...",
  "claims": [
    {"text": "...", "char_start": 0, "char_end": 42, "label": "correct"}
  ],
  "metadata": {"split": "train"}
}
```

Training reports dataset size and validation metrics before writing a probe. For a few hundred examples, expect roughly 10 minutes on an A100 when activations must be extracted.

## Scaffolding Datasets

```python
from probemon.datasets import load_dataset

maud = load_dataset("maud-legal-qa")
factscore = load_dataset("factscore-biographies")
```

The wheel does not bundle source datasets. MAUD redistribution must be checked against the Atticus Project terms before bundling, so the loader expects local methods-paper artifacts. FActScore is released publicly by its authors in the MIT-licensed upstream repository, but ProbeMon still keeps the data external and recommends checking the data-specific redistribution terms before bundling the human annotations.

The FActScore adapter uses parent-sentence-span pooling for canonicalized atomic-fact labels. The atomic fact remains the label target, but the residual vector is pooled over the sentence containing that fact because exact atomic-fact substring matching succeeds for only 0.15% of facts in the methods run.

## CLI

```bash
probemon list-probes
probemon score --probe legal-qa-llama-3.1-8b-v1 --model meta-llama/Llama-3.1-8B-Instruct \
  --prompt-file prompt.txt --generation-file generation.txt
probemon train --dataset factscore-biographies --model meta-llama/Llama-3.1-8B-Instruct \
  --layer 19 --output my_probe.npz
```

## Limitations

Probe scores are only as good as the training labels and claim spans. The OOD warning is a reminder, not a robust detector: it checks prompt length and a small set of domain-indicator terms. Suppressing it requires `suppress_ood_warning=True`.

v0.1 supports only Llama 3.1 8B Instruct and layer-19 residual probes. It does not ship a FELM probe because FELM is a documented boundary case in the paper, and it does not support streaming token-by-token visualization.

## License And Data Terms

ProbeMon code is MIT-licensed as part of the parent repository. The shipped probe artifacts are compact research artifacts derived from upstream datasets and model outputs; they do not relicense MAUD, FActScore, Llama, Goodfire, or judge-model outputs. Check upstream terms before redistributing source datasets, raw annotations, or regenerated artifacts.

## Citation

Methods paper placeholder: *Activation Probes Recover Claim-Level Correctness Signal Across Legal and Biographical QA*.
