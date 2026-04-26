# Activation Probes for Claim-Level Correctness

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C.svg)](https://pytorch.org/)
[![Backbone: Llama 3.1 8B](https://img.shields.io/badge/backbone-Llama%203.1%208B-ff6b6b.svg)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
[![Library: ProbeMon](https://img.shields.io/badge/library-ProbeMon-9cdcfe.svg)](library/README.md)
[![MAUD ΔAUROC: +0.260](https://img.shields.io/badge/MAUD%20%CE%94AUROC-%2B0.260-3dd68c.svg)](docs/paper_draft.md)
[![FActScore ΔAUROC: +0.262](https://img.shields.io/badge/FActScore%20%CE%94AUROC-%2B0.262-3dd68c.svg)](docs/paper_draft.md)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-0a9edc.svg)](tests/)

This repository tests whether simple linear probes on Llama 3.1 8B hidden activations recover claim-level correctness signal that the model's own structured self-report misses. The public branch exposes three main surfaces:

- A methods paper draft and browser-readable report on MAUD legal QA and FActScore biography factuality.
- Reproducible experiment scripts for MAUD, FActScore, FELM boundary checks, and probe interpretation.
- `ProbeMon`, a small Python library with two domain-specific pretrained probes and training infrastructure for new labeled claim datasets.

This is research code. Probe scores are not legal advice, factual ground truth, or a universal hallucination detector.

## Headline Results

The core finding is bounded but consistent across the two strongest settings:

| Dataset | Label source | Llama self-report AUROC | Residual probe AUROC | Paired AUROC delta |
| --- | --- | ---: | ---: | ---: |
| MAUD merger-agreement QA | GPT-5.4 judge proxy | 0.511 | 0.771 | 0.260 |
| FActScore biographies | Human atomic-fact labels | 0.541 | 0.802 | 0.262 |
| FELM world-knowledge QA | Human segment labels | 0.511 | 0.652 | 0.141, inconclusive |

AUROC is a ranking score. An AUROC of `0.50` is random guessing; `1.00` is perfect ranking. Here, it answers: if we pick one correct claim and one incorrect claim, how often does the method give the correct claim the better score? The paired AUROC delta is the residual probe's AUROC minus Llama self-report's AUROC on the same claims, so a delta around `0.26` means the hidden-activation probe is substantially better at separating correct from incorrect claims.

The MAUD second-judge sensitivity pass preserves the method ordering under Kimi K2.6 labels and shows that GPT-5.4 external scoring is inflated by same-family judge coupling: AUROC drops from 0.944 under the GPT-5.4 judge to 0.872 under Kimi. FELM remains a documented boundary case because its reference-bounded labels often measure evidence coverage rather than claim correctness.

Start with:

- `docs/paper_draft.md` for the methods-paper draft.
- `docs/paper_blog.html` for the standalone blog-style webpage.
- `artifacts/runs/maud_proxy_findings.html` for the earlier MAUD-focused report.
- `docs/factscore_validation.md` and `docs/felm_validation.md` for transfer and boundary notes.
- `library/README.md` for ProbeMon.

## Repository Layout

- `configs/`: experiment settings and prompt templates.
- `src/interp_experiment/`: reusable pipeline code for data prep, generation, activations, evaluation, and reporting.
- `scripts/`: command-line entrypoints for materialization, evaluation, bootstrap CIs, transfer checks, and report generation.
- `docs/`: paper drafts, methods notes, validation reports, and release documentation.
- `artifacts/runs/`: curated aggregate release artifacts only. Raw logs, generated answers, feature matrices, and batch outputs are intentionally ignored.
- `demo/`: static MAUD probe-uncertainty demo.
- `library/`: ProbeMon package, tests, docs, and compact pretrained probe artifacts.
- `tests/`: lightweight validation tests for schemas, metrics, leakage controls, and package behavior.

## Quickstart

Create a development environment:

```bash
uv sync --group dev
```

Run the main test suite:

```bash
uv run pytest
```

Run the combined main-repo plus ProbeMon tests:

```bash
PYTHONPATH=src:library pytest -q tests library/tests
```

List the packaged ProbeMon probes:

```bash
PYTHONPATH=library python -m probemon.cli list-probes
```

Build the ProbeMon wheel without network dependency resolution:

```bash
cd library
python -m pip wheel . --no-deps -w /tmp/probemon-wheel
```

## Reproducing Experiments

The public release does not include raw datasets, raw LLM judge responses, raw human audit packets, generated answer runs, Modal logs, or large activation feature matrices. Reproduction scripts expect you to regenerate or restore those local artifacts.

Common entrypoints:

```bash
uv run python scripts/phase0_smoke.py --help
uv run python scripts/materialize_judge_annotations.py --help
uv run python scripts/evaluate_baselines_against_labels.py --help
uv run python scripts/evaluate_probe_proxy_smoke.py --help
uv run python scripts/compute_bootstrap_ci.py --help
```

GPU-backed materialization uses Modal through `scripts/modal_gpu.py`. Credential-aware scripts load repo-local `.env` values when present, but `.env` is intentionally excluded from the public release.

Expected local environment variables for full regeneration include:

- `HF_TOKEN` for gated Hugging Face model access.
- `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` for Modal jobs.
- `PRIME_API_KEY` for Prime Intellect second-judge inference, if rerunning that lane.

## Artifact Policy

This release keeps only small, publication-facing artifacts:

- aggregate metric JSON files;
- report HTML/Markdown files;
- compact probe-direction artifacts;
- ProbeMon pretrained probe `.npz` files;
- small fixtures needed by tests or demos.

It excludes:

- raw judge prompts/responses;
- raw human audit packets and labels;
- generated answer JSONL files;
- large residual/SAE feature matrices;
- raw dataset archives and converted private working data;
- Modal logs, retry batches, and local caches.

If you rerun experiments, generated files should remain ignored unless they are intentionally promoted as a small aggregate release artifact.

## License And Data Terms

Repository code is released under the MIT License. That license does not relicense third-party datasets, model weights, model outputs, or derived artifacts governed by upstream terms.

MAUD/Atticus Project data, FActScore data, FELM data, Llama model weights, Goodfire SAE artifacts, Prime Intellect model metadata, and any generated LLM judge outputs remain subject to their respective upstream licenses and usage terms. Check those terms before redistributing raw data or derived annotations.

## Limitations

- All primary probes target Llama 3.1 8B Instruct at layer 19.
- MAUD labels are judge-proxy labels, not full expert legal ground truth.
- FActScore uses ChatGPT-generated biographies read by Llama, so it is a transfer check rather than a matched-generation experiment.
- FActScore atomic facts are mapped to parent-sentence spans because exact atomic-fact substring matching is rare.
- FELM is a boundary case, not a shipped ProbeMon domain.
- The interpretability follow-up suggests the residual correctness signal is distributed rather than localized to a compact set of SAE features.

## Citation

Paper draft placeholder:

```text
Activation Probes Recover Claim-Level Correctness Signal Across Legal and Biographical QA.
```

Use `docs/paper_references.bib` for current bibliography entries while the paper is in draft form.
