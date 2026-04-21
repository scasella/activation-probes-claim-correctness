# Interpretability-Derived Uncertainty Decomposition

This repository contains the bounded proof-of-concept experiment for comparing:

1. Llama 3.1 8B Instruct self-reported claim uncertainty
2. GPT-5.4 external claim decomposition and uncertainty scoring
3. Probe-based uncertainty from Llama activations, with Goodfire SAE features and raw residual baselines

The experiment is intentionally frozen around a small number of public artifacts:

- `docs/abstract_v1.md`: the pre-data abstract
- `study_protocol.md`: phase gates, stop conditions, and locked defaults
- `configs/*.yaml`: experiment settings and prompts
- `src/` + `scripts/`: executable pipeline code

## Working conventions

- `uv sync --group dev` creates the local development environment.
- `uv run pytest` runs local tests.
- `uv run python scripts/phase0_smoke.py --help` shows the phase-0 smoke interface.
- `uv run python scripts/export_gpt54_requests.py ...` prepares cached GPT-5.4 baseline requests.
- `uv run python scripts/materialize_llama_self_report.py ...` materializes local Llama self-report baseline artifacts.
- `uv run python scripts/materialize_gpt54_baseline.py ...` materializes GPT-5.4 cached baseline artifacts through the local Codex CLI.
- `uv run python scripts/precompute_maud_baselines.py` runs the MAUD baseline precompute lane end to end.
- `uv run python scripts/export_annotator_packets.py ...` writes one packet file per annotator from the combined MAUD pilot packet.
- `uv run python scripts/evaluate_annotation_pilot.py ...` computes agreement metrics and a Markdown report from completed annotation files.
- `uv run python scripts/export_disagreement_packet.py ...` exports disagreement rows for adjudication after agreement evaluation.

Credential-aware scripts load repo-local `.env` automatically for `HF_TOKEN`, `MODAL_TOKEN_ID`, and `MODAL_TOKEN_SECRET`, while keeping secret values out of artifacts and terminal summaries.

The GPT-5.4 baseline is treated as a cached artifact workflow. Modal reproducibility starts from committed prompt packets plus cached raw/parsed baseline outputs; it does not try to regenerate GPT-5.4 calls inside Modal.

## Layout

- `configs/`: dataset, prompt, probe, compute, and report settings
- `data/`: local working directories for source pools, annotations, and cached baselines
- `docs/`: frozen study documents and annotation guidance
- `scripts/`: command-line entrypoints
- `src/interp_experiment/`: library code
- `tests/`: lightweight validation of schemas, splitting, metrics, and leakage prevention
