# Phase 0 Findings

## Status

Phase 0 live smoke completed on Modal with a single `A100-80GB` sandbox using:

- `meta-llama/Llama-3.1-8B-Instruct`
- TransformerLens as the primary activation path
- Goodfire SAE loaded through SAELens

The smoke completed the intended path:

- Llama answer generation
- residual-stream extraction at layer 19
- Goodfire SAE encoding
- canonical-claim construction
- synthetic cached-baseline JSON parse and prompt-packet generation

## Reality mismatches resolved

### Goodfire SAE id

The repo initially assumed SAE id `llama3.1-8b-it/19-resid-post-gf`.

The live SAELens release `goodfire-llama-3.1-8b-instruct` currently exposes:

- SAE id `layer_19`

Repo defaults were updated accordingly.

### CUAD QA source

The legacy dataset id `theatticusproject/cuad-qa` is script-based and rejected by the current `datasets` runtime.

The repo now uses:

- `chenghao/cuad_qa`

This dataset matches the expected SQuAD-style QA schema and supports the CUAD side of the source pool.

### Modal GPU packaging

The straightforward `uv_sync` path pulled a PyTorch build incompatible with the current Modal driver stack for the A100 sandbox.

The runner now installs a curated inference/interp runtime in Modal:

- `torch==2.6.0` from the CUDA 12.4 PyTorch index
- `transformers`, `accelerate`, `sae-lens`, `transformer-lens`, and core numeric deps

This made the live smoke reproducible inside Modal.

## Notes from the smoke output

- TransformerLens remained the primary extractor after a small fix to clone the generated token tensor before `run_with_cache`.
- The phase-0 prompt produced a repetitive long answer tail and an `Answer:` prefix that became its own canonical claim.
- Before large-scale answer generation, the generation prompt or decode settings should be tightened so claim construction is cleaner:
  - stronger instruction to omit answer labels
  - lower `max_new_tokens`
  - explicit short-answer constraint

## Why this matters

These are not cosmetic changes. They are the concrete environment and artifact mismatches that would have blocked a clean first pass if left undocumented. The repo now reflects the live-resolved versions instead of the stale assumptions.
