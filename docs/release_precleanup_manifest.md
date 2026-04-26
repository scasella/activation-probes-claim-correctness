# Public Release Pre-Cleanup Manifest

Date: 2026-04-26
Branch target: `codex/public-release-prep`
Source branch before release prep: `main`

This manifest records the repository state observed before public-release cleanup so the curated release surface can be audited later.

## Dirty Tracked Files

Tracked files with local modifications before cleanup:

- `artifacts/runs/maud_proxy_findings.html`
- `docs/maud_paper_draft.md`
- `docs/next_steps.md`
- `scripts/materialize_claim_targets.py`
- `scripts/materialize_gpt54_baseline.py`
- `scripts/materialize_judge_annotations.py`
- `scripts/modal_gpu.py`
- `scripts/remote_claim_targets.py`
- `scripts/remote_probe_features.py`
- `src/interp_experiment/activations/extractors.py`
- `src/interp_experiment/baselines/utils.py`
- `src/interp_experiment/codex_app_server_client.py`
- `src/interp_experiment/evaluation/metrics.py`

## Major Untracked Work Areas

Untracked areas present before cleanup:

- `library/`: ProbeMon v0.1 package, docs, tests, and pretrained probe artifacts.
- `data/factscore/` and `data/felm/`: local converted/raw dataset working files.
- `docs/paper_*.md`, `docs/paper_*.html`, `docs/factscore_*.md`, `docs/felm_*.md`, and `docs/library_design_notes.md`: paper, transfer-report, and library documentation drafts.
- `scripts/*factscore*`, `scripts/*felm*`, and batched materialization helpers: follow-on experiment and materialization scripts.

## Ignored Local/Generated State

Ignored local state observed before cleanup included:

- Local credentials and environments: `.env`, `.venv/`.
- macOS/cache/build debris: `.DS_Store`, `__pycache__/`, `.pytest_cache/`, `src/interp_experiment.egg-info/`, `library/build/`.
- Generated run artifacts under `artifacts/runs/`, including logs, answer runs, feature matrices, batch directories, and retry/shard outputs.
- Generated/raw data under `data/source_pool/`, `data/annotations/`, `data/cached_baselines/`, `data/factscore/`, and `data/felm/`.
- Local dependency lock: `uv.lock` was present and ignored before cleanup.

## Large Files Observed

Large local/generated files observed before cleanup included:

- `artifacts/runs/factscore_chatgpt_probe_features_residual.npz` (~16 MB)
- `artifacts/runs/factscore_chatgpt_interim_probe_features_residual.npz` (~13 MB)
- `artifacts/runs/maud_full_probe_features_residual.jsonl` (~13 MB)
- `artifacts/runs/source_pool_smoke.jsonl` (~6.1 MB)
- `data/source_pool/examples.jsonl` (~6.8 MB)
- `data/factscore/factscore_chatgpt_claims.jsonl` (~3.0 MB)
- `data/factscore/raw/data.zip` (~2.1 MB)
- Multiple Modal `remote.log` files between ~1 MB and ~2 MB.

## Public-Release Policy Applied

The release cleanup should keep code, final aggregate reports, paper/blog/demo docs, small summaries, and small pretrained probe artifacts. It should exclude raw judge prompts/responses, raw human/audit packets and labels, raw dataset archives, large feature matrices, generated answer JSONL, Modal logs, local credentials, local virtualenvs, and generated build/cache output.
