# Public Release Checklist

This checklist defines the expected public-release surface for the repository.

## Release Surface

Keep public:

- Source code under `src/`, runnable scripts under `scripts/`, tests, and configs.
- Paper/report docs: `docs/paper_draft.md`, `docs/paper_blog.html`, validation reports, methods notes, and interpretation reports.
- Static demo code under `demo/` and its precomputed public examples.
- ProbeMon under `library/`, including its compact pretrained probe artifacts.
- Curated aggregate artifacts under `artifacts/runs/`: metric JSON, report HTML, compact probe direction/summary files, and release-facing bootstrap/evaluation summaries.

Exclude public:

- `.env`, `.venv/`, `.DS_Store`, package build directories, pycache, and test caches.
- Raw judge prompts/responses and raw human audit packets or labels.
- Generated answers, generated claim JSONL, raw source-pool data, cached baseline raw/parsed predictions, Modal logs, shard/batch retries, and large activation feature matrices.
- Raw dataset downloads and converted local working datasets under `data/factscore/` and `data/felm/`.

## Required Credentials For Regeneration

Only needed when rerunning pipelines, not for reading the public repo:

- `HF_TOKEN`: gated Hugging Face model and dataset access.
- `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`: Modal execution.
- `PRIME_API_KEY`: Prime Intellect second-judge reruns.

Do not commit credential files. Repo-local `.env` is intentionally ignored.

## Verification Commands

Run from the repository root:

```bash
uv run pytest
PYTHONPATH=src:library pytest -q tests library/tests
uv run python scripts/phase0_smoke.py --help
PYTHONPATH=library python -m probemon.cli list-probes
```

Build ProbeMon without dependency resolution:

```bash
cd library
python -m pip wheel . --no-deps -w /tmp/probemon-wheel
```

## Public-Surface Audit

Before publishing, verify:

```bash
git status --short
git status --short --ignored
find . -path ./.git -prune -o -path ./.venv -prune -o -type f -size +5M -print
git ls-files -z | xargs -0 grep -IEn 'sk-[A-Za-z0-9_-]{20,}|OPENAI_API'_'KEY=[^[:space:]]+|ANTHROPIC_API'_'KEY=[^[:space:]]+|PRIME_API'_'KEY=[^[:space:]]+|MODAL'_'TOKEN(_ID|_SECRET)?=[^[:space:]]+|HF'_'TOKEN=[^[:space:]]+|hf_[A-Za-z0-9]{20,}'
```

Expected results:

- Large files should be either ignored local regeneration artifacts or intentionally small public artifacts.
- The grep may find environment variable names in code/docs, but it must not find secret values.
- `git status --short` should show only intentional release changes.

## Known Release Limitations

- The human MAUD audit is represented as pending aggregate status, not completed human validation.
- FActScore is a transfer result over ChatGPT-generated biographies read by Llama, not a matched-generation Llama result.
- FELM is documented as a boundary case and is not shipped as a ProbeMon pretrained domain.
- ProbeMon v0.1 supports only Llama 3.1 8B Instruct layer-19 residual probes.
- Raw upstream data and raw annotations are intentionally absent from the public release; reproduction requires regeneration or local restoration.
