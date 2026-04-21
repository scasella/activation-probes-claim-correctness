# MAUD Baseline Workflow

This workflow precomputes everything on the MAUD-only lane up to, but not including, ground-truth evaluation.

## Inputs

- `data/annotations/maud_pilot_examples.jsonl`
- `data/annotations/maud_pilot_claims.jsonl`

## Llama self-report baseline

Run:

```bash
uv run python scripts/materialize_llama_self_report.py \
  --examples-jsonl data/annotations/maud_pilot_examples.jsonl \
  --claims-jsonl data/annotations/maud_pilot_claims.jsonl
```

Outputs:

- Raw completions: `data/cached_baselines/llama_self_report/raw/maud_pilot/`
- Parsed predictions: `data/cached_baselines/llama_self_report/parsed/maud_pilot/`
- Summary: `artifacts/runs/maud_llama_self_report_summary.json`

## GPT-5.4 cached baseline

Run:

```bash
uv run python scripts/materialize_gpt54_baseline.py \
  --examples-jsonl data/annotations/maud_pilot_examples.jsonl \
  --claims-jsonl data/annotations/maud_pilot_claims.jsonl
```

Outputs:

- Request packets: `data/cached_baselines/gpt54/requests/maud_pilot/`
- Raw GPT-5.4 outputs: `data/cached_baselines/gpt54/raw/maud_pilot/`
- Parsed predictions: `data/cached_baselines/gpt54/parsed/maud_pilot/`
- Summary: `artifacts/runs/maud_gpt54_summary.json`

This path now uses Codex App Server for the GPT-5.4 calls.

## One-shot command

To run both materializers in sequence:

```bash
uv run python scripts/precompute_maud_baselines.py
```

## What this does not do

- It does **not** use human annotation labels.
- It does **not** compute correctness or load-bearing evaluation against ground truth.
- It does **not** decide the study result.

Those steps remain blocked on completed human annotation files.
