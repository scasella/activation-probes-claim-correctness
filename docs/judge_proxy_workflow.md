# Judge Proxy Workflow

This workflow uses a judge LLM in place of human annotation for the MAUD pilot.

It is a **proxy-label** lane, not expert ground truth.

## Materialize proxy labels

Run:

```bash
uv run python scripts/materialize_judge_annotations.py \
  --examples-jsonl data/annotations/maud_pilot_examples.jsonl \
  --claims-jsonl data/annotations/maud_pilot_claims.jsonl
```

Outputs:

- Proxy annotations: `data/annotations/maud_pilot_judge_annotations.jsonl`
- Raw judge outputs: `data/annotations/judge_llm_raw/maud_pilot/`
- Summary: `artifacts/runs/maud_judge_annotation_summary.json`

This path uses Codex App Server with `gpt-5.4`.

## Score precomputed baselines against proxy labels

Run:

```bash
uv run python scripts/evaluate_baselines_against_labels.py \
  --labels-jsonl data/annotations/maud_pilot_judge_annotations.jsonl \
  --llama-predictions data/cached_baselines/llama_self_report/parsed/maud_pilot/_all_predictions.jsonl \
  --gpt54-predictions data/cached_baselines/gpt54/parsed/maud_pilot \
  --output-json artifacts/runs/maud_proxy_baseline_eval.json \
  --label-source judge_llm_proxy
```

## Caveat

These numbers answer a different question than expert-ground-truth evaluation:

- useful for iteration
- useful for proxy comparisons
- not equivalent to the original human-annotation study claim
