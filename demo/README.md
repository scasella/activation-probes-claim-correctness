# Claim-Level Probe Uncertainty Demo

This is a static demo for the MAUD judge-proxy methods result: linear probes on Llama 3.1 8B hidden activations can flag claim-level correctness risk that Llama's own structured confidence does not expose.

The demo is intentionally precomputed. It loads `web/examples.json` in the browser and does not call a model server, GPU worker, or API.

## What The Demo Shows

The page shows 8 frozen MAUD merger-agreement QA examples. Click an example question to view Llama's answer. Each sentence is colored on a continuous red-to-green scale using a calibrated residual-probe score:

- redder means lower probe score
- greener means higher probe score
- hover or tap a sentence to see raw and calibrated scores

The first version aggregates overlapping frozen claim-level probe scores onto answer sentences. It does not rerun live generation.

## Disclaimer Text

This is the exact disclaimer shown in the demo:

> This is a research demo. The probe was trained on Llama 3.1 8B activations during merger-agreement question answering, supervised on labels from an LLM judge. Flag colors reflect what the probe learned to associate with judge-assigned correctness on this domain. Do not use this for legal decisions.

## Selected Examples

The examples are selected deterministically from non-test MAUD examples to cover a mix of score patterns. The current source IDs are:

| MAUD source ID | Split | Why included |
| --- | --- | --- |
| `maud-88ea798310ca` | train | Mixed low/high sentence scores |
| `maud-f44781413158` | train | Mixed low/high sentence scores |
| `maud-1c2624700335` | validation | Mixed low/high sentence scores |
| `maud-6a7280b2d481` | train | Mostly low probe score |
| `maud-0201dde23af2` | train | Mostly low probe score |
| `maud-656c0af11083` | train | Mostly high probe score |
| `maud-0f73330d9ab2` | train | Mostly high probe score |
| `maud-1d61b4637adb` | train | Mixed low/high sentence scores |

All source rows come from the frozen MAUD artifacts under `artifacts/runs/`.

## Regenerating `examples.json`

From the repo root:

```bash
uv run python demo/build_examples.py
```

This reads:

- `artifacts/runs/maud_full_examples.jsonl`
- `artifacts/runs/maud_full_answer_runs.jsonl`
- `artifacts/runs/maud_full_claims.jsonl`
- `artifacts/runs/maud_full_probe_features_residual.jsonl`
- `data/annotations/maud_full_judge_annotations.jsonl`
- `artifacts/runs/residual_correctness_probe_direction.npz`

Those source inputs are local regeneration artifacts and are not all included in the curated public release. Restore or regenerate them before rebuilding the demo data.

It writes:

- `demo/web/examples.json`
- `demo/probe_calibration.json`

The calibration is Platt scaling fit on the methods-paper train split using raw residual-probe logits and judge-1 labels. Do not retune it on the demo examples.

## Local Preview

Serve the static folder:

```bash
python3 -m http.server 8787 --directory demo/web
```

Then open:

```text
http://localhost:8787
```

## Deployment

Any static host works. For example, deploy `demo/web/` to GitHub Pages, Netlify, or Vercel. There is no backend.

Current deployment:

- https://scasella.github.io/maud-probe-demo/
- https://github.com/scasella/maud-probe-demo
