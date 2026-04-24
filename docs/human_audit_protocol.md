# MAUD Human Audit Protocol

Date: 2026-04-24

## Purpose

This audit anchors the two-judge MAUD proxy-label result against a small set of human legal-literacy judgments. The audit is validation only. It must not be used to train probes, tune prompts, select claims after the fact, or revise the frozen 150-claim corpus.

## Packet

The blinded annotation packet is `data/annotations/maud_human_audit_packet.jsonl`.

Each row shows:

- MAUD contract excerpt
- Question
- Llama answer
- One extracted claim
- The same three-way correctness rubric used by the LLM judges

The packet intentionally excludes GPT-5.4 labels, Kimi K2.6 labels, judge-agreement status, and sampling stratum. The unblinded reproducibility manifest is `artifacts/runs/maud_human_audit_sample_manifest.json`.

## Sample Composition

The packet contains 30 distinct claims sampled from the frozen 150-claim set with seed `240424`.

Final composition:

- 20 judge-agreement claims
- 10 judge-disagreement claims
- 11 both-judge-true claims
- 7 both-judge-partially-true claims
- 2 both-judge-false claims

The composition differs from the requested nominal strata because the both-true and both-false/partially-true strata are subsets of the agreement set. After deduplication, the sampler topped up from underrepresented available claims to preserve 30 distinct claims.

## Annotator Instructions

Annotators should have legal literacy: law student, paralegal, lawyer, or comparable legal-research experience.

For each row, assign exactly one `correctness_label`:

- `true`: the extracted claim is fully supported by the excerpt in context.
- `partially_true`: the claim captures part of the truth but omits or distorts a material condition, carveout, timing requirement, or qualifier.
- `false`: the claim is contradicted or unsupported by the excerpt.

Annotators must use only the excerpt, question, Llama answer, and extracted claim shown in the packet. They should add a brief justification. They should not search externally or infer missing contract context.

## Analysis

Completed labels should be written to `data/annotations/maud_human_audit_labels.jsonl` using the template rows already created there. Then run:

```bash
uv run python scripts/analyze_maud_human_audit.py
uv run python scripts/bootstrap_maud_ci.py
```

The audit report must state the audit N wherever audit-derived numbers appear. If two annotators are used, compute human-human Cohen's kappa before using adjudicated labels as validation evidence.
