# Matched-Generation FELM Validation

## Summary

Phase 0 pilot completed, but the annotation gate did not pass cleanly enough to justify scaling yet. The pipeline generated Llama 3.1 8B Instruct answers for 20 reference-available FELM world-knowledge prompts, segmented them with `spacy_en_core_web_sm`, and judged 52 generated segments with GPT-5.4 via Codex App Server against cached FELM reference evidence. The mechanics worked, but the label distribution and inspection show a validity problem: only 2/52 segments were labeled `true`, 7/52 `partially_true`, and 43/52 `false`. Many `false` labels were not obvious factual contradictions; they were cases where the cached FELM reference snippet did not contain enough evidence for a newly generated Llama claim.

The matched-generation experiment should not be scaled until the annotation target is repaired. As currently prompted, the judge is partly measuring reference-coverage mismatch, not just claim correctness.

## What Changed From ChatGPT-FELM

The previous FELM validation used FELM's original ChatGPT responses and human segment labels. This pilot changed the generation source: Llama generated new answers from the same FELM prompts, and those answers were segmented into sentence-level claims. Because the original FELM labels no longer align with the new claims, labels were produced by a GPT-5.4 judge using FELM reference evidence.

Reference handling used FELM's embedded `ref_contents` cache first and live-fetched missing URLs where possible. Across the full `wk` set, 163/183 examples have usable cached reference content after fetch attempts, so reference availability is not the main blocker. The blocker is evidence adequacy: many cached snippets are too thin for judging arbitrary regenerated claims.

## Pilot Artifacts

| Artifact | Value |
| --- | ---: |
| Pilot examples | 20 |
| Generated answers | 20 |
| Segments | 52 |
| Judge-labeled segments | 52 |
| Judge model | GPT-5.4 via Codex App Server |
| Segmenter | `spacy_en_core_web_sm` |
| True labels | 2 |
| Partially true labels | 7 |
| False labels | 43 |

Relevant files:

- `data/felm/felm_wk_matched_pilot_examples.jsonl`
- `data/felm/felm_wk_matched_pilot_segments.jsonl`
- `data/felm/felm_wk_matched_pilot_reference_cache.jsonl`
- `data/annotations/felm_wk_matched_pilot_judge_labels.jsonl`
- `artifacts/runs/felm_wk_matched_pilot_judge_summary.json`

## Gate Decision

The pilot should stop here. The judge outputs are mostly internally coherent: when the reference directly contradicts a claim, the judge catches it; when a claim is overbroad, it often uses `partially_true`. But the references frequently fail to cover claims that may be true or false in the world. Under the current prompt, those become `false`. That makes the label set too dependent on what the cached snippet happens to include.

This is not a probe failure. It is an annotation-target failure. Scaling would produce numbers, but those numbers would answer a muddier question: "Do activations predict support in the available reference snippets?" rather than "Do activations predict factual correctness on matched-generation FELM?"

## Recommended Repair

Use one of these before scaling:

1. Fetch fuller reference pages with a readability extractor and rerun the same pilot. This is the smallest repair if URL content is available.
2. Add an explicit `not_enough_evidence` judge label, then exclude or analyze those rows separately rather than folding them into `false`.
3. Use a small human audit for the matched-generation pilot if fuller references remain too sparse.

The library decision should stay unchanged from the previous FELM result for now: FELM is a candidate scaffolding dataset, but not yet a confirmed transfer replication. Matched-generation remains the right hypothesis test, but Option A needs a stronger evidence surface before it can answer the question.

## Methods-Paper Paragraph Draft

We attempted a matched-generation FELM pilot to test whether the smaller FELM effect was caused by probing Llama activations over ChatGPT-generated text. Llama 3.1 8B Instruct generated new answers for 20 reference-available FELM world-knowledge prompts, which were segmented into 52 sentence-level claims and judged by GPT-5.4 against FELM reference evidence. The pilot exposed an annotation validity issue: many generated claims were labeled false because the cached reference snippets did not contain enough evidence, not because the claims were clearly contradicted. We therefore did not scale this pass. This leaves generation mismatch unresolved and motivates either fuller reference retrieval or a small human-labeled matched-generation audit before using FELM as a transfer claim.
