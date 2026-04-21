# Annotation Guidelines

## Purpose

Annotators work on canonical claims derived from Llama answers to legal contract questions. Public dataset answers are seeds only. Expert judgment controls final labels.

## Required fields per claim

- `correctness_label`: `true`, `false`, or `partially_true`
- `load_bearing_label`: `yes` or `no`
- `flip_evidence_text`: short description of evidence that would change the conclusion, required when `load_bearing_label=yes`

## Correctness rubric

- `true`: the claim is supported by the excerpt and question context
- `false`: the claim is contradicted by the excerpt or materially unsupported
- `partially_true`: the claim mixes correct and incorrect content, or is directionally right but materially overstated

Annotators should label the atomic claim as written, not the conclusion they think the model meant to express.

## Load-bearing rubric

A claim is load-bearing if changing that claim would likely change the answer's overall conclusion to the question.

Use `no` when:

- the claim is merely descriptive background
- the claim duplicates another stronger claim
- the claim adds detail without affecting the final answer

Use `yes` when:

- the claim is a direct premise for the conclusion
- the claim resolves the key legal interpretation
- the claim determines whether the answer is yes/no or materially changes the legal outcome

## Flip evidence

When a claim is load-bearing, write what evidence would flip the conclusion. Keep it specific:

- cite the missing clause type or contractual language
- describe the contrary wording or fact pattern
- avoid generic text like "more context needed"

## Pilot policy

- Pilot size: `30` examples
- Annotators: `2`
- Reliability metric: Cohen's kappa or Krippendorff's alpha on load-bearingness
- Threshold: `0.6`
- Allowed adjustment: one guideline revision followed by one repilot

If the repilot remains below threshold, the main study drops load-bearing as a target and proceeds with correctness-only labeling.
