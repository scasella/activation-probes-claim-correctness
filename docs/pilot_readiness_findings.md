# Pilot Readiness Findings

## Result

The rebuilt mixed-domain pilot is **not annotation-ready**.

The rebuilt artifacts fixed the substrate issues that mattered most:

- pilot rows are now isolated from the held-out test split
- the source pool is contract-diverse on both MAUD and CUAD
- generated answers no longer contain special-token artifacts
- answer-run bundles are persisted separately from generated claims
- generated claims no longer carry placeholder correctness/load-bearing labels

## Mixed Pilot Readiness Outcome

The rebuilt mixed pilot failed the bounded CUAD readiness gate.

Current machine-readable summary:

- `split=test` rows in pilot: `0`
- CUAD pilot examples: `15`
- CUAD examples with suspect answers: `9`
- CUAD examples with junk claims: `7`
- total junk claims in mixed pilot: `13`
- readiness heuristic: `false`

The dominant remaining failure mode is task mismatch on the CUAD side. Even after prompt cleanup, many CUAD answers remain off-target or clause-fragment-like, which makes the canonical claim list too unstable for a fair mixed-domain reliability pilot.

## Fallback Decision

Per the bounded-remediation rule, the project now falls back to a `MAUD-first`, `correctness-first` Phase 1 pilot.

Local fallback artifacts have been materialized under `data/annotations/`:

- `maud_pilot_examples.jsonl`
- `maud_pilot_claims.jsonl`
- `maud_pilot_annotation_packet.jsonl`
- `maud_pilot_readiness_summary.json`

The MAUD-only readiness heuristic currently passes.

## What This Means

- The current mixed pilot remains useful as diagnostic evidence only.
- CUAD is not deleted from the project; it is now a documented failure mode and a candidate follow-up lane that likely needs a task-specific evidence-window and extraction pipeline.
- The next annotation work should happen on the MAUD-only pilot, with correctness as the primary reliable target.
