# MAUD Follow-Up Decision Memo

Date: 2026-04-24

## Current Decision

Do not submit the paper as human-validated yet. The two-judge result is strong enough for a methods draft, but the final validity claim depends on the frozen 30-claim human audit.

## If The Audit Supports The LLM Judges

If humans agree strongly with the two-judge agreement stratum and do not strongly favor one LLM judge in disagreement cases, the next step is a submission pass:

- Fill in the human-audit section in `docs/maud_paper_draft.md`.
- Add human-subset AUROC CIs to `artifacts/runs/maud_bootstrap_ci.json`.
- Reword the conclusion from "proxy-label evidence" to "proxy-label evidence with a small human validation anchor."
- Keep the scope narrow: one domain, one base model, one SAE, small audit.

This would support a compact methods paper about claim-level uncertainty signals in legal QA.

## If Humans Prefer One Judge

If humans match GPT-5.4 or Kimi much more often on judge-disagreement claims, the paper should pivot to a judge-sensitivity result:

- Report the favored judge plainly.
- Treat the other judge's errors as evidence about prompt portability or model-family bias.
- Recompute the human-subset AUROCs and paired deltas, but do not retrain probes.
- Emphasize that the probe result is meaningful only insofar as it aligns with the better-supported label surface.

This is still publishable if the method ordering mostly survives.

## If Humans Disagree With Both Judges

If humans frequently disagree with both LLM judges, lead with that negative result:

- Do not claim stable legal correctness detection.
- Frame the study as showing that LLM judge-proxy labels can be unstable in legally nuanced claim scoring.
- Use the activation-probe findings as diagnostic, not as validated correctness detection.
- Propose a larger expert-label study before further modeling work.

This would be less clean, but scientifically valuable.

## Concrete Next Step

Send `data/annotations/maud_human_audit_packet.jsonl` and the rubric in `docs/human_audit_protocol.md` to one or two legal-literate annotators. When labels return, write them to `data/annotations/maud_human_audit_labels.jsonl`, then run:

```bash
uv run python scripts/analyze_maud_human_audit.py
uv run python scripts/compute_bootstrap_ci.py
```

Only after that should the paper be called submission-ready.
