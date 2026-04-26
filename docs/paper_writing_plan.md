# Paper Writing Plan

## Paper Spine

The paper argues that Llama 3.1 8B's structured self-reported confidence does not reliably rank claim-level correctness, while linear probes on layer-19 residual activations recover correctness signal in two materially different settings: MAUD merger-agreement question answering and FActScore biography factuality. FELM remains a boundary case: the direction matches the main result, but the effect is smaller and the matched-generation repair exposed a reference-bounded annotation target mismatch.

The central claim is methodological, not legal or factual omniscience: hidden activations contain recoverable claim-level correctness signal that the model's verbal self-report fails to expose.

## Source Artifacts

| Evidence area | Source artifacts |
| --- | --- |
| MAUD judge-1 baseline and self-report | `artifacts/runs/maud_full_proxy_baseline_eval.json` |
| MAUD judge-1 residual and SAE probes | `artifacts/runs/maud_full_probe_proxy_smoke.json` |
| MAUD Kimi K2.6 second-judge baseline | `artifacts/runs/maud_full_proxy_baseline_eval_v2.json` |
| MAUD Kimi K2.6 residual and SAE probes | `artifacts/runs/maud_full_probe_proxy_smoke_v2.json` |
| MAUD judge agreement | `artifacts/runs/maud_judge_agreement_analysis.json` |
| MAUD agreement-set evaluation | `artifacts/runs/maud_agreement_set_eval.json` |
| MAUD 1000-resample CIs | `artifacts/runs/maud_bootstrap_ci_v2.json` |
| FActScore adapter and span policy | `artifacts/runs/factscore_chatgpt_adapter_summary.json`, `artifacts/runs/factscore_chatgpt_token_alignment_summary.json` |
| FActScore materialization and evaluation | `artifacts/runs/factscore_chatgpt_materialization_summary.json`, `artifacts/runs/factscore_chatgpt_validation_eval.json` |
| FActScore 1000-resample CIs | `artifacts/runs/factscore_bootstrap_ci.json` |
| FELM minimum viable validation | `artifacts/runs/felm_wk_validation_eval.json`, `docs/felm_validation.md` |
| FELM matched-generation repair | `docs/felm_matched_generation_repair.md`, `artifacts/runs/felm_wk_matched_pilot_judge_summary_v2.json` |
| Human audit status | `data/annotations/maud_human_audit_packet.jsonl`, `artifacts/runs/maud_human_audit_analysis.json` |
| Interpretability follow-up | `docs/probe_interpretation.md`, `docs/probe_interpretation_gate.md` |

## Section Mapping

| Section | Main job | Artifact basis |
| --- | --- | --- |
| Abstract | State the two-domain effect and boundary case in numbers. | MAUD and FActScore CI artifacts, FELM validation report |
| Introduction | Motivate claim-level uncertainty and why self-report is not enough. | Cross-dataset result synthesis |
| Related Work | Place the work between verbal calibration, truth probes, semantic entropy, and mechanistic interpretability. | External citations in `docs/paper_references.bib` |
| Method | Describe frozen claims, self-report baseline, layer-19 residual probes, FActScore sentence-span pooling, FELM adaptation, and paired bootstrap. | Scripts and adapter/evaluation artifacts |
| Results | Present MAUD table, second-judge sensitivity, FActScore replication, FELM boundary, and cross-dataset comparison. | Bootstrap and evaluation artifacts |
| Discussion | Interpret label provenance as the central transfer variable. | MAUD/FActScore/FELM comparison |
| Threats to Validity | Keep boundaries explicit: one base model, one layer, proxy labels, generation mismatch, sentence-span pooling, no layer sweep, distributed SAE signal. | Human audit status and interpretability report |
| Conclusion | Three-to-four sentence close with no new claims. | Defensible claims list |

## Missing Or Pending Evidence

The human audit packet exists, but completed human labels are not present in the analysis artifact. The draft will therefore describe the audit as pending rather than as completed validation.

The MAUD `maud_bootstrap_ci_v2.json` artifact is source-compatible with the original MAUD point estimates. A later tie-aware AUROC helper changes tied self-report AUROC values slightly if MAUD is recomputed. This is recorded in `docs/paper_revision_log.md`; the paper preserves the existing source artifacts instead of silently rebaselining MAUD.

## Claim Boundary

The draft should make exactly the bounded claims listed in the user task: self-report does not track claim correctness on MAUD, FActScore, or FELM; residual probes recover signal on MAUD and FActScore; the two-domain replication differs in label source, generation source, and domain; residual and SAE probes are comparable on current evidence; GPT-5.4 is strongest on MAUD but inflated by same-family judge coupling; label provenance matters; CUAD remains an upstream claim-extraction boundary.
