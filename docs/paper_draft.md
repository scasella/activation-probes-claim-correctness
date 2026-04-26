# Activation Probes Recover Claim-Level Correctness Signal Across Legal and Biographical QA

## Abstract

Long-form language-model answers can be wrong one claim at a time, but the model's own stated confidence does not reliably reveal which claims are unreliable. We test whether linear probes on hidden activations recover claim-level correctness signal that structured self-report misses. On MAUD merger-agreement question answering, Llama 3.1 8B self-report is near random under a GPT-5.4 judge (AUROC 0.511), while a layer-19 residual-stream probe reaches 0.771, a paired AUROC delta of 0.260 [0.137, 0.387]. A second Kimi K2.6 judge preserves the method ordering and shows that GPT-5.4 external scoring is inflated by same-family judge coupling: its AUROC is 0.944 under the GPT-5.4 judge and 0.872 under Kimi, making 0.87 the more defensible estimate. On FActScore biographies, using human atomic-fact labels over ChatGPT generations, we find a comparable qualitative gap: self-report reaches 0.541 AUROC and the residual probe reaches 0.802, delta 0.262 [0.219, 0.302]. FELM world-knowledge QA is directionally positive but inconclusive, and its matched-generation repair shows that reference-bounded labels often measure evidence coverage rather than claim correctness. The evidence supports a bounded methods claim: residual activations encode recoverable claim-level correctness signal across legal and biographical QA, but transfer depends on label provenance and stable claim construction.

## 1. Introduction

Long-form language-model outputs fail locally. A generated answer may contain several correct claims, one unsupported claim, and one overconfidently wrong claim. This is especially problematic in settings such as legal and factual question answering, where a user needs to know which parts of an answer are safe to rely on. A single answer-level confidence score is too coarse for this use case. The practical target is claim-level uncertainty: for each factual claim in an answer, estimate whether the model's answer is likely to be correct.

The simplest way to elicit claim-level uncertainty is to ask the model. Prior work has shown that language models can sometimes express calibrated uncertainty in words or probabilities when trained or prompted appropriately. But there is a gap between "the model can produce a confidence statement" and "the confidence statement ranks the correctness of claims in a demanding long-form answer." In the settings studied here, Llama 3.1 8B Instruct's structured self-reported confidence is near random. It does not reliably distinguish correct from incorrect claims on MAUD merger-agreement QA, FActScore biography factuality, or FELM world-knowledge QA.

This paper tests a different signal: the model's hidden activations. We train simple linear probes on layer-19 residual-stream activations pooled over claim spans. The probe is supervised on claim-level correctness labels, then evaluated on held-out claims. The question is specific and bounded. We do not ask whether probes discover "truth" in an absolute sense. We ask whether a linear readout of hidden activations recovers correctness signal that the same model's verbal self-report fails to surface.

The main result is positive in two domains. On MAUD merger-agreement QA, a residual probe reaches 0.771 AUROC under the GPT-5.4 judge, compared with 0.511 for Llama self-report. The paired AUROC delta is 0.260 [0.137, 0.387]. On FActScore ChatGPT biographies, using original human atomic-fact labels, the residual probe reaches 0.802 AUROC, compared with 0.541 for self-report. The paired AUROC delta is 0.262 [0.219, 0.302]. The effect sizes are strikingly similar despite changes in label source, generation source, and content domain. These headline values are reported with source artifacts in Section 4.

The paper also studies boundary conditions. On MAUD, we run a second-judge sensitivity analysis with Kimi K2.6 labels. The qualitative ordering remains stable, but the GPT-5.4 external scorer's AUROC drops from 0.944 under the GPT-5.4 judge to 0.872 under Kimi, showing that same-family scorer-judge coupling inflated the original number. Residual and sparse-autoencoder probes remain above self-report, but residual-vs-SAE differences are not statistically stable. On FELM world-knowledge QA, residual probes are directionally better than self-report, 0.652 versus 0.511 AUROC, but the paired delta interval crosses zero. A matched-generation FELM repair further shows that reference-bounded annotation often yields `not_enough_evidence` rather than factual labels when Llama generates beyond curated snippets.

The resulting contribution is a methods claim with explicit boundaries. Activation probes can recover claim-level correctness signal that self-report misses in MAUD and FActScore. The evidence does not show that the method works for arbitrary domains, models, layers, claim extractors, or label sources. Instead, it suggests a more useful design rule: claim-level uncertainty methods depend as much on label provenance and claim construction as on the probe architecture.

This framing is intentionally different from building another factuality evaluator. An evaluator can be useful even if it is entirely external to the model being evaluated. A probe-based uncertainty method asks a different question: does the model's own hidden computation contain a signal that could have been exposed to the user but was not surfaced in the answer? The practical value is not only better scoring. It is evidence about the gap between internal representation and external self-report, which is why the self-report comparison is load-bearing throughout the paper.

## 2. Related Work

This work sits between verbal uncertainty elicitation, hidden-state truthfulness probes, semantic uncertainty estimation, and mechanistic interpretability.

Verbal self-report is the most direct uncertainty baseline. Lin, Hilton, and Evans show that models can be trained to express uncertainty about answers in words, producing calibrated verbal probabilities under some settings. Kadavath et al. study whether language models can evaluate their own claims and estimate whether they know the answer, finding encouraging self-evaluation behavior in the right formats. Our results are compatible with that literature but less optimistic for the specific setting here: structured self-report from Llama 3.1 8B does not rank claim-level correctness well on long-form factual outputs.

Hidden-state probing for truthfulness is the closer methodological ancestor. Azaria and Mitchell train classifiers on hidden activations to predict whether statements are truthful. Burns et al. search for latent knowledge in activations without direct supervision, and Marks and Tegmark study linear structure in representations of true and false statements. Those works mostly focus on statement-level truthfulness in controlled datasets. Our contribution is narrower but more operational: we evaluate claim-level correctness inside long-form answers, compare directly with model self-report, and test the probe-over-self-report gap across two domains.

Semantic entropy methods attack a related uncertainty problem from the generation side. Farquhar et al. estimate uncertainty over meanings by sampling multiple model generations, and Kossen et al. propose semantic entropy probes that approximate semantic entropy from hidden states of a single generation. Our work shares the idea that hidden states can encode uncertainty-relevant information, but the target differs. We do not predict semantic entropy over possible generations; we predict correctness of fixed extracted claims.

Recent real-time hallucination detection work moves closer to deployment. Obeso et al. train probes for hallucinated entity detection in long-form generation and report scalable real-time detection across model families. Their work is entity- or token-level on long-form generation, while ours is claim-level on fixed extracted factual units. Both results point in the same direction: hidden activations expose error-related signal that is not cheaply available from text or self-report alone.

## 3. Method

### 3.1 Overview

Each experiment converts model outputs into a claim-labeled dataset. For each claim, we compare the model's self-reported confidence with a linear probe score computed from Llama 3.1 8B Instruct hidden activations. The probe uses layer-19 residual-stream vectors pooled over the claim span. Evaluation uses AUROC for ranking correctness and Brier score for probability quality. Confidence intervals use paired bootstrap resampling at the claim level with 1000 resamples for the publication tables.

The experiments differ in their label source and claim construction:

| Dataset | Domain | Generation source | Label source | Unit |
| --- | --- | --- | --- | --- |
| MAUD | Merger-agreement QA | Llama 3.1 8B | LLM judge proxy, plus second-judge sensitivity | Extracted legal claim |
| FActScore | Biographies | ChatGPT | Human FActScore labels | Atomic fact mapped to parent sentence span |
| FELM-wk | World knowledge QA | ChatGPT | Human FELM labels | Response segment |

The model being probed is always Llama 3.1 8B Instruct. FActScore and FELM therefore include a generation-mismatch caveat: activations are extracted from Llama reading ChatGPT-generated text, not from Llama generating its own answer. This makes the transfer test harder in one respect and different in another. It tests whether Llama's representation of text content contains factuality signal, not only whether Llama exposes generation-time uncertainty about its own sampled output.

The probe-training protocols differ because the sample sizes differ. MAUD has only 150 claim-level units, so a conventional train/validation/test split would leave very little data for either fitting or evaluation; the headline MAUD probe table therefore uses leave-one-out evaluation over the frozen full claim set. FActScore has 4886 atomic facts and comfortably supports a 70/15/15 split by biography. These are different protocols, but both are standard choices for their respective sample regimes: cross-validation for a small benchmark and held-out train/validation/test evaluation for a larger one.

### 3.2 MAUD Protocol

MAUD is a merger-agreement question-answering setting. The frozen full run contains 150 claim-level units drawn from 135 examples. Each row includes a contract excerpt, a question, the Llama answer, and a fixed extracted claim. The claim list is not edited during evaluation.

The primary label source is a GPT-5.4 judge-proxy protocol. The judge scores each fixed claim against the provided evidence bundle: contract excerpt, question, answer, and claim. The label schema is `true`, `partially_true`, or `false`; the binary evaluation target treats `true` as positive and all non-true labels as negative. The central MAUD artifacts are `data/annotations/maud_full_judge_annotations.jsonl`, `artifacts/runs/maud_full_proxy_baseline_eval.json`, and `artifacts/runs/maud_full_probe_proxy_smoke.json`.

To test judge-family sensitivity, the same frozen claims are relabeled by Kimi K2.6 through Prime Intellect Inference. The prompt structure and claim serialization are preserved from the GPT-5.4 judge. The second-judge artifacts are `artifacts/runs/maud_full_proxy_baseline_eval_v2.json`, `artifacts/runs/maud_full_probe_proxy_smoke_v2.json`, and `artifacts/runs/maud_judge_agreement_analysis.json`. The probes are not retrained on Kimi labels. They are fixed from judge-1 labels and rescored against judge-2 labels.

The MAUD residual and SAE probe quick-check uses leave-one-out binary classification over the 150 claims with standardized logistic regression and fixed `C=1.0`, matching `scripts/evaluate_probe_proxy_smoke.py`. The residual probe uses raw layer-19 residual activations from `artifacts/runs/maud_full_probe_features_residual.jsonl`. The SAE probe uses Goodfire layer-19 SAE features from `artifacts/runs/maud_full_probe_features_sae.npz`. A richer internal dry-run artifact also evaluates train/validation/test probe variants, but the headline MAUD comparison in this paper uses the full 150-claim source-compatible probe-smoke artifacts.

### 3.3 FActScore Adaptation

FActScore provides human-labeled atomic facts for long-form biographies. We use the ChatGPT portion of the human-annotated release, because it is comparable in example count to FELM-wk and provides a clean non-legal transfer setting. The adapter maps `Supported` to true, `Not-supported` to false, and drops `Irrelevant`, because irrelevance is not a factuality label. The resulting dataset contains 157 usable biographies and 4886 non-IR atomic facts, with 3194 true and 1692 false labels (`artifacts/runs/factscore_chatgpt_adapter_summary.json`).

FActScore introduces a span-alignment issue. Atomic facts are human-written canonical statements and almost never appear verbatim in the generated biography. Exact substring matching succeeds for only 8 of 4886 facts, an exact-match rate of 0.15 percent. To avoid fabricating token spans, the adapter pools activations over the parent annotated sentence span while retaining the atomic-fact label. This is a compromise: the representation is sentence-level, while the label is atomic-fact-level. It is still preferable to unreliable string matching, and the token alignment check is clean. All 4886 claims align to token spans; the median span length is 27 tokens, with no 1-2 token spans (`artifacts/runs/factscore_chatgpt_token_alignment_summary.json`).

FActScore uses a 70/15/15 split by biography, not by atomic fact, to prevent facts from the same biography from leaking across train and test. Split sizes are 3353 train claims, 791 validation claims, and 742 test claims. The residual probe is an L2 logistic regression over standardized layer-19 residual features. We sweep `C` over 0.01, 0.1, 1.0, and 10.0 on validation; `C=0.01` is selected. The final probe is trained on train plus validation and evaluated once on the held-out test split (`artifacts/runs/factscore_chatgpt_validation_eval.json`).

### 3.4 FELM Adaptation

FELM world-knowledge QA is used as a boundary-case transfer check. We treat each human-annotated response segment as a claim. The labels are human factuality annotations from FELM, and the source generations are ChatGPT responses. As in FActScore, activations are extracted from Llama reading ChatGPT-generated text.

The minimum-viable FELM run materializes 519 segment features and 518 self-report predictions, with one self-report failure explicitly tracked (`artifacts/runs/felm_wk_validation_eval.json`). It uses a 70/15/15 split by example and L2 logistic regression with validation-selected `C=0.1`. The held-out test split has 71 claims in the artifact. This is smaller and noisier than FActScore, and the paper treats it as a boundary rather than a replication.

We also attempted a matched-generation FELM annotation repair. Llama-generated answers often went beyond FELM's curated reference snippets. Adding fuller readability-extracted references and a fourth `not_enough_evidence` label improved the annotation target, but the repaired pilot still produced 30 `not_enough_evidence` labels out of 52 segments, or 57.7 percent (`docs/felm_matched_generation_repair.md`). We therefore do not scale the matched-generation FELM run. The substantive finding is that reference-bounded labels can measure evidence coverage rather than claim correctness when generated claims exceed the reference scope. In that setting, a false-looking label may mean "unsupported by this reference bundle," not "factually wrong."

### 3.5 Self-Report Baseline

The self-report baseline asks Llama 3.1 8B for structured confidence over fixed claims or segments. The model is shown the unit being scored, so the baseline tests whether Llama can map a presented claim to a useful confidence score, not whether it can extract claims or infer claim boundaries. If the baseline fails here, the failure cannot be blamed on implicit claim boundaries or on unstructured confidence language. The resulting probability-like confidence is used directly for AUROC and Brier evaluation, with no threshold tuning for the headline ranking results.

### 3.6 Evaluation And Bootstrap

AUROC measures whether a score ranks true claims above not-true claims. Brier score measures squared error of probability estimates against the binary label. For paired deltas, claims are resampled once per bootstrap iteration and every method is evaluated on the same resampled claim indices. This paired design is important because the methods are evaluated on the same claims.

The MAUD paper tables use `artifacts/runs/maud_bootstrap_ci_v2.json`, a source-compatible 1000-resample CI artifact preserving the frozen MAUD point estimates. FActScore uses `artifacts/runs/factscore_bootstrap_ci.json`, also with 1000 paired resamples. FELM uses `artifacts/runs/felm_wk_bootstrap_ci.json` for the boundary-case table, preserving the original FELM point estimates. Degenerate AUROC resamples are tracked; none were dropped in the headline FActScore or FELM CIs. The MAUD kappa interval is also computed by paired claim-level bootstrap.

## 4. Results

### 4.1 MAUD Core Result

Table 1 reports the MAUD method comparison under the original GPT-5.4 judge, the independent Kimi K2.6 judge, and the 110-claim subset where both judges agree. The central MAUD finding is that Llama self-report is near random, while activation probes recover claim-level correctness signal. GPT-5.4 external scoring is strongest, but its same-family judge coupling must be corrected when interpreting the headline number.

**Table 1: MAUD Methods Comparison**

| Method | GPT-5.4 judge AUROC | GPT-5.4 judge Brier | Kimi K2.6 judge AUROC | Kimi K2.6 judge Brier | Agreement set AUROC | Agreement set Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama self-report | 0.511 [0.411, 0.597] | 0.580 [0.500, 0.653] | 0.466 [0.413, 0.604] | 0.480 [0.407, 0.560] | 0.486 [0.392, 0.598] | 0.509 [0.400, 0.600] |
| GPT-5.4 external scorer | 0.944 [0.906, 0.984] | 0.161 [0.118, 0.206] | 0.872 [0.807, 0.926] | 0.169 [0.124, 0.217] | 0.981 [0.931, 1.000] | 0.101 [0.064, 0.144] |
| Residual probe | 0.771 [0.687, 0.838] | 0.244 [0.189, 0.308] | 0.707 [0.627, 0.792] | 0.290 [0.227, 0.350] | 0.793 [0.709, 0.870] | 0.231 [0.166, 0.299] |
| SAE probe | 0.677 [0.582, 0.763] | 0.300 [0.229, 0.372] | 0.652 [0.563, 0.735] | 0.326 [0.259, 0.395] | 0.712 [0.616, 0.812] | 0.268 [0.193, 0.350] |

Source artifacts: `artifacts/runs/maud_full_proxy_baseline_eval.json`, `artifacts/runs/maud_full_probe_proxy_smoke.json`, `artifacts/runs/maud_full_proxy_baseline_eval_v2.json`, `artifacts/runs/maud_full_probe_proxy_smoke_v2.json`, `artifacts/runs/maud_agreement_set_eval.json`, and `artifacts/runs/maud_bootstrap_ci_v2.json`.

Under the GPT-5.4 judge, the residual probe reaches 0.771 AUROC versus 0.511 for Llama self-report. The paired residual-minus-self-report delta is 0.260 [0.137, 0.387] (`artifacts/runs/maud_bootstrap_ci_v2.json`). This is the core MAUD probe result: the model's hidden activations contain correctness signal that its structured self-report does not surface. The uncertainty interval is wider than FActScore's because MAUD is much smaller. The source-compatible headline table uses leave-one-out evaluation over 150 claims, while the richer held-out dry-run has only 22 test claims (`artifacts/runs/maud_full_proxy_dry_run.json`); both views make clear that MAUD is statistically underpowered relative to FActScore. The qualitative result is stable, but the quantitative point estimate should be read with that sample-size constraint in mind.

The GPT-5.4 external scorer is the strongest MAUD method, but the raw 0.944 AUROC under GPT-5.4 labels is not the fairest estimate. Because GPT-5.4 also produced the original judge labels, that number includes same-family scorer-judge coupling. Under the independent Kimi K2.6 judge, the same GPT-5.4 scorer reaches 0.872 AUROC. The corrected reading is therefore that GPT-5.4 scoring is very strong, but its defensible MAUD estimate is closer to 0.87 than 0.94.

The residual-vs-SAE comparison is directionally residual-favored but not a stable finding. Residual minus SAE is 0.094 [0.003, 0.180] under judge 1, 0.055 [-0.039, 0.150] under judge 2, and 0.081 [-0.021, 0.185] on the agreement set (`artifacts/runs/maud_bootstrap_ci_v2.json`). Since two of the three intervals cross zero, the paper should not claim that residual probes reliably beat SAE probes. The supported claim is narrower: raw residual features perform at least as well as the Goodfire SAE features evaluated here, and both carry more signal than self-report.

### 4.2 Second-Judge Sensitivity

The Kimi K2.6 sensitivity pass asks whether the MAUD ordering survives a judge-family change. It does. Under Kimi labels, GPT-5.4 external scoring remains strongest at 0.872 AUROC, residual probes remain next at 0.707, SAE probes remain above self-report at 0.652, and Llama self-report remains near chance at 0.466 (`artifacts/runs/maud_full_proxy_baseline_eval_v2.json`, `artifacts/runs/maud_full_probe_proxy_smoke_v2.json`). The absolute numbers move, but the ranking does not reverse.

The cleanest evidence of same-family inflation is the GPT-5.4 scorer's drop from 0.944 under GPT-5.4 labels to 0.872 under Kimi labels. The paired judge-1-minus-judge-2 delta is 0.080 [0.019, 0.149] (`artifacts/runs/maud_bootstrap_ci_v2.json`). This does not make the GPT-5.4 scorer weak; it remains the top method. It does mean that the original 0.944 number should be treated as a same-family upper estimate, not as an independent measurement of external-scoring performance.

The two judges agree on 110 of 150 claims. Raw agreement is 0.733 and Cohen's kappa is 0.572 [0.456, 0.682] (`artifacts/runs/maud_judge_agreement_analysis.json`, `artifacts/runs/maud_bootstrap_ci_v2.json`). That is moderate agreement: stronger than single-judge self-consistency, but not enough to turn either judge into ground truth. The labels are most stable on clearly true and clearly false claims and least stable on partially true claims, which is exactly where the rubric boundary should be hardest.

### 4.3 The Agreement-Set Result

The judge-agreement set is not merely a robustness check; it is the most interpretable MAUD subset. On the 110 claims where GPT-5.4 and Kimi assign the same correctness label, all methods improve relative to at least one full-set condition. GPT-5.4 external scoring reaches 0.981 AUROC, the residual probe reaches 0.793, the SAE probe reaches 0.712, and Llama self-report reaches 0.486 (`artifacts/runs/maud_agreement_set_eval.json`, `artifacts/runs/maud_bootstrap_ci_v2.json`). Self-report remains near random, but even it is less badly calibrated on the cleaner subset, with Brier 0.509 rather than 0.580 under judge 1.

The important pattern is that the stronger methods improve on labels both judges accept. If a probe were primarily fitting idiosyncrasies of GPT-5.4's judging style, restricting evaluation to GPT-5.4/Kimi agreement would not be expected to help. The improvement instead suggests that the methods track signal that survives judge disagreement, while noisy or ambiguous labels suppress performance on the full set. This is also why the agreement-set result is central to the paper's interpretation: it separates the claim that probes fit one judge from the claim that probes track a more stable correctness-related target.

The agreement set also bounds the current ceiling. GPT-5.4 scoring approaches saturation at 0.981 AUROC, while residual probes improve but remain far below the external scorer. This suggests that both label noise and modeling limitations matter. Cleaner labels help, but they do not make a simple residual probe match a strong external evaluator.

### 4.4 Human Audit Status

The planned MAUD human audit is not completed in the current artifact set. The audit packet is frozen at `data/annotations/maud_human_audit_packet.jsonl`, but `artifacts/runs/maud_human_audit_analysis.json` does not contain completed valid human labels. This draft therefore treats human validation as forthcoming rather than as evidence already in hand.

The missing audit does not block the paper's LLM-judge-proxy claim, but it limits the legal-validity claim. Stable ordering across GPT-5.4 and Kimi is stronger than single-judge self-consistency. It is not the same as expert legal adjudication. A completed audit could show that humans mostly endorse the two-judge agreement set, that humans systematically prefer one judge on disagreements, or that both LLM judges miss legally important distinctions. Until that audit is analyzed, MAUD should be described as judge-proxy evidence.

### 4.5 FActScore Replication

FActScore provides the strongest non-MAUD evidence. It differs from MAUD in label source, generation source, domain, and unit of analysis: FActScore uses human labels rather than LLM judges, ChatGPT biographies rather than Llama legal answers, biographical factuality rather than merger-agreement QA, and canonical atomic facts pooled to parent sentence spans rather than extracted legal claims. Despite those differences, the qualitative probe-over-self-report gap is comparable to MAUD.

| Method | Test claims | AUROC | Brier |
| --- | ---: | ---: | ---: |
| Llama self-report | 742 | 0.541 [0.514, 0.570] | 0.298 [0.262, 0.330] |
| Residual probe | 742 | 0.802 [0.769, 0.833] | 0.171 [0.156, 0.189] |

Source artifacts: `artifacts/runs/factscore_chatgpt_validation_eval.json` and `artifacts/runs/factscore_bootstrap_ci.json`.

The headline FActScore result is residual probe AUROC 0.802 versus self-report AUROC 0.541, with paired residual-minus-self-report delta 0.262 [0.219, 0.302]. The Brier scores point the same way: 0.171 for the residual probe versus 0.298 for self-report. The tighter confidence interval compared with MAUD is expected because the FActScore test split contains 742 claims rather than a small legal-claim set.

The probe diagnostics are consistent with a real held-out effect rather than a fragile overfit. The residual probe reaches 0.939 AUROC on train, 0.728 on validation, and 0.802 on test (`artifacts/runs/factscore_chatgpt_validation_eval.json`). Test AUROC above validation is not surprising given biography-level splits, where individual biographies vary substantially in difficulty and fact density. The pattern is consistent with the `C=0.01` model generalizing across that variation rather than overfitting to validation.

The comparison to MAUD is the key point. The two experiments are not methodologically identical: MAUD uses leave-one-out over a small judge-labeled legal claim set, while FActScore uses a 70/15/15 train/validation/test split over a much larger human-labeled biography set. What transfers is not the exact protocol, but the qualitative gap between self-report and residual activations with nearly identical paired effect size: 0.260 on MAUD under judge 1 and 0.262 on FActScore. That is the replication: hidden activations recover claim-level correctness signal that structured self-report misses across two substantially different evidence sources.

### 4.6 FELM Boundary Case

FELM-wk is directionally consistent with MAUD and FActScore, but it is not a clean replication. On the held-out test split, self-report reaches 0.511 AUROC and the residual probe reaches 0.652. The original scoping bootstrap delta is 0.141 [-0.036, 0.314], and the source-compatible 1000-resample helper interval is 0.141 [-0.063, 0.320] (`artifacts/runs/felm_wk_validation_eval.json`, `artifacts/runs/felm_wk_bootstrap_ci.json`). The interval includes zero, so FELM should be described as positive in direction but statistically inconclusive.

The matched-generation FELM repair clarifies why FELM is a boundary case rather than a decisive negative. When Llama-generated answers were judged against fuller but still reference-bounded FELM evidence, the repaired pilot produced a 57.7 percent `not_enough_evidence` rate: 30 of 52 segments could not be judged from the evidence bundle (`docs/felm_matched_generation_repair.md`). This shows that the annotation target had drifted from claim correctness to reference coverage. In that setting, the label tells us whether the reference bundle supports the claim, not necessarily whether the claim is true or false.

**Table 2: Cross-Dataset Probe Versus Self-Report Comparison**

| Dataset | Label source | Unit | Self-report AUROC | Residual probe AUROC | Residual - self-report AUROC |
| --- | --- | --- | ---: | ---: | ---: |
| MAUD | GPT-5.4 judge proxy | Legal claim | 0.511 [0.411, 0.597] | 0.771 [0.687, 0.838] | 0.260 [0.137, 0.387] |
| FActScore | Human factuality labels | Biography atomic fact | 0.541 [0.514, 0.570] | 0.802 [0.769, 0.833] | 0.262 [0.219, 0.302] |
| FELM-wk | Human segment labels | World-knowledge segment | 0.511 [0.393, 0.673] | 0.652 [0.493, 0.815] | 0.141 [-0.063, 0.320] |

Source artifacts: `artifacts/runs/maud_bootstrap_ci_v2.json`, `artifacts/runs/factscore_bootstrap_ci.json`, `artifacts/runs/felm_wk_validation_eval.json`, and `artifacts/runs/felm_wk_bootstrap_ci.json`.

The cross-dataset comparison is therefore coherent but bounded. MAUD and FActScore show comparable, statistically supported probe-over-self-report gaps. FELM points the same way, but its labels and evidence structure make the result weaker. The wide FELM bootstrap interval reflects both the smaller test set and noisier label structure. A more definitive FELM result would require either substantially more data or annotation methodology that addresses the reference-coverage problem identified in the matched-generation repair. The lesson is not that the method fails on world knowledge; it is that reference-bounded segment labels are a poorer target for claim-level correctness than fixed legal claims or human atomic facts.

**Table 3: FActScore Probe Diagnostics**

| Split or method | Claims | AUROC | Brier |
| --- | ---: | ---: | ---: |
| Residual probe train | 3353 | 0.939 | 0.097 |
| Residual probe validation | 791 | 0.728 | 0.220 |
| Residual probe test | 742 | 0.802 [0.769, 0.833] | 0.171 [0.156, 0.189] |
| Llama self-report test | 742 | 0.541 [0.514, 0.570] | 0.298 [0.262, 0.330] |

Source artifacts: `artifacts/runs/factscore_chatgpt_validation_eval.json`, `artifacts/runs/factscore_bootstrap_ci.json`, `artifacts/runs/factscore_chatgpt_adapter_summary.json`, and `artifacts/runs/factscore_chatgpt_token_alignment_summary.json`.

## 5. Discussion

The central interpretive claim is that probe-based uncertainty works when labels capture claim correctness directly and becomes less reliable when labels capture an adjacent property such as reference support. MAUD uses judge-proxy labels over fixed legal claims; FActScore uses human labels over atomic biography facts. Both produce a clear residual-probe-over-self-report gap. FELM uses human segment labels and reference-bounded evidence, but the matched-generation repair shows that its labels can collapse into coverage judgments when generated claims exceed the reference scope. Label provenance matters more than domain similarity. Applying probe-based methods to new domains is therefore not primarily a domain-transfer problem; it is a label-construction problem.

The agreement-set result strengthens that interpretation. All methods improve on the subset where GPT-5.4 and Kimi agree, which suggests the methods track signal that survives judge disagreement. Probes are unlikely to be primarily fitting one judge's private artifacts if their scores improve on labels accepted by a different judge family. Combined with the Kimi second-judge stability and the FActScore result under human labels, this cross-judge and cross-source robustness is the strongest evidence the paper offers that residual probes track something more general than one labeler's quirks.

FActScore also changes how to read generation mismatch. FActScore biographies were generated by ChatGPT, not Llama, yet the residual probe over Llama activations produces a clean effect comparable to MAUD's Llama-generated legal answers. This weakens the hypothesis that generation mismatch is the dominant explanation for FELM's weaker result. The matched-generation FELM repair points instead to annotation-target mismatch: reference-bounded judging was unable to adjudicate many Llama-generated claims, producing `not_enough_evidence` rather than correctness labels.

The practical lesson is that a probe-based uncertainty system needs to treat claim construction, label provenance, and label-claim alignment as first-class design choices. What is the claim? What exactly does the label capture? Does the label target match the unit being pooled in the activation stream? FActScore works because its human atomic-fact labels directly capture factual support, even though the representation must pool to the parent sentence. MAUD works because the judges apply a clear rubric to a fixed claim list. FELM is weaker because segment units and reference coverage are coarser. Claim construction and label provenance affect probe trainability as much as model architecture choices do.

The accompanying ProbeMon v0.1 library packages this bounded design into two pretrained probes, dataset adapters, and user-training infrastructure while keeping those claim-construction and label-provenance assumptions explicit.

The narrower finding also fits a broader pattern in activation-probing research. Azaria and Mitchell, Burns et al., Marks and Tegmark, and Kossen et al. all document cases where hidden states contain truthfulness, knowledge, or uncertainty-related signal not directly exposed in model text. Our contribution is the operational extension to claim-level scoring inside long-form outputs, with an explicit self-report baseline and transfer evidence across two domains and label sources. The paper does not claim that probes reveal truth in general. It shows that, in these settings, Llama's hidden activations contain a correctness signal that Llama's own structured confidence does not faithfully report.

## 6. Threats To Validity

**Single base model.** All experiments use Llama 3.1 8B Instruct as the probed model. There is no evidence here that the result holds at other scales or in other model families. The probe-over-self-report gap may reflect this model's specific representation geometry, post-training, or self-report behavior.

**Small MAUD sample.** MAUD has 150 claim-level units, and the richer held-out dry-run has only 22 test claims. The source-compatible MAUD LOO table supports the qualitative direction, but its bootstrap intervals are wider than FActScore's and the point estimates should not be overread. FActScore strengthens confidence in the qualitative finding precisely because it supplies the statistical power MAUD lacks.

**Judge-mediated MAUD labels.** MAUD labels remain LLM-generated even after the Kimi second-judge sensitivity pass. The second judge addresses same-family self-consistency, but it does not establish expert legal truth. The frozen human audit packet is a start, but completed human labels are not present in the current analysis artifact. The fundamental constraint is that probes trained on judge labels learn signal correlated with what judges call correctness, which is one step removed from what competent legal reasoning would conclude.

**FActScore canonicalization compromise.** FActScore atomic-fact labels are pooled over parent sentence spans because exact substring matching succeeds for only 0.15 percent of facts (`artifacts/runs/factscore_chatgpt_adapter_summary.json`). The probe reads sentence-level activations even though labels are atomic-fact-level. The result may partly reflect sentence-level signal aligned with atomic-fact correctness rather than atomic-fact-specific signal.

**Generation mismatch.** FActScore and FELM contain ChatGPT-generated text, while the activations come from Llama reading that text. Section 5 argues that this is plausibly less load-bearing than initially feared, because FActScore produces a clean effect despite the mismatch. But matched-generation experiments would address it more directly. The matched-generation FELM attempt failed for annotation-target reasons, leaving the generation-source question unresolved.

**Single layer.** All probes operate at layer 19. No layer sweep was performed. Layer 19 was selected because the Goodfire pretrained SAE used for the SAE comparison is trained at this layer. Other layers might produce stronger, weaker, or more interpretable correctness signals.

**Distributed signal.** A separate interpretability analysis found that the MAUD residual correctness-probe direction does not decompose into a compact set of Goodfire layer-19 SAE features (`docs/probe_interpretation.md`). The probe works as a predictive readout, but the current evidence does not support a clean feature-level explanation. A useful behavioral readout is not automatically a compact mechanism. Feature-steering interventions to improve self-report are therefore unlikely to be straightforward at this configuration.

**CUAD boundary.** A separate attempt to apply the methodology to CUAD found claim-extraction instability that prevented reliable evaluation. The methodology depends on stable upstream claim construction. Benchmarks that do not naturally support fixed, adjudicable claim units require a different preprocessing pipeline before this probe protocol can be meaningfully evaluated.

## 7. Conclusion

Linear probes on residual activations recover claim-level correctness signal that Llama 3.1 8B's structured self-report misses. The gap is comparable across MAUD legal QA, where labels are judge-proxy legal-claim judgments, and FActScore biographies, where labels are human atomic-fact annotations over ChatGPT generations. FELM is a documented boundary: its labels point in the same direction but become less usable when reference-bounded evidence measures coverage rather than correctness.

The methodological lesson is that probe-based uncertainty depends on label provenance and claim construction as much as probe architecture. Domains with labels that directly adjudicate stable claim units are plausible candidates. Domains where labels capture adjacent properties, such as reference support for broad segments, require substantially more care.

Several questions remain open. Matched-generation transfer is unresolved, transfer to other model families is untested, and feature-level decomposition did not yield a compact mechanistic explanation of the residual probe direction. Those results point downstream work toward human-grounded validation, broader model coverage, and subspace- or behavior-level interventions rather than simple individual-feature steering. Downstream work building on these results should expose label-provenance and claim-construction assumptions to users rather than hiding them behind opaque scores.
