# Activation Probes Recover Claim-Level Correctness Signal in Legal QA

Preregistered abstract date: 2026-04-24

Status: submission-candidate draft, blocked on completed human-audit labels.

## Abstract

Llama 3.1 8B Instruct can answer merger-agreement questions with high stated confidence even when individual factual claims in its answer are wrong. We study 150 frozen claim-level units from MAUD merger-agreement question answering and compare four uncertainty signals: Llama's structured self-reported confidence, a GPT-5.4 external scorer, linear probes on residual activations, and linear probes on Goodfire sparse-autoencoder features. Correctness is measured with a judge-proxy protocol using GPT-5.4 labels plus an independent Kimi K2.6 sensitivity pass; a small human audit is used only as a validation anchor, not for probe training. Self-report confidence is near random at ranking claim correctness under both LLM judges. Residual activation probes recover meaningful correctness signal that self-report misses, while SAE features are not reliably distinguishable from residual features on current evidence. GPT-5.4 external scoring remains strongest, but its same-family headline AUROC is inflated; the Kimi-scored estimate is the more defensible comparison. The results support a bounded methods claim: hidden activations encode claim-level correctness signal in this MAUD setting, but the finding remains single-domain, single-model, and dependent on proxy labels plus a small audit.

## Introduction

Language models are increasingly used for legal question answering, where the relevant unit of failure is often smaller than the answer. A response can be directionally useful while still containing a material overstatement, missing condition, or unsupported carveout. This is a bad fit for answer-level confidence. A model may confidently produce an answer that sounds legally fluent while one extracted claim inside the answer is wrong.

This paper studies that claim-level setting. The object of evaluation is not whether an entire Llama response is good, but whether specific factual claims extracted from that response are correct with respect to the contract excerpt and question. The domain is MAUD merger-agreement question answering. The base model is Llama 3.1 8B Instruct. The claim list is frozen at 150 claims from 135 examples.

The central empirical question is simple: when Llama states confidence about its own answer, does that confidence rank correct claims above incorrect claims? And if not, do the model's hidden activations carry a signal that its verbal self-report misses?

The answer is yes to the second question and no to the first. Llama self-report is near random under both LLM judges. A linear probe on residual activations reaches 0.771 AUROC under the GPT-5.4 judge and 0.707 under the Kimi K2.6 judge. A sparse-autoencoder feature probe is also above chance, but not reliably better than residual features. GPT-5.4 external scoring is strongest, but the original 0.944 AUROC is partially inflated by same-family coupling with the GPT-5.4 judge labels; under Kimi labels, its AUROC is 0.872.

The contribution is methodological, not a claim of legal adjudication. We show that claim-level correctness signal is present in Llama hidden states in a narrow legal QA setting, and that the result survives a judge-family change. We do not claim that probes detect legal truth in any absolute sense. A 30-claim human audit packet is frozen for validation, but completed human labels are still pending at the time of this draft.

This distinction matters for deployment. A legal assistant does not merely need to know whether its answer sounds plausible. It needs to know where the answer might be overclaiming relative to a source document. The experiment therefore uses a stricter unit than answer-level success. It asks whether each extracted claim is fully supported, partially supported, or unsupported. This makes the evaluation harsher, but it also makes the output more useful: a downstream reviewer can inspect a concrete claim and a concrete excerpt rather than a global confidence number.

## Related Work

Semantic entropy probes are the closest methodological ancestor. Kossen et al. propose probing hidden states to approximate semantic entropy from a single generation rather than repeatedly sampling model outputs. Their motivation is computational: semantic entropy can detect hallucination-like uncertainty, but repeated sampling is expensive. Our work shares the idea that hidden states can carry uncertainty signal, but changes the target from answer-level semantic uncertainty to claim-level correctness in legal QA.

The broader SAPLMA-style line of work studies whether language-model activations can predict statement accuracy. Our setting is a constrained version of that question: statements are fixed extracted legal claims, correctness is judged against a contract excerpt, and the comparison includes both verbal self-report and activation probes. This makes the measurement more brittle but also more interpretable, because each row has a visible claim and evidence context.

Real-time hallucination detection work, including Obeso et al. (arXiv:2509.03531), trains probes or classifiers for token- or entity-level hallucination detection in long-form generation. That work is operationally motivated: detection should happen during generation and avoid costly external verification. Our task is offline and claim-level. We care less about streaming intervention and more about whether an answer's internal representation contains information about later claim correctness.

Verbal uncertainty work, including work by Ferrando and others on model uncertainty expressions and hallucination probes, motivates the self-report comparison. If a model's stated confidence were reliable, it would be the simplest usable uncertainty signal. In this setting, it is not. Llama's structured confidence is near random at ranking correctness, even though its Brier score shifts with judge label distribution.

Anthropic's feature work, including "Mapping the Mind of a Large Language Model" and "On the Biology of a Large Language Model," motivates the SAE comparison. Sparse features can make internal representations more interpretable and sometimes causally meaningful. Our result is deliberately narrower: Goodfire SAE features at one layer do carry signal, but they do not clearly improve over raw residual activations for this MAUD correctness task.

## Method

### Frozen Claim Protocol

The corpus is a frozen MAUD-only set of 135 merger-agreement QA examples with 150 extracted claims. Each row contains a contract excerpt, a question, Llama's answer, and one claim extracted from that answer. The correctness label is three-way: `true`, `partially_true`, or `false`. For AUROC, `true` is the positive class and both `partially_true` and `false` are treated as not-true. This is intentionally strict: a partially true legal claim may still omit a material condition.

The frozen-claim design is load-bearing. If claims are regenerated after labels or activations are computed, the experiment can silently change the target. If answers are regenerated during feature extraction, activation spans can become attached to different text. The pipeline therefore preserves the original Llama answers, extracted claims, and token spans. This is less flexible than a fully dynamic benchmark, but it prevents accidental leakage between generation, labeling, and probing.

CUAD was not included in the final experiment. Under the same prompt-and-extraction setup, CUAD did not produce stable claim objects in the readiness screen. This is a boundary result, not merely setup failure: the claim-level protocol requires stable extracted factual units, and CUAD as prompted did not satisfy that requirement.

### Judge-Proxy Labels

The first label surface is GPT-5.4. The judge sees the contract excerpt, question, Llama answer, and fixed claim list. It assigns correctness and load-bearing labels using a fixed JSON schema. The second label surface is Kimi K2.6 through Prime Intellect Inference, using the identical prompt and schema. The only intended change is judge family.

The two judges label all 150 claims. They agree on 110 claims, for raw agreement 0.733 and Cohen's kappa 0.572. Agreement is highest on claims GPT-5.4 calls true and lower on partially true claims, which is consistent with partially true being the most ambiguous class.

The confusion matrix supports that interpretation. GPT-5.4 and Kimi agree on 54 of 62 GPT-5.4-true claims and 15 of 20 GPT-5.4-false claims, but only 41 of 68 GPT-5.4-partially-true claims. Many disagreements move between `partially_true` and `true`, exactly where legal qualifiers and omitted conditions matter most. This does not invalidate the proxy labels, but it explains why the agreement set is a cleaner target.

### Methods Compared

We compare four scoring methods.

Llama self-report is a structured confidence score produced by Llama about its own claims. It is the cheapest and most direct uncertainty signal.

GPT-5.4 external scoring is a separate scorer that assigns confidence to each claim. It is a strong baseline, but same-family coupling matters because the first judge labels also come from GPT-5.4.

The residual probe is a linear probe trained on Llama hidden-state residual features around the answer tokens. The probe path uses frozen answer token spans so that features remain attached to the original answer text rather than regenerated text.

The SAE probe uses Goodfire sparse-autoencoder features at layer 19. It tests whether a more interpretable feature representation improves over raw residual vectors.

The probes are not retrained on Kimi labels or audit labels. For second-judge sensitivity, the probe scores are fixed from the first study and scored against Kimi labels. This avoids leaking judge 2 into the probe.

The bootstrap analysis is paired at the claim level. Every resample draws claims with replacement and evaluates all methods on the same drawn claims. This is important because the method comparisons are not independent: a hard or ambiguous claim affects every method. Paired deltas estimate whether one method ranks the same claims better than another, rather than mixing method variance with sample composition variance.

Confidence intervals use 1000 paired claim-level bootstrap resamples with fixed seed 20260424 and percentile 95% intervals. AUROC-undefined resamples, which occur only if a bootstrap draw contains a single class, are skipped and counted; none occurred in the headline run. Agreement-set intervals resample within the 110-claim agreement set rather than from the full 150 claims.

### Human Audit

A blinded 30-claim audit packet has been frozen at `data/annotations/maud_human_audit_packet.jsonl`. It contains 20 claims where the two LLM judges agree and 10 where they disagree. The packet excludes LLM labels and sampling strata. Annotators see only the excerpt, question, Llama answer, extracted claim, and rubric.

Completed audit labels are pending. The audit is validation only. It will not be used for probe training, prompt selection, or claim-set revision. Once labels arrive, the planned analysis computes GPT-5.4-vs-human kappa, Kimi-vs-human kappa, agreement by stratum, and the raw count of whether humans side more often with GPT-5.4 or Kimi on judge-disagreement claims.

## Results

### Central AUROC Results

All intervals are 95% paired bootstrap confidence intervals over claims, using 1000 resamples.

| Method | GPT-5.4 judge | Kimi K2.6 judge | Judge-agreement set |
| --- | ---: | ---: | ---: |
| Llama self-report | 0.511 [0.411, 0.597] | 0.466 [0.413, 0.604] | 0.486 [0.392, 0.598] |
| GPT-5.4 external scorer | 0.944 [0.906, 0.984] | 0.872 [0.807, 0.926] | 0.981 [0.931, 1.000] |
| Residual activation probe | 0.771 [0.687, 0.838] | 0.707 [0.627, 0.792] | 0.793 [0.709, 0.870] |
| SAE feature probe | 0.677 [0.582, 0.763] | 0.652 [0.563, 0.735] | 0.712 [0.616, 0.812] |

The ordering is stable: GPT-5.4 external scoring is strongest, residual probes are next, SAE probes are above self-report, and self-report is near random. The important correction is that GPT-5.4's original 0.944 should not be treated as an independent estimate. Under Kimi, the same method scores 0.872. That is still strong, but the defensible headline is closer to 0.87 than 0.94.

### Paired Deltas

| Context | Delta | AUROC delta | 95% CI | Excludes zero? |
| --- | --- | ---: | ---: | --- |
| GPT-5.4 judge | GPT-5.4 - residual | 0.173 | [0.107, 0.264] | yes |
| GPT-5.4 judge | residual - SAE | 0.094 | [0.003, 0.180] | yes, narrowly |
| GPT-5.4 judge | residual - self-report | 0.260 | [0.137, 0.387] | yes |
| Kimi judge | GPT-5.4 - residual | 0.165 | [0.070, 0.251] | yes |
| Kimi judge | residual - SAE | 0.055 | [-0.039, 0.150] | no |
| Kimi judge | residual - self-report | 0.241 | [0.075, 0.320] | yes |
| Agreement set | GPT-5.4 - residual | 0.188 | [0.093, 0.268] | yes |
| Agreement set | residual - SAE | 0.081 | [-0.021, 0.185] | no |
| Agreement set | residual - self-report | 0.307 | [0.153, 0.428] | yes |

The residual-vs-SAE comparison is not reliably distinguishable on current evidence. It narrowly excludes zero under the GPT-5.4 judge, but does not exclude zero under Kimi or on the agreement set. The safe claim is that SAE does not clearly improve over raw residual features here.

The GPT-5.4 scorer's same-family inflation is also statistically visible: its AUROC is 0.080 higher under the GPT-5.4 judge than under Kimi, with paired 95% CI [0.019, 0.149].

### What the Agreement Set Tells Us

The agreement set is the most informative subset. All strong methods improve when both judges agree on the label: GPT-5.4 rises to 0.981, residual to 0.793, and SAE to 0.712. Llama self-report remains near random at 0.486.

This pattern argues against a pure judge-artifact explanation. If the residual probe only learned quirks of GPT-5.4 labels, performance should degrade when restricted to a two-judge high-confidence subset. Instead, it improves. The more plausible interpretation is that judge disagreement is a major source of label noise, and that stronger methods separate claims better when the label surface is cleaner.

This also bounds future progress. With current labels, some residual error is methodological, but some is label noise. Improving probes without improving labels may yield diminishing returns.

### Brier Sensitivity

| Method | GPT-5.4 judge | Kimi K2.6 judge | Judge-agreement set |
| --- | ---: | ---: | ---: |
| Llama self-report | 0.580 [0.500, 0.653] | 0.480 [0.407, 0.560] | 0.509 [0.400, 0.600] |
| GPT-5.4 external scorer | 0.161 [0.118, 0.206] | 0.169 [0.124, 0.217] | 0.101 [0.064, 0.144] |
| Residual activation probe | 0.244 [0.189, 0.308] | 0.290 [0.227, 0.350] | 0.231 [0.166, 0.299] |
| SAE feature probe | 0.300 [0.229, 0.372] | 0.326 [0.259, 0.395] | 0.268 [0.193, 0.350] |

Llama self-report's Brier score improves from 0.580 under GPT-5.4 labels to 0.480 under Kimi labels even though AUROC remains noise. This suggests a calibration-distribution coincidence: the confidence values happen to align better with Kimi's base rate, but they do not rank correct claims better.

### Defensible Claims

The paper makes five substantive claims:

1. Llama 3.1 8B's structured self-reported confidence does not track claim-level correctness on MAUD merger-agreement QA as measured by two independent LLM judges; the frozen human audit remains pending.
2. Linear probes on Llama hidden-state activations recover meaningful correctness signal that self-report does not, and this finding is robust to judge family change.
3. Raw residual features and Goodfire SAE features perform comparably; the residual-vs-SAE difference is not significant under Kimi or on the agreement set.
4. A GPT-5.4 external scorer is the strongest method, with corrected AUROC around 0.87 after accounting for same-family judge coupling.
5. The claim-level judge-proxy protocol does not transfer to CUAD as prompted; claim extraction is unstable on that task without a different upstream pipeline.

### Human Audit Status

The human audit is not complete. The packet is frozen and committed, but no valid human labels have been provided. Therefore this draft does not claim human validation. The planned audit will be reported with N=30 prominently and treated as convergent evidence, not as a replacement for the 150-claim LLM-judge analysis.

The audit can change the paper's interpretation. If humans mostly match the two-judge agreement set, the LLM-judge protocol becomes more credible as a proxy labeler for this narrow domain. If humans systematically prefer one LLM judge on disagreement claims, the second-judge result becomes a model-family sensitivity result. If humans often disagree with both judges, then the main conclusion must weaken: the probes may track LLM-judge labels without tracking human legal judgment.

## Threats to Validity

First, all correctness judgments are still mediated by LLM judges until the human audit is complete. The second judge reduces same-family coupling, but it does not create legal ground truth.

Second, the human audit is small by design. Thirty claims can reveal whether the LLM judges are obviously misaligned with legal-literate annotators, but it cannot support broad claims about legal correctness across MAUD.

Third, the study uses one base model: Llama 3.1 8B Instruct. The result does not imply that larger Llama models, GPT-family models, or other open models expose the same activation signal in the same way.

Fourth, the domain is one legal QA slice: MAUD merger-agreement questions. CUAD failed the readiness screen under this extraction protocol. That means the claim-level method is not automatically portable to other legal corpora.

Fifth, the SAE comparison is narrow. It uses one Goodfire SAE configuration and one layer. A different SAE, layer, or aggregation strategy could behave differently.

Sixth, the held-out train/test probe split is small, with 22 test claims in the richer probe run. The full-corpus quick check is useful for signal detection, but not a substitute for a larger independent test set.

Seventh, AUROC is a ranking metric. It shows whether higher scores tend to land on true claims, not whether the reported probabilities are calibrated for downstream decision thresholds. Brier scores help, but calibration under one judge can improve while ranking does not, as seen with Llama self-report under Kimi labels.

## Discussion

The defensible result is not that probes know law. The defensible result is that Llama hidden states contain claim-level correctness signal that Llama's verbal self-report fails to expose, in a frozen MAUD merger-agreement QA setting, under two independent LLM judge families.

This matters because self-report is tempting. It is cheap, easy to ask for, and legible to users. But in this experiment, it does not rank correctness. The model can present confidence without that confidence tracking the factual status of individual claims.

The residual probe result suggests that the model internally represents something relevant to correctness even when its explicit confidence is poor. This aligns with a growing body of activation-probe work: hidden states can contain uncertainty or hallucination-related information not faithfully surfaced in text.

The SAE result is more cautious. SAE features are appealing because they may be more interpretable than residual vectors. Here, however, they do not clearly outperform residuals. For this task, interpretability does not yet buy better discrimination.

The GPT-5.4 scorer result is useful but must be corrected. The same-family score of 0.944 is not the number to headline as independent performance. The Kimi-scored value of 0.872 is stronger evidence that GPT-5.4 scoring transfers beyond self-consistency, and it remains the best single method in the table.

The next paper-quality requirement is the human audit. If humans mostly agree with the LLM judges on the agreement stratum and split sensibly on disagreement cases, the methods claim becomes much stronger. If humans disagree with both judges, the result becomes more interesting but less publishable as a correctness-detection paper: it would show that the judge-proxy target is not tracking human legal judgment well enough.

The most likely productive follow-up is not a larger probe. It is a better label study. More architectures would be easy to add, but they would not answer the main validity question. A small human audit answers whether the current proxy target is pointed in the right direction. A larger expert-labeled set would then let the field ask the modeling question cleanly: which internal representations best predict human-grounded claim correctness?

## Conclusion

Llama self-report does not track claim-level correctness in this MAUD setting. Llama hidden activations do contain usable correctness signal, and that signal survives a judge-family change. The result is promising but remains proxy-label evidence until the frozen human audit is completed.

## References

- Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, and Yarin Gal. "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs." https://arxiv.org/abs/2406.15927
- Oscar Obeso, Andy Arditi, Javier Ferrando, Joshua Freeman, Cameron Holmes, and Neel Nanda. "Real-Time Detection of Hallucinated Entities in Long-Form Generation." https://arxiv.org/abs/2509.03531
- Anthropic. "Mapping the Mind of a Large Language Model." https://www.anthropic.com/research/mapping-mind-language-model
- Lindsey et al. "On the Biology of a Large Language Model." https://transformer-circuits.pub/2025/attribution-graphs/biology.html
