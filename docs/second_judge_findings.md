# Second-Judge Sensitivity Study: MAUD Judge-Proxy Results

## Header numbers fixed before interpretation

| Method | AUROC under judge 1 (GPT-5.4) | AUROC under judge 2 (Kimi K2.6) | AUROC on judge-agreement set |
| --- | ---: | ---: | ---: |
| Llama self-report | 0.511 [0.416, 0.599] | 0.466 [0.420, 0.600] | 0.486 [0.385, 0.609] |
| GPT-5.4 external scorer | 0.944 [0.909, 0.983] | 0.872 [0.806, 0.923] | 0.981 [0.932, 1.000] |
| Residual activation probe | 0.771 [0.692, 0.841] | 0.707 [0.626, 0.784] | 0.793 [0.700, 0.868] |
| SAE feature probe | 0.677 [0.584, 0.762] | 0.652 [0.555, 0.736] | 0.712 [0.611, 0.810] |

Intervals are 95% percentile bootstrap CIs from 1000 paired claim-level resamples.

The same table with Brier scores:

| Method | Brier under judge 1 | Brier under judge 2 | Brier on judge-agreement set |
| --- | ---: | ---: | ---: |
| Llama self-report | 0.580 | 0.480 | 0.509 |
| GPT-5.4 external scorer | 0.161 | 0.169 | 0.101 |
| Residual activation probe | 0.244 | 0.290 | 0.231 |
| SAE feature probe | 0.300 | 0.326 | 0.268 |

## Second-judge model choice

The second judge is `moonshotai/kimi-k2.6` via Prime Intellect Inference. It was selected because Kimi K2.6 was available at runtime, it is from the Moonshot/Kimi family rather than the GPT family, and it is the highest-priority available model under the requested Kimi -> Qwen -> GLM selection order.

The recorded model-choice note is `docs/second_judge_model_choice.md`.

The completed run produced 150/150 labels across 135/135 examples after retrying nine timeout failures with a longer read timeout. No malformed rows or refused examples remained in the final artifact. Recorded usage was 123,427 prompt tokens, 448,765 completion tokens, and about $1.91 estimated cost.

## Judge agreement

- Shared claims: 150
- Missing v2 labels: 0
- Raw agreement: 0.733
- Cohen's kappa: 0.572

| Judge 1 label | Judge 2 true | Judge 2 partially_true | Judge 2 false |
| --- | ---: | ---: | ---: |
| true | 54 | 5 | 3 |
| partially_true | 19 | 41 | 8 |
| false | 4 | 1 | 15 |

Agreement by GPT-5.4 judge label:

- `true`: n=62, agreement=0.871
- `partially_true`: n=68, agreement=0.603
- `false`: n=20, agreement=0.750

## Outcome call

Method ordering is stable across both judges and in the judge-agreement set, supporting the original methods claim. However, the GPT-5.4 external scorer's judge-1 AUROC of 0.944 is meaningfully inflated by same-family coupling with the label source. Under the independent Kimi judge, its AUROC is 0.872 - still the strongest method, but 0.07 lower than the original headline. Readers should treat 0.87, not 0.94, as the defensible estimate of that method's performance.

The absolute scores moved, which is expected. GPT-5.4 external scoring fell from 0.944 to 0.872 under Kimi labels, so the same-family coupling concern was real in direction. But it did not collapse, and it remained the top-ranked method. Residual and SAE probes also dropped under judge 2, but residual stayed ahead of SAE and both stayed well above Llama self-report. The agreement set strengthened the same ordering rather than reversing it.

## What the agreement set tells us

The 110-claim agreement set is the most interpretively valuable subset in the current evidence. All strong methods improve there: GPT-5.4 external scoring rises to 0.981, the residual probe to 0.793, and the SAE probe to 0.712. Llama self-report remains noise at 0.486.

This pattern is evidence that the methods are tracking a real claim-correctness signal rather than merely fitting one judge's artifacts. A method that mainly captured idiosyncrasies of GPT-5.4 labels would not be expected to improve on the high-confidence two-judge subset. The agreement-set result also suggests that label noise is now a dominant remaining source of error: once the two judges agree, the strong methods separate true from not-true claims more cleanly.

## Paired method deltas

| Context | Delta | AUROC delta | 95% CI | Excludes zero? |
| --- | --- | ---: | ---: | --- |
| Judge 1 | GPT-5.4 - residual | 0.173 | [0.108, 0.256] | yes |
| Judge 1 | residual - SAE | 0.094 | [0.008, 0.184] | yes, narrowly |
| Judge 1 | residual - self-report | 0.260 | [0.152, 0.395] | yes |
| Judge 2 | GPT-5.4 - residual | 0.165 | [0.066, 0.248] | yes |
| Judge 2 | residual - SAE | 0.055 | [-0.048, 0.145] | no |
| Judge 2 | residual - self-report | 0.241 | [0.066, 0.327] | yes |
| Agreement set | GPT-5.4 - residual | 0.188 | [0.094, 0.273] | yes |
| Agreement set | residual - SAE | 0.081 | [-0.020, 0.186] | no |
| Agreement set | residual - self-report | 0.307 | [0.164, 0.430] | yes |

The residual-vs-SAE comparison should not be treated as a stable finding. It is directionally residual-favored, but it does not reliably exclude zero under the independent judge or on the agreement set.

Llama self-report's Brier score improves under judge 2 even though its AUROC does not, suggesting the self-report confidences happen to be better calibrated against Kimi's label distribution than GPT-5.4's. This is a calibration-distribution coincidence rather than a ranking improvement.

## Human audit status

The human-audit packet is frozen at `data/annotations/maud_human_audit_packet.jsonl` with 30 blinded claims. Completed human labels are still pending, so this report does not yet claim human validation. The pending analysis artifact is `artifacts/runs/maud_human_audit_analysis.json`.

## What this result does and does not show

This sensitivity pass tests whether the first MAUD judge-proxy story survives a change in LLM judge family. It does not turn either judge into legal ground truth, and it does not validate the labels against human legal judgment. Stable ordering across two LLM judges would make the proxy result more credible; broad judge disagreement would instead show that the target label itself is unstable.

The probe numbers should be read as fixed-artifact sensitivity scores: the probe scoring path is fit only from judge-1 labels and then compared with judge-2 labels or the judge-agreement subset. Judge-2 labels are not used to retrain the probes.

To move from stable across two LLM judges to stable against human legal judgment, the next step is a small source-bound human audit: sample claims from agreement and disagreement regions, have legal annotators score the frozen claim list against the same excerpt/question/answer bundle, and report where both LLM judges agree or diverge from the human labels.
