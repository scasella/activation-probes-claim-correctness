# FActScore Validation Report

## Executive Summary

The FActScore transfer check is a strong positive result for the residual-probe methodology. On the full ChatGPT portion of the human-annotated FActScore corpus, a linear probe trained on Llama 3.1 8B layer-19 residual activations substantially outperformed Llama's own structured self-report confidence at ranking human-labeled atomic facts.

The headline result is large: residual probe AUROC **0.802** versus Llama self-report AUROC **0.541** on held-out FActScore claims. Brier score also favors the residual probe, **0.171** versus **0.298**. The paired AUROC delta is **0.262**, with a 200-resample paired bootstrap interval of **[0.220, 0.301]**, excluding zero.

This matters because FActScore is a cleaner transfer target than FELM for this project. It has atomic-fact annotations, a constrained biography domain, and human factuality labels. Unlike MAUD, it is not legal QA. Unlike FELM, its claim granularity matches the probe target more naturally. The result therefore weakens the concern that the MAUD finding is just a legal-domain artifact.

Scope note: this run covers the **ChatGPT-generated FActScore biographies**, not every model source in the full FActScore release. The adapter produced 157 usable ChatGPT biographies and 4,886 non-IR atomic facts, all of which were materialized cleanly for this report.

## Research Question

The original MAUD study showed that Llama self-report was near-random at claim-level correctness, while linear probes on hidden activations recovered substantial signal. FELM then produced a directionally positive but statistically inconclusive transfer result: residual AUROC 0.652 versus self-report AUROC 0.511, with the paired delta interval crossing zero.

FActScore was chosen to separate two possibilities:

- The MAUD finding might be narrow to legal QA and contract-grounded claims.
- FELM might have been a noisy transfer target because its segment labels and curated reference snippets did not align cleanly with Llama-activation probing.

The question for this run was therefore simple: **does the residual probe beat Llama self-report on FActScore, using FActScore's original human factuality labels?**

The answer is yes. On FActScore, the residual probe beats self-report by an even larger AUROC margin than in MAUD.

## Dataset And Label Mapping

The experiment uses the ChatGPT portion of the human-annotated FActScore data. Each biography is treated as one example, and each human atomic fact is treated as one claim-level unit. This is a better match to the MAUD probe setup than FELM's broader response segments, because the evaluation unit is already close to a single factual proposition.

FActScore labels were mapped into the binary correctness target used for AUROC:

| FActScore label | Canonical target | Handling |
| --- | --- | --- |
| `Supported` | `true` | Positive class |
| `Not-supported` | `false` | Negative class |
| `Irrelevant` | dropped | Excluded from factuality target |

Dropping `Irrelevant` is important. Those rows are off-topic or non-factuality judgments; mixing them into the correctness target would conflate factual support with relevance.

The adapter produced **157 usable biographies** and **4,886 non-IR atomic facts**. The post-drop label distribution was:

| Label | Count |
| --- | ---: |
| `true` / Supported | 3,194 |
| `false` / Not-supported | 1,692 |

All 4,886 adapted claims were materialized with both residual features and self-report predictions. The materialization summary reports **0 failures** and **0.0 self-report failure rate**.

## Span And Activation Design

The main methodological wrinkle is that FActScore's human atomic facts are often canonicalized. They are human-written factual statements derived from the biography, not necessarily exact substrings of the generated biography text. Exact atomic-fact substring matching occurred for only **8** facts, an exact match rate of about **0.15%**.

To avoid unreliable string matching, the adapter pools activations over the parent annotated sentence span from which each atomic fact came. The label remains atomic-fact-level, but the representation is sentence-span pooled. This is a compromise, but a defensible one: it preserves the original human labels and avoids hallucinating token spans for canonicalized claims.

The token alignment check was clean. All **4,886/4,886** adapted claims aligned to token spans. Median span length was **27 tokens**, with an interquartile range from **22** to **33** tokens and no 1-2 token spans. That matters because very short spans would make mean-pooling unstable; here the parent-sentence spans are long enough to be robust.

## Model Signals Compared

The evaluation compares two signals:

| Signal | Description |
| --- | --- |
| Llama self-report | Llama 3.1 8B is asked for structured confidence on each fixed atomic fact. |
| Residual probe | A logistic-regression probe is trained on Llama layer-19 residual activations pooled over each claim's span. |

No SAE probe, second judge, external scorer, or matched-generation condition is included here. This is intentionally a narrow transfer check: residual probe versus self-report.

The train/validation/test split is by biography, not by claim, to avoid leakage from facts within the same biography appearing in both train and test. Split sizes were:

| Split | Claims |
| --- | ---: |
| Train | 3,353 |
| Validation | 791 |
| Test | 742 |

The probe used L2 logistic regression with a C-sweep selected on validation. The selected value was **C = 0.01**. Final evaluation was performed once on the held-out test split.

## Results

### Headline Test Metrics

| Method | Test AUROC | Test Brier |
| --- | ---: | ---: |
| Llama self-report | 0.541 | 0.298 |
| Residual probe | 0.802 | 0.171 |

The residual probe is well above chance and substantially above self-report. Llama self-report is only slightly above random on this held-out subset, while the probe reaches a strong factuality-ranking score.

The Brier result points the same direction. Self-report is also worse calibrated against the FActScore labels than the residual-probe probabilities. This differs from some judge-label settings where self-report can have a tolerable Brier score because it happens to match the label base rate. Here both ranking and calibration favor the probe.

### Probe Fit Diagnostics

| Split | Residual probe AUROC | Residual probe Brier |
| --- | ---: | ---: |
| Train | 0.939 | 0.097 |
| Validation | 0.728 | 0.220 |
| Test | 0.802 | 0.171 |

The training AUROC is high, as expected for a high-dimensional linear probe, but the validation and test numbers remain strong. The test AUROC being higher than validation is reassuring: the result is not just an overfit validation artifact. It also clears the preregistered stop condition by a wide margin.

### Paired Delta

The paired bootstrap compares the residual-probe score and self-report score on the same held-out claims.

| Comparison | AUROC delta | 95% bootstrap interval | Excludes zero? |
| --- | ---: | ---: | --- |
| Residual probe - self-report | 0.262 | [0.220, 0.301] | yes |

This is a scoping bootstrap with 200 resamples, not the 1,000-resample publication pass used in the MAUD paper tables. Still, the result is not marginal. The interval excludes zero comfortably, and the observed point estimate is large.

## Interpretation

The main finding is that the residual-probe signal transfers to FActScore much more cleanly than it did to FELM. That pushes against the hypothesis that MAUD was a one-off artifact of legal QA or judge-proxy labels.

FActScore differs from MAUD in several important ways:

- It uses biography generation rather than contract question answering.
- It uses original human labels rather than LLM judge-proxy labels.
- It contains thousands of atomic facts rather than 150 legal claims.
- The generations come from ChatGPT, while activations are extracted from Llama reading those generations.

Despite those differences, the same qualitative pattern appears: Llama's self-report does not track factual correctness well, while a linear probe over hidden activations does.

The generation-mismatch caveat is real. These are not Llama-generated biographies, so the probe is not measuring Llama's hidden state while producing its own answer. It is measuring Llama's hidden representation while reading ChatGPT text. But that makes the result interesting in a different way: the correctness signal appears to be present in Llama's representation of the text-content, not only in its generation-time state.

This is exactly the kind of result a future runtime library cares about. If activation probes can recover factuality signal from model-internal representations across MAUD and FActScore, then the library can be framed around claim-level uncertainty scaffolding rather than a single legal benchmark. The library should still be conservative, but the evidence base is no longer MAUD-only.

## Comparison To MAUD And FELM

| Dataset | Label source | Unit | Residual AUROC | Self-report AUROC | Interpretation |
| --- | --- | --- | ---: | ---: | --- |
| MAUD | LLM judges, with human audit pending | Legal claim | ~0.77 under GPT-5.4 judge | ~0.51 | Strong core result |
| FELM-wk | Human segment labels | Response segment | 0.652 | 0.511 | Directional but inconclusive |
| FActScore | Human atomic-fact labels | Biography atomic fact | 0.802 | 0.541 | Strong transfer result |

The pattern is now more coherent. MAUD and FActScore both show a large probe-over-self-report gap. FELM remains the odd case: positive in direction, but smaller and not statistically distinguishable in the scoping bootstrap. The most plausible explanation is not simply generation mismatch, because FActScore also has generation mismatch and still produces a strong result. FELM's weaker result is more likely tied to its segment granularity, evidence coverage, or annotation-target mismatch.

That interpretation is reinforced by the matched-generation FELM repair. Once Llama-generated FELM answers were judged against reference-bounded evidence, many claims became `not_enough_evidence` rather than cleanly true or false. The FActScore setup avoids much of that problem by using already-annotated atomic facts in a constrained biography domain.

## What This Does And Does Not Show

This result supports three claims:

- Linear probes on residual activations recover factual correctness signal outside the MAUD legal-QA setting.
- Llama self-report is not a reliable claim-level factuality signal on FActScore.
- FActScore is a positive transfer scaffold for a future claim-level uncertainty library.

It does not show:

- That probes detect truth in an absolute sense.
- That the method works for arbitrary domains or arbitrary generation settings.
- That sentence-span pooling is the final best representation for FActScore.
- That live runtime flagging will work without additional engineering.

The strongest version of the claim is methodological: hidden activations contain claim-level factuality signal that self-report fails to surface, and that pattern now appears in both MAUD and FActScore.

## Library Implications

The library roadmap should change from "MAUD-only until proven otherwise" to "MAUD-first, with FActScore-supported transfer." The conservative version of the library story is:

- MAUD remains the legal-QA anchor.
- FActScore becomes the first positive non-legal transfer scaffold.
- FELM remains a documented boundary or unresolved dataset.
- Early users should be told that scaffolding quality depends heavily on claim granularity and label quality.

The practical design lesson is that dataset adapters matter as much as probe training. FActScore worked because its annotation unit matched the desired probe unit. FELM was messier because the labels were coarser and the reference-bounded matched-generation repair produced too much `not_enough_evidence`. A runtime library should therefore make claim construction and label provenance first-class concepts, not hidden preprocessing details.

## Remaining Work

The full ChatGPT subset is complete. The main remaining experimental extension is not more FActScore materialization, but a matched-generation non-legal dataset where the target model generates the claims itself and labels remain well grounded. FActScore tells us the signal transfers across domain and label source. It does not fully answer the matched-generation question.

If this result is promoted into a publication table, rerun the bootstrap with the same 1,000-resample protocol used for the MAUD paper. The 200-resample interval is sufficient for this scoping validation, but the paper should keep statistical procedures consistent across headline tables.

## Bottom Line

FActScore is the best transfer evidence so far. The residual probe substantially outperforms Llama self-report on human-labeled biography atomic facts, and the effect size is close to the MAUD result rather than the weaker FELM result. The clean conclusion is that the MAUD probe finding is no longer isolated: hidden activations appear to carry claim-level correctness signal across at least two meaningfully different factuality settings.
