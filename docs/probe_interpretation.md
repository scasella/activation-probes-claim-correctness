# Interpreting the Residual Correctness Probe

## Summary

The residual correctness probe's signal is distributed across the residual stream rather than concentrated in a small set of Goodfire layer-19 SAE features. Projecting the fixed residual probe direction into the SAE basis produces top-aligned features, but their aggregate mass is close to what appears for random residual directions. Per the preregistered gate, this is a negative feature-localization result: the probe contains real claim-correctness signal, but this analysis does not support a compact, interpretable feature story.

This result is still useful. It says the residual probe's advantage over self-report is not obviously explained by a few monosemantic features that the model could simply expose or steer. The safer interpretation is that claim-level correctness information is weakly distributed across many residual directions, at least at layer 19 and in this Goodfire SAE basis.

## Inputs

This analysis used the frozen MAUD full-run artifacts:

- Residual claim features: `artifacts/runs/maud_full_probe_features_residual.jsonl`
- SAE claim features: `artifacts/runs/maud_full_probe_features_sae.npz`
- Judge-1 labels: `data/annotations/maud_full_judge_annotations.jsonl`
- Goodfire SAE: `goodfire-llama-3.1-8b-instruct`, `layer_19`

No persisted residual-probe model object existed in the repo. To make the interpretation target reproducible, `scripts/materialize_residual_probe_direction.py` materializes a fixed residual-space direction by fitting the same standardized `C=1.0` logistic probe form on the frozen full-corpus residual features and judge-1 labels. The saved direction exactly reproduces the fitted pipeline decision function up to numerical precision: maximum absolute error `4.44e-15`.

This direction is an interpretation target. It does not revise the reported methods-paper metrics.

## Sparsity Gate

The decision gate asked whether the residual probe direction is meaningfully sparse in the SAE feature basis relative to random residual directions. The answer is no.

| Projection view | Probe top-1 L2 mass | Probe top-10 L2 mass | Probe top-50 L2 mass | Random top-50 mean | Random top-50 p95 | Probe Gini | Random Gini mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decoder cosine | 0.065% | 0.498% | 1.962% | 1.365% | 1.461% | 0.604 | 0.594 |
| Encoder projection | 0.081% | 0.611% | 2.178% | 1.925% | 2.076% | 0.603 | 0.602 |

The decoder-cosine view is slightly above the random baseline, but not by enough to support a compact-feature interpretation: the top 50 of 65,536 SAE features still explain only about 2% of the alignment mass. The encoder-projection view is even closer to random. Its top-50 mass barely exceeds the random 95th percentile, and its Gini is essentially identical to the null.

The important distinction is between "some features rank highest" and "a small set of features explains the probe." Every random vector has a top 50. Here, the top 50 are not concentrated enough to justify treating them as the mechanism.

## Gate Outcome

The gate outcome is distributed. Detailed feature characterization is skipped to avoid interpretive overreach.

This means:

- The residual probe's correctness signal is not well localized to a compact set of SAE features at Goodfire layer 19.
- A top-feature narrative would likely be post hoc storytelling rather than a load-bearing explanation.
- The residual-vs-SAE gap in the methods paper is consistent with signal that remains easier to exploit in raw residual space than in the current SAE feature basis.

This does not mean SAE features are useless. The methods paper already showed SAE features carry some correctness signal. The narrower finding here is that the residual probe direction itself does not decompose cleanly into a small, high-coverage SAE feature set.

## Residual-vs-SAE Interpretation

The methods-paper result was that residual probes outperform structured self-report and directionally outperform SAE probes. This analysis sharpens the interpretation:

- Hypothesis A, "the SAE basis lacks a compact feature representation of the relevant signal," is plausible.
- Hypothesis B, "the SAE has a few relevant features but the SAE probe failed to combine them," is not supported by the sparsity diagnostics.
- Hypothesis C, "the signal spans many SAE features and loses effective sample efficiency in SAE space," is the best current explanation.

Because the gate was negative, the constrained top-50 SAE probe test was not run. Training a new top-50 probe after observing that top-50 coverage is only about 2% would not answer the intended interpretability question; it would mostly test whether a very small, weakly selected subset can overfit the available labels.

## Self-Report Correlation

The self-report correlation analysis was also skipped after the negative gate. Its planned interpretation depended on a small set of probe-aligned features worth tracking. Once the signal is distributed, feature-by-feature quartile bins are no longer the right unit of analysis.

The methods-paper finding still stands: Llama 3.1 8B's structured confidence is near-random for ranking claim correctness, while residual activations contain recoverable signal. This follow-up suggests that the missing self-report signal is not a small, obvious set of SAE features that the verbal confidence head simply ignores.

## What This Shows

This analysis shows that a residual correctness-probe direction can be projected into the Goodfire SAE basis, but its alignment is broadly distributed. The top SAE features do not capture enough mass to support a compact mechanistic explanation.

It also supports a cautious explanation for why residual features can outperform SAE features on this task: the residual stream may retain weak correctness information across many directions that are individually small in the SAE basis. Linear probes can aggregate that distributed signal, while sparse feature views may not expose it cleanly.

## What This Does Not Show

This does not prove that claim correctness is intrinsically uninterpretable. It only tests one model, one layer, one SAE, one probe direction, and one legal-QA claim dataset.

It does not rule out:

- A cleaner feature decomposition at another layer.
- A different SAE with better coverage of correctness-relevant directions.
- Multi-feature circuits that are interpretable only above the single-feature level.
- Causal interventions that improve self-report despite distributed evidence.

It also does not establish that the residual probe tracks legal correctness in an absolute sense. The interpretation target inherits the methods paper's proxy-label limitations.

## Implications

The next experiment should not be feature steering against a few hand-picked SAE features. The evidence is too distributed for that to be the first intervention.

The more promising path is representation-level training or calibration:

- Train the model to map its own hidden-state correctness signal into better self-report.
- Use probe scores as an auxiliary target or diagnostic, not as a small feature-steering recipe.
- Evaluate whether self-report improves without degrading answer quality.

If a mechanistic follow-up is still desired, it should move from single-feature inspection to subspace-level analysis: probe subspace stability across folds, layer sweeps, alternative SAEs, and causal tests on low-rank residual directions.
