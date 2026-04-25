# Training Intervention Design Memo

## Decision

Do not design the next intervention around steering a small set of Goodfire SAE features. The residual correctness probe's signal is distributed in the SAE basis, so a feature-steering intervention would be poorly motivated at this stage.

The better next experiment is self-report alignment: train or prompt-tune the model to expose correctness signal that is already present in hidden activations but not reflected in its structured confidence.

## Rationale

The methods-paper result established a gap: residual activations predict claim-level correctness substantially better than Llama's self-reported confidence. The interpretability follow-up asked whether that gap is explained by a compact set of interpretable SAE features. It is not. The top 50 SAE features explain only about 2% of alignment mass, close to random-vector behavior.

That points away from feature-level steering and toward subspace-level supervision. The model appears to have distributed internal evidence about correctness, but the self-report pathway is not using that evidence reliably.

## Candidate Intervention

The most direct intervention is a small supervised self-report calibration pass:

1. Keep answer generation fixed or minimally changed.
2. Add a self-report head or structured confidence target after answer generation.
3. Supervise confidence against judge-proxy correctness labels, with a small human-audit validation set held out.
4. Compare against the original self-report, residual probe, and external scorer.

The key question is whether training can move verbal confidence toward the hidden-state probe signal without simply teaching the model to mimic a judge label distribution.

## Measurement

Primary metrics:

- Claim-level AUROC for self-reported correctness confidence.
- Brier score for calibration.
- Paired deltas against the original self-report baseline.

Guardrails:

- Do not train on the 30-claim human audit.
- Keep the frozen 150-claim MAUD set as the continuity benchmark, but add a fresh held-out MAUD slice before making a stronger claim.
- Track answer quality separately so confidence calibration does not improve by making answers vague or underconfident.

## Alternative Mechanistic Follow-Up

If the next goal is mechanistic understanding rather than training, use subspace-level tests:

- Fit residual probes across folds and measure direction stability.
- Sweep layers to identify where correctness signal appears and disappears.
- Compare multiple SAE releases or expansion factors if available.
- Run causal tests on low-rank residual directions, not individual SAE features.

Those tests are better matched to the distributed-signal finding than Neuronpedia-style top-feature inspection.

## Boundary

This memo sketches intervention design only. It does not implement training, causal patching, feature steering, or new probe architectures.
