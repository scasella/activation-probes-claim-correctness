# Tutorial: Training A Domain Probe

1. Build a canonical JSONL file where each row contains a prompt, generation, and labeled claims.
2. Use stable claim spans. If the label describes an atomic fact but the exact fact does not appear in text, document the pooling compromise.
3. Run `probemon train --dataset path/to/data.jsonl --model meta-llama/Llama-3.1-8B-Instruct --output domain_probe.npz`.
4. Inspect validation AUROC and Brier before using the probe.
5. Use `score_generation` with the same model family and layer. Treat OOD warnings as reminders that scores are domain-bound.

FActScore is the worked example in the methods paper: atomic facts are human-labeled, but activations are pooled over parent sentence spans because canonical facts rarely appear verbatim in the biography text.
