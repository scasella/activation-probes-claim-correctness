# ProbeMon v0.1 Design Notes

ProbeMon lives under `library/` to keep the reusable package separate from the methods-paper code while still allowing the initial pretrained artifacts to be exported from local research artifacts.

The v0.1 artifact format is intentionally small: a raw residual-space direction, a bias, Platt parameters, and JSON metadata. The MAUD probe uses the existing materialized residual direction directly. The FActScore probe is exported from the existing full-run feature matrix by reproducing the documented `C=0.01` logistic-regression fit over train+validation and checking that held-out AUROC remains within the expected 0.78-0.82 range.

The runtime path delays heavyweight imports. Importing `probemon` does not import torch or transformers; those are only required when a user asks the library to extract activations from a HuggingFace model. Tests use small dummy extractors and precomputed feature matrices.

The OOD warning is deliberately weak and documented as such. It compares prompt length against training-prompt length metadata and checks overlap with a small set of domain-indicator terms. The warning is a user-facing reminder that probe scores are domain-bound, not a robust distribution-shift detector.

Datasets are not bundled. MAUD redistribution should be reviewed against Atticus Project terms before packaging. FActScore is publicly released through the upstream MIT-licensed repository, but data-specific redistribution terms should still be checked before bundling the human annotations. The v0.1 loader therefore expects local artifacts to avoid shipping a large corpus and to keep provenance explicit.
