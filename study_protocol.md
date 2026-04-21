# Study Protocol

## Research question

On a domain-specific long-form generation task, do interpretability-derived uncertainty signals outperform a model's structured self-report of its own uncertainty, measured against expert-annotated ground truth?

## Head-to-head variants

- Variant A1: Llama 3.1 8B Instruct self-report on a frozen claim list
- Variant A2: GPT-5.4 cached external decomposition/scoring on the same frozen claim list
- Variant B: Probe-based claim uncertainty from Llama activations

The claim list is canonical and shared. Decomposition quality is intentionally removed as a confound for the main comparison.

## Dataset contract

The main source pool is `300` examples:

- `150` MAUD-style merger-agreement QA examples
- `150` CUAD-derived non-merger contract-review examples

Public answers are seeds only. Expert annotators remain the source of truth for claim correctness, load-bearingness, and flip evidence.

Implementation note: the legacy `theatticusproject/cuad-qa` dataset script is not supported by the current `datasets` runtime. The current repo default uses the public QA derivative `chenghao/cuad_qa` as the CUAD-side source pool.

## Splits and holdouts

- Freeze splits at the contract level before probe tuning.
- Reserve a dedicated `pilot` partition before the final train/validation/test split.
- Default split ratios are `70/15/15` over the non-pilot remainder.
- The test split is not inspected until Phase 4.
- Cross-distribution evaluation is source-family shift:
  - primary: train on CUAD-derived, test on MAUD-derived
  - appendix: reverse direction

## Claim labeling defaults

Generated claim rows contain:

- `claim_id`
- `example_id`
- `claim_text`
- `token_start`
- `token_end`
- `annotation_version`

Generated claims are unlabeled. Annotation labels live in downstream annotation artifacts only.

Answer-run bundles are persisted separately and contain:

- `example_id`
- `source_corpus`
- `task_family`
- `prompt_text`
- `answer_text`
- `model_name`
- `extractor_name`
- `token_ids`
- `token_offsets`

Primary correctness evaluation is binary with `partially-true -> incorrect`.

Sensitivity tables also report:

- `partially-true` excluded
- `partially-true -> correct`

## Baseline schema

Both self-report methods emit one record per canonical claim with:

- `claim_id`
- `correctness_confidence`
- `load_bearing_label`
- `load_bearing_confidence`
- `flip_evidence_text`
- `raw_json`
- `prompt_version`
- `model_name`

GPT-5.4 outputs are cached artifacts. The repo stores prompt packets, raw outputs, and parsed outputs. Modal reproducibility starts from those cached artifacts.

Implementation note: the current SAELens release `goodfire-llama-3.1-8b-instruct` exposes the Goodfire checkpoint under SAE id `layer_19`.

## Probe targets

- Correctness probe: `Ridge` on claim-level resampling entropy estimated from `8` stochastic Llama generations per train/validation example
- Load-bearing probe: `LogisticRegression(L2)` on annotated load-bearing labels
- Stability probe: `LogisticRegression(L2)` on claim survival under `3` prompt paraphrases
- Direct correctness logistic probe: ablation only

Claim features are mean-pooled over exact-answer-token spans.
The probe path must consume the exact persisted answer run that produced the canonical claim spans; it must never regenerate a different answer at feature-extraction time.

## Reliability gate

Phase 1 uses a `30`-example pilot with `2` annotators.

- Measure Cohen's kappa or Krippendorff's alpha on load-bearingness.
- If agreement is below `0.6`, revise guidelines once and run one repilot.
- If agreement is still below `0.6`, drop load-bearing from the main study and continue as correctness-only.
- The failed load-bearing reliability result is itself a reportable finding and must remain visible in the write-up.

## Phase gates

### Phase 0

- Verify gated access to `meta-llama/Llama-3.1-8B-Instruct`
- Verify a single end-to-end prompt through:
  - Llama answer generation
  - token span extraction
  - Goodfire SAE encoding
  - raw residual export
  - cached baseline JSON validation
- Commit `docs/abstract_v1.md` before any data collection

### Phase 1

- Build the source pool
- Run the pilot readiness gate on a non-test pilot before annotation
- Run annotation pilot and repilot policy if needed
- Stop here only if the entire study becomes non-viable; otherwise fallback to correctness-only

### Phase 2

- Tune prompting baselines on train/validation only
- Freeze prompts before touching test artifacts

### Phase 3

- Train probes on train, tune on validation
- Do not inspect test metrics

### Phase 4

- Evaluate all frozen methods on held-out test
- Report AUROC, Brier, and bootstrap CIs with `1000` paired resamples
- Include `10` representative examples, threats to validity, and the decision memo

## Scope guards

- Do not add new domains before finishing the legal-contract pass.
- Do not add new base models before finishing the Llama 3.1 8B pass.
- Do not add MLP probes unless linear probes clearly fail and the deviation is explicitly documented.
- Do not use synthetic labels as ground truth.
- Do not use Goodfire hosted APIs.
