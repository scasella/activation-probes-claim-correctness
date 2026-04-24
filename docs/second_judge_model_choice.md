# Second-Judge Model Choice

Date: 2026-04-24

Selected model: `moonshotai/kimi-k2.6`

Provider: Prime Intellect Inference API

Model-list artifact: `artifacts/runs/prime_inference_models_2026-04-24.json`

## Rationale

The requested selection order was Kimi, then Qwen, then GLM. Prime Intellect listed `moonshotai/kimi-k2.6` at runtime, so the primary target family was available and no fallback was needed.

Kimi is from Moonshot rather than OpenAI, Anthropic, or Google. That family distance is the purpose of this sensitivity pass: judge 2 should not be another GPT-family judge.

The model is priced at $0.95 per million input tokens and $4.00 per million output tokens in the Prime model listing captured for this run. The full 150-claim judge pass should remain comfortably below the requested $50 budget unless the endpoint produces unusually long responses or repeated retries.

Actual completed run usage: 123,427 prompt tokens, 448,765 completion tokens, and an estimated $1.91 total cost. The model produced valid schema-conforming annotations for all 135 examples / 150 claims after retrying a small tail of timeout failures with a longer read timeout.

## Guardrails

This phase uses exactly one second judge: `moonshotai/kimi-k2.6`.

The judge prompt is copied from `scripts/materialize_judge_annotations.py`; only the transport changes from Codex App Server to Prime Intellect Inference.

GPT, Claude, and Gemini-family models remain out of scope for judge 2.
