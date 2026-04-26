# ProbeMon API

## `load_probe(id_or_path)`

Loads a pretrained probe id or a `.npz` probe artifact. The artifact contains `direction`, `bias`, `platt_a`, `platt_b`, and JSON `metadata`.

## `score_generation(...)`

Scores an existing generation. By default, the generation is split into sentences and each sentence receives a calibrated score. You may pass explicit `claim_spans` with `text`, `char_start`, and `char_end`.

## `generate_with_monitoring(...)`

Generates text through an activation extractor and then scores the generated answer. v0.1 returns post-hoc sentence scores rather than token-streaming overlays.

## `fit_probe(...)`

Fits an L2 logistic-regression probe on canonical labeled claims. The function can extract activations itself or consume precomputed feature matrices for tests and reproducibility checks.
