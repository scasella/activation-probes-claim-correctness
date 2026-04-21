# MAUD Annotation Workflow

This workflow is the current clean path for the human-annotation phase.

## Files to use

- Combined packet: `data/annotations/maud_pilot_annotation_packet.jsonl`
- Per-annotator packet exports: `data/annotations/maud_pilot_packets/`
- Agreement outputs: `data/annotations/maud_pilot_annotation_agreement.json` and `.md`

## Export per-annotator files

Run:

```bash
uv run python scripts/export_annotator_packets.py
```

This writes one file per annotator in both `jsonl` and `csv` form by default.

## Annotator instructions

Each annotator should edit only their own packet file.

Required fields per row:

- `correctness_label`: `true`, `false`, or `partially_true`
- `load_bearing_label`: `yes` or `no`
- `flip_evidence_text`: required when `load_bearing_label=yes`
- `notes`: optional

Do not change:

- `claim_id`
- `example_id`
- `question_text`
- `claim_text`
- `annotator_id`

## Evaluate completed annotations

After both annotators return completed files, run:

```bash
uv run python scripts/evaluate_annotation_pilot.py \
  --input-jsonl data/annotations/maud_pilot_packets/a1.jsonl data/annotations/maud_pilot_packets/a2.jsonl \
  --output-json data/annotations/maud_pilot_annotation_agreement.json \
  --output-md data/annotations/maud_pilot_annotation_agreement.md \
  --attempt-index 1
```

If the annotators worked in CSV instead, pass the `.csv` files instead of `.jsonl`.

## Interpret the gate

- `status=incomplete`: at least one packet still has blank required labels
- `status=pass`: load-bearing agreement met the threshold
- `status=fail_revise_once`: revise guidelines once and rerun the pilot
- `status=fail_drop_load_bearing`: drop load-bearing and continue correctness-only

## Export disagreement packet

After evaluation, create an adjudication packet with:

```bash
uv run python scripts/export_disagreement_packet.py \
  --input-json data/annotations/maud_pilot_annotation_agreement.json \
  --output-jsonl data/annotations/maud_pilot_disagreements.jsonl \
  --output-csv data/annotations/maud_pilot_disagreements.csv
```

This packet is only for disagreement review and adjudication. It is not a substitute for the original annotator files.
