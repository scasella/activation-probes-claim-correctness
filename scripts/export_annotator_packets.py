from __future__ import annotations

import argparse
from pathlib import Path

from interp_experiment.data.annotation import load_annotation_rows, split_rows_by_annotator
from interp_experiment.io import write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one JSONL annotation packet per annotator from a combined packet.")
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/annotations/maud_pilot_annotation_packet.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/annotations/maud_pilot_packets"))
    parser.add_argument("--manifest-json", type=Path, default=Path("data/annotations/maud_pilot_packets/manifest.json"))
    args = parser.parse_args()

    rows = load_annotation_rows([args.input_jsonl])
    by_annotator = split_rows_by_annotator(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"input_jsonl": str(args.input_jsonl), "annotators": {}}
    for annotator_id, annotator_rows in sorted(by_annotator.items()):
        output_path = args.output_dir / f"{annotator_id}.jsonl"
        write_jsonl(output_path, annotator_rows)
        manifest["annotators"][annotator_id] = {
            "row_count": len(annotator_rows),
            "path": str(output_path),
        }
        print(f"Wrote {len(annotator_rows)} rows for annotator {annotator_id} to {output_path}")
    write_json(args.manifest_json, manifest)
    print(f"Wrote packet manifest to {args.manifest_json}")


if __name__ == "__main__":
    main()
