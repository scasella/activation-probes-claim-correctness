from __future__ import annotations

import argparse
from pathlib import Path

from interp_experiment.data.annotation import load_annotation_rows, split_rows_by_annotator
from interp_experiment.io import write_csv, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one JSONL annotation packet per annotator from a combined packet.")
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/annotations/maud_pilot_annotation_packet.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/annotations/maud_pilot_packets"))
    parser.add_argument("--manifest-json", type=Path, default=Path("data/annotations/maud_pilot_packets/manifest.json"))
    parser.add_argument(
        "--format",
        choices=["jsonl", "csv", "both"],
        default="both",
        help="Packet file format(s) to export for each annotator.",
    )
    args = parser.parse_args()

    rows = load_annotation_rows([args.input_jsonl])
    by_annotator = split_rows_by_annotator(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"input_jsonl": str(args.input_jsonl), "annotators": {}}
    for annotator_id, annotator_rows in sorted(by_annotator.items()):
        paths: dict[str, str] = {}
        if args.format in {"jsonl", "both"}:
            jsonl_path = args.output_dir / f"{annotator_id}.jsonl"
            write_jsonl(jsonl_path, annotator_rows)
            paths["jsonl"] = str(jsonl_path)
        if args.format in {"csv", "both"}:
            csv_path = args.output_dir / f"{annotator_id}.csv"
            write_csv(csv_path, annotator_rows)
            paths["csv"] = str(csv_path)
        manifest["annotators"][annotator_id] = {
            "row_count": len(annotator_rows),
            "paths": paths,
        }
        print(f"Wrote {len(annotator_rows)} rows for annotator {annotator_id} to {paths}")
    write_json(args.manifest_json, manifest)
    print(f"Wrote packet manifest to {args.manifest_json}")


if __name__ == "__main__":
    main()
