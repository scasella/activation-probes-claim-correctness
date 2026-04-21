from __future__ import annotations

import argparse
from pathlib import Path

from interp_experiment.io import read_json
from interp_experiment.reporting.writer import render_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Markdown study report from metric and example JSON files.")
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--examples-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--title", default="Interpretability-Derived Uncertainty Decomposition")
    args = parser.parse_args()

    metrics_payload = read_json(args.metrics_json)
    examples_payload = read_json(args.examples_json)
    metrics_rows = examples_payload.pop("__metrics_rows__", None) or [
        {"metric": key, "value": value}
        for key, value in metrics_payload.items()
        if not isinstance(value, list)
    ]
    representative_examples = examples_payload["representative_examples"]
    threats = examples_payload.get("threats_to_validity", [])
    render_report(args.title, metrics_rows, representative_examples, threats, args.output_md)
    print(f"Wrote report to {args.output_md}")


if __name__ == "__main__":
    main()
