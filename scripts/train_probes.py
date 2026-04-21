from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from interp_experiment.io import read_jsonl
from interp_experiment.schemas import ClaimFeatureRow
from interp_experiment.utils import ensure_parent
from interp_experiment.probes.train import train_binary_probe, train_correctness_ridge


def main() -> None:
    parser = argparse.ArgumentParser(description="Train correctness/load-bearing/stability probes from pooled features.")
    parser.add_argument("--features-jsonl", type=Path, required=True)
    parser.add_argument("--task", choices=["correctness", "load_bearing_target", "stability_target"], required=True)
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--c-value", type=float, default=1.0)
    args = parser.parse_args()

    rows = [ClaimFeatureRow.from_dict(item) for item in read_jsonl(args.features_jsonl)]
    if args.task == "correctness":
        bundle = train_correctness_ridge(rows, alpha=args.alpha)
    else:
        bundle = train_binary_probe(rows, target_name=args.task, c_value=args.c_value)
    ensure_parent(args.output_pkl)
    with args.output_pkl.open("wb") as handle:
        pickle.dump(bundle, handle)
    print(f"Saved {bundle.task_name} probe to {args.output_pkl}")


if __name__ == "__main__":
    main()
