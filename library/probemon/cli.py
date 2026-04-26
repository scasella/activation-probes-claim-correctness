from __future__ import annotations

import argparse
import json
from pathlib import Path

from probemon import load_probe, score_generation
from probemon.datasets import load_dataset
from probemon.pretrained import list_probes
from probemon.training import fit_probe, load_canonical_dataset


def _cmd_list_probes(_: argparse.Namespace) -> None:
    for probe_id in list_probes():
        probe = load_probe(probe_id)
        print(f"{probe_id}\t{probe.metadata.get('domain_description', '')}")


def _cmd_score(args: argparse.Namespace) -> None:
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    generation = Path(args.generation_file).read_text(encoding="utf-8")
    result = score_generation(
        model=args.model,
        probe=args.probe,
        prompt=prompt,
        generation=generation,
        suppress_ood_warning=args.suppress_ood_warning,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


def _cmd_train(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.dataset) if args.dataset in {"maud-legal-qa", "factscore-biographies"} else load_canonical_dataset(args.dataset)
    print(f"Training probe on {dataset.name}: {len(dataset.examples)} examples, {dataset.n_claims} claims")
    result = fit_probe(
        model=args.model,
        dataset=dataset,
        layer=args.layer,
        output_path=args.output,
        model_name=args.model,
        probe_id=Path(args.output).stem,
    )
    print(
        json.dumps(
            {
                "selected_c": result.selected_c,
                "validation_auroc": result.validation_auroc,
                "validation_brier": result.validation_brier,
                "dataset_stats": result.dataset_stats,
                "output": args.output,
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probemon")
    sub = parser.add_subparsers(dest="command", required=True)
    list_parser = sub.add_parser("list-probes")
    list_parser.set_defaults(func=_cmd_list_probes)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--probe", required=True)
    score_parser.add_argument("--model", required=True)
    score_parser.add_argument("--prompt-file", required=True)
    score_parser.add_argument("--generation-file", required=True)
    score_parser.add_argument("--suppress-ood-warning", action="store_true")
    score_parser.set_defaults(func=_cmd_score)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--model", required=True)
    train_parser.add_argument("--layer", type=int, default=19)
    train_parser.add_argument("--output", required=True)
    train_parser.set_defaults(func=_cmd_train)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
