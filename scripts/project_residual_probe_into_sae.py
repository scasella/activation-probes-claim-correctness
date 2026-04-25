from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from interp_experiment.io import write_json


def _extract_json_section(text: str, label: str) -> dict[str, Any]:
    begin = f"===BEGIN_{label}==="
    end = f"===END_{label}==="
    start = text.find(begin)
    if start == -1:
        raise RuntimeError(f"Could not find {begin} in remote output")
    start += len(begin)
    stop = text.find(end, start)
    if stop == -1:
        raise RuntimeError(f"Could not find {end} in remote output")
    return json.loads(text[start:stop].strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run remote SAE projection for the residual correctness probe and save local artifacts."
    )
    parser.add_argument("--probe-npz", type=Path, default=Path("artifacts/runs/residual_correctness_probe_direction.npz"))
    parser.add_argument("--alignment-json", type=Path, default=Path("artifacts/runs/probe_feature_alignment.json"))
    parser.add_argument("--sparsity-json", type=Path, default=Path("artifacts/runs/probe_sparsity_diagnostics.json"))
    parser.add_argument("--gpu", default="A100-80GB")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260424)
    args = parser.parse_args()

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/modal_gpu.py",
        "--gpu",
        args.gpu,
        "--timeout",
        str(args.timeout),
        "--sync-extra",
        "interp",
        "--",
        "python",
        "scripts/remote_probe_sae_projection.py",
        "--probe-npz",
        str(args.probe_npz),
        "--n-random",
        str(args.n_random),
        "--seed",
        str(args.seed),
    ]
    result = subprocess.run(cmd, cwd=str(Path.cwd()), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(result.returncode)

    remote_output = result.stdout + "\n" + result.stderr
    alignment = _extract_json_section(remote_output, "PROBE_FEATURE_ALIGNMENT_JSON")
    sparsity = _extract_json_section(remote_output, "PROBE_SPARSITY_DIAGNOSTICS_JSON")
    write_json(args.alignment_json, alignment)
    write_json(args.sparsity_json, sparsity)
    print(f"Wrote probe-feature alignment to {args.alignment_json}")
    print(f"Wrote probe sparsity diagnostics to {args.sparsity_json}")


if __name__ == "__main__":
    main()
