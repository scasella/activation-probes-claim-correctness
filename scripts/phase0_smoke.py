from __future__ import annotations

import argparse
import os
from pathlib import Path

from interp_experiment.config import load_all_configs
from interp_experiment.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the phase-0 environment and gating smoke checks.")
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/runs/phase0_smoke.json"))
    parser.add_argument("--require-hf-token", action="store_true", help="Fail if HF_TOKEN is missing.")
    args = parser.parse_args()

    configs = load_all_configs()
    has_hf_token = bool(os.environ.get("HF_TOKEN"))
    has_modal = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
    checks = {
        "configs_loaded": sorted(configs),
        "hf_token_present": has_hf_token,
        "modal_credentials_present": has_modal,
        "phase0_ready_for_live_model": has_hf_token and has_modal,
        "notes": [
            "Live model smoke still depends on gated Meta model access.",
            "Goodfire SAE loading is checked at runtime when SAE extras are installed.",
        ],
    }
    write_json(args.output_json, checks)
    if args.require_hf_token and not has_hf_token:
        raise SystemExit("HF_TOKEN missing; phase-0 live-model smoke is not ready.")
    print(f"Wrote smoke summary to {args.output_json}")


if __name__ == "__main__":
    main()
