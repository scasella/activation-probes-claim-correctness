from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from interp_experiment.config import load_all_configs
from interp_experiment.env import load_repo_env
from interp_experiment.io import write_json


def _check_hf_file_access(repo_id: str, filename: str, token: str | None) -> dict[str, object]:
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = Request(url, headers=headers, method="HEAD")
    try:
        with urlopen(request, timeout=20) as response:
            return {"ok": True, "status": response.status, "repo_id": repo_id, "filename": filename}
    except HTTPError as exc:
        return {"ok": False, "status": exc.code, "repo_id": repo_id, "filename": filename, "error": "http_error"}
    except URLError as exc:
        return {"ok": False, "status": None, "repo_id": repo_id, "filename": filename, "error": str(exc.reason)}


def _check_modal_auth() -> dict[str, object]:
    try:
        result = subprocess.run(
            ["modal", "app", "list"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "returncode": None, "error": str(exc)}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stderr_tail": result.stderr.strip()[-300:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the phase-0 environment and gating smoke checks.")
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/runs/phase0_smoke.json"))
    parser.add_argument("--require-hf-token", action="store_true", help="Fail if HF_TOKEN is missing.")
    parser.add_argument("--live-access", action="store_true", help="Check live HF file access and Modal auth.")
    args = parser.parse_args()

    loaded_env = load_repo_env()
    configs = load_all_configs()
    has_hf_token = bool(os.environ.get("HF_TOKEN"))
    has_modal = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
    live_checks: dict[str, object] = {}
    if args.live_access:
        token = os.environ.get("HF_TOKEN")
        live_checks["llama_model_access"] = _check_hf_file_access(
            "meta-llama/Llama-3.1-8B-Instruct",
            "config.json",
            token,
        )
        live_checks["goodfire_sae_access"] = _check_hf_file_access(
            "Goodfire/Llama-3.1-8B-Instruct-SAE-l19",
            "config.yaml",
            token,
        )
        live_checks["modal_auth"] = _check_modal_auth()
        live_ready = all(item.get("ok") for item in live_checks.values())
    else:
        live_ready = has_hf_token and has_modal
    checks = {
        "configs_loaded": sorted(configs),
        "env_keys_loaded": sorted(loaded_env),
        "hf_token_present": has_hf_token,
        "modal_credentials_present": has_modal,
        "phase0_ready_for_live_model": live_ready,
        "live_checks": live_checks,
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
