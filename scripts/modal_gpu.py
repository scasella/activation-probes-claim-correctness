from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from interp_experiment.env import load_repo_env


def _load_modal():
    try:
        import modal
    except ImportError as exc:
        raise RuntimeError("modal is required. Install with `uv sync --extra modal`.") from exc
    return modal


def _ignore_mutable_outputs(path: Path) -> bool:
    if path.name == ".DS_Store":
        return True
    if path.name in {"remote.log", "summary.json", "targets.jsonl", "features.jsonl", "features.npz"}:
        parts = set(path.parts)
        return "artifacts" in parts and "runs" in parts
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a command inside a Modal GPU sandbox.")
    parser.add_argument("--gpu", default="A100-80GB")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in minutes.")
    parser.add_argument(
        "--sync-extra",
        action="append",
        default=[],
        help="Optional pyproject extra to install into the sandbox image before running the command.",
    )
    parser.add_argument(
        "--sync-group",
        action="append",
        default=[],
        help="Optional dependency group to install into the sandbox image before running the command.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("Usage: python scripts/modal_gpu.py --gpu=A100-80GB -- python -m pytest")

    load_repo_env()
    modal = _load_modal()
    root = Path(__file__).resolve().parents[1]
    forwarded_env = {
        key: os.environ[key]
        for key in ("HF_TOKEN", "OPENAI_API_KEY", "PRIME_API_KEY", "TINKER_API_KEY")
        if os.environ.get(key)
    }
    forwarded_env["PYTHONPATH"] = "/app/src"
    image = modal.Image.debian_slim(python_version="3.11").pip_install("uv")
    if args.sync_extra or args.sync_group:
        extra_set = set(args.sync_extra)
        if extra_set and extra_set.issubset({"inference", "interp"}) and not args.sync_group:
            image = image.uv_pip_install(
                "torch==2.6.0",
                extra_index_url="https://download.pytorch.org/whl/cu124",
            )
            runtime_packages = [
                "numpy>=1.26",
                "PyYAML>=6.0",
                "scipy>=1.13",
                "scikit-learn>=1.5",
                "accelerate>=1.2",
                "transformers>=4.48",
            ]
            if "interp" in extra_set:
                runtime_packages.extend(
                    [
                        "sae-lens>=6.0",
                        "transformer-lens>=2.11",
                    ]
                )
            image = image.uv_pip_install(*runtime_packages)
        else:
            image = image.uv_sync(
                uv_project_dir=str(root),
                extras=args.sync_extra or None,
                groups=args.sync_group or None,
                frozen=False,
            )
    image = image.add_local_dir(str(root), remote_path="/app", ignore=_ignore_mutable_outputs)
    app = modal.App.lookup("interp-experiment", create_if_missing=True)
    sandbox = modal.Sandbox.create(
        *command,
        image=image,
        env=forwarded_env,
        gpu=args.gpu,
        timeout=args.timeout * 60,
        workdir="/app",
        app=app,
    )
    for line in sandbox.stdout:
        sys.stdout.write(line)
    sandbox.wait(raise_on_termination=False)
    for line in sandbox.stderr:
        sys.stderr.write(line)
    if sandbox.returncode not in (0, None):
        raise SystemExit(sandbox.returncode)


if __name__ == "__main__":
    main()
