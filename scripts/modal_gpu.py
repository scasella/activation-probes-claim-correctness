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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a command inside a Modal GPU sandbox.")
    parser.add_argument("--gpu", default="A100-80GB")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in minutes.")
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
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("uv")
        .add_local_dir(str(root), remote_path="/app")
    )
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
    if sandbox.returncode not in (0, None):
        raise SystemExit(sandbox.returncode)


if __name__ == "__main__":
    main()
