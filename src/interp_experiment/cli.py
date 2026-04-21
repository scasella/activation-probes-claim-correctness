from __future__ import annotations

import argparse

from .config import load_all_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Utility entrypoint for the interpretability study scaffold.")
    parser.add_argument("--show-configs", action="store_true", help="Print loaded YAML configs and exit.")
    args = parser.parse_args()
    if args.show_configs:
        for name, payload in load_all_configs().items():
            print(f"[{name}]")
            print(payload)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
