"""Entry point for ``python -m math_discovery``.

Provides a unified CLI that dispatches to subcommands:

    python -m math_discovery generate   --dataset D0 --count 24
    python -m math_discovery features   --input output/surfaces/D0.json
    python -m math_discovery conjecture --dataset D0 --n-steps 5
    python -m math_discovery train      --n-episodes 24
    python -m math_discovery evaluate   --from-training
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        print(
            "Usage: python -m math_discovery <command> [options]\n\n"
            "Commands:\n"
            "  generate    Generate triangulated surface datasets\n"
            "  features    Extract chain-complex features from surfaces\n"
            "  conjecture  Run the conjecturing agent\n"
            "  train       Train the multi-agent system\n"
            "  evaluate    Run the 4-variant ablation evaluation\n"
            "\nAdd --help to any command for details.\n"
            "\nQuick start:\n"
            "  python -m math_discovery generate --dataset D0 --count 24 --output output/surfaces/D0.json\n"
            "  python -m math_discovery train --n-episodes 24 --json\n"
            "  python -m math_discovery evaluate --from-training --json\n"
        )
        sys.exit(0)

    command = sys.argv[1]
    sys.argv = [f"math_discovery {command}"] + sys.argv[2:]

    if command == "generate":
        from math_discovery.surface_data_gen import main as run
    elif command == "features":
        from math_discovery.feature_extractor import main as run
    elif command == "conjecture":
        from math_discovery.conjecturing_agent import main as run
    elif command == "train":
        from math_discovery.run_training import main as run
    elif command == "evaluate":
        from math_discovery.evaluate import main as run
    else:
        print(f"Unknown command: {command}")
        print("Run 'python -m math_discovery --help' for available commands.")
        sys.exit(1)

    run()


if __name__ == "__main__":
    main()
