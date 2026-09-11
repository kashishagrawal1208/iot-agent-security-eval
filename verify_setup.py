"""
Phase 1 smoke test.

This script does NOT test any agent, dataset, or evaluation logic
(none of that exists yet). It only checks that:
  1. The expected folder structure exists.
  2. config.yaml exists and loads correctly.
  3. The values in config.yaml look sane.

Run with:
    python verify_setup.py
"""

import sys
from pathlib import Path

import yaml

REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "data/adversarial",
    "src/agents",
    "src/tools",
    "src/defenses",
    "src/evaluation",
    "src/data",
    "notebooks",
    "results/tables",
    "results/plots",
]

REQUIRED_INIT_FILES = [
    "src/__init__.py",
    "src/agents/__init__.py",
    "src/tools/__init__.py",
    "src/defenses/__init__.py",
    "src/evaluation/__init__.py",
    "src/data/__init__.py",
]


def check_dirs(root: Path) -> list[str]:
    problems = []
    for rel in REQUIRED_DIRS:
        if not (root / rel).is_dir():
            problems.append(f"Missing directory: {rel}")
    return problems


def check_init_files(root: Path) -> list[str]:
    problems = []
    for rel in REQUIRED_INIT_FILES:
        if not (root / rel).is_file():
            problems.append(f"Missing __init__.py: {rel}")
    return problems


def check_config(root: Path) -> list[str]:
    problems = []
    config_path = root / "config.yaml"
    if not config_path.is_file():
        return ["Missing config.yaml"]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        return ["config.yaml did not parse into a dictionary"]

    expected_top_keys = ["project", "paths", "experiment"]
    for key in expected_top_keys:
        if key not in config:
            problems.append(f"config.yaml missing top-level key: {key}")

    return problems


def main() -> int:
    root = Path(__file__).resolve().parent

    problems = []
    problems += check_dirs(root)
    problems += check_init_files(root)
    problems += check_config(root)

    if problems:
        print("Phase 1 setup check FAILED:\n")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("Phase 1 setup check PASSED.")
    print("  - All required folders exist.")
    print("  - All required __init__.py files exist.")
    print("  - config.yaml loads correctly and has expected structure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
