"""Quick environment verification for Unsupervised folder dependencies.

Run from repository root:
    python Unsupervised/scripts/verify_unsupervised_setup.py
"""

from __future__ import annotations

import importlib
import sys

REQUIRED = [
    "pandas",
    "numpy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "scipy",
]


def main() -> int:
    print("Checking Unsupervised dependencies...")
    failures: list[str] = []

    for pkg in REQUIRED:
        try:
            importlib.import_module(pkg)
            print(f"  [OK] {pkg}")
        except Exception as exc:  # pragma: no cover
            failures.append(pkg)
            print(f"  [MISSING] {pkg} ({exc})")

    if failures:
        print("\nMissing packages detected.")
        print("Install with: python -m pip install -r Unsupervised/requirements-unsupervised.txt")
        return 1

    print("\nAll required Unsupervised packages are installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
