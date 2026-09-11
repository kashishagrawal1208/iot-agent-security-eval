"""
Phase 2 CLI entry point.

Run this from the project root to turn the raw IoT dataset CSV into a
cleaned, labeled, split, reproducible dataset in data/processed/.

Example:
    python run_preprocessing.py --input data/raw/edge_iiotset.csv

Run with --help to see all options:
    python run_preprocessing.py --help
"""

from src.data.preprocess import main

if __name__ == "__main__":
    main()