"""
Phase 2: Data pipeline for the IoT flow dataset.

What this file does, in plain terms:
  1. Loads the raw dataset CSV (the original file is never modified).
  2. Prints basic info about it so we can see what we're working with.
  3. Cleans up column names so they're predictable (lowercase, no spaces).
  4. Finds the column that holds the attack label.
  5. Drops rows where the label itself is missing (can't call those BENIGN or ATTACK).
  6. Turns the label into exactly two values: BENIGN or ATTACK.
  7. Drops columns that are empty, constant, or look like random IDs
     (they can't help any classifier learn anything).
  8. Fills in missing/invalid values in a simple, documented way.
  9. Splits the data into train/validation/test sets, keeping the same
     BENIGN/ATTACK ratio in each split (this is called "stratified" splitting).
  10. Saves everything to data/processed/, plus a small stats file.

What this file deliberately does NOT do (that's for later phases):
  - No LLM prompts or agents.
  - No adversarial examples.
  - No defenses.
  - No mock reputation tool.
  - No accuracy/F1/attack-success-rate metrics (only basic dataset stats).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Read config.yaml so we don't hardcode settings in this file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_csv(path: str) -> pd.DataFrame:
    """
    Load the raw dataset CSV.

    This never writes back to `path` — the raw file stays exactly as
    downloaded. low_memory=False avoids pandas guessing column types
    chunk-by-chunk, which can cause inconsistent dtypes on large CSVs.
    """
    return pd.read_csv(path, low_memory=False)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make column names predictable: lowercase, trimmed, spaces -> underscores.

    Example: "Attack_type " -> "attack_type"
    """
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def detect_label_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """
    Find which column holds the attack label by checking a list of
    known possible names (from config.yaml) in order.
    """
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not find a label column. Looked for: "
        f"{candidates}. Available columns: {list(df.columns)}"
    )


def drop_missing_label_rows(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, int]:
    """
    Drop rows where the label column itself is missing (NaN/blank).

    A row with no label isn't BENIGN or ATTACK -- it's "we don't know."
    Without this step, those rows would fall through the BENIGN/ATTACK
    string match in map_labels() and get silently counted as ATTACK,
    which would quietly skew the class distribution.

    Returns the cleaned dataframe and how many rows were dropped, so
    this is visible in the printed output and in dataset_stats.json.
    """
    n_before = len(df)
    df = df[df[label_col].notna()].copy()
    n_dropped = n_before - len(df)
    return df, n_dropped


def map_labels(df: pd.DataFrame, label_col: str, benign_values: list[str]) -> pd.DataFrame:
    """
    Create a new 'label' column with exactly two values: BENIGN or ATTACK.

    Any value in the original label column that matches something in
    `benign_values` (case-insensitive) becomes BENIGN. Everything else
    becomes ATTACK.
    """
    df = df.copy()
    benign_set = {str(v).strip().lower() for v in benign_values}

    def to_binary(value) -> str:
        text = str(value).strip().lower()
        return "BENIGN" if text in benign_set else "ATTACK"

    df["label"] = df[label_col].apply(to_binary)
    return df


def drop_unusable_columns(
    df: pd.DataFrame, id_like_threshold: float = 0.9
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove columns that can't be useful flow features:
      - entirely missing columns,
      - constant columns (same value in every row -> no information),
      - very high-cardinality text columns (e.g. random session/connection
        IDs) that are essentially unique per row and won't generalize.

    Returns the cleaned dataframe and the list of dropped column names,
    so we can show you exactly what was removed and why.
    """
    df = df.copy()
    n_rows = len(df)
    cols_to_drop = []

    for col in df.columns:
        if col == "label":
            continue

        n_missing = int(df[col].isna().sum())
        if n_missing == n_rows:
            cols_to_drop.append(col)
            continue

        n_unique = df[col].nunique(dropna=True)
        if n_unique <= 1:
            cols_to_drop.append(col)
            continue

        # NOTE: on pandas >= 2.x/3.x, CSV text columns can load with a
        # "str" dtype instead of the classic "object" dtype, so we check
        # with is_string_dtype() rather than `df[col].dtype == object`
        # (that comparison silently misses these columns).
        is_text = pd.api.types.is_string_dtype(df[col])
        if is_text and n_rows > 0 and (n_unique / n_rows) > id_like_threshold:
            cols_to_drop.append(col)

    return df.drop(columns=cols_to_drop), cols_to_drop


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill obvious missing/invalid values:
      - numeric columns: fill with that column's median, and replace
        +/- infinity (which can show up in rate-based features) with
        the same median,
      - text columns: fill with the literal string "unknown".

    This is intentionally simple and documented rather than clever —
    Phase 2 is about having a clean, reproducible baseline dataset.
    """
    df = df.copy()
    for col in df.columns:
        if col == "label":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            median = df[col].median()
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(median)
        else:
            df[col] = df[col].fillna("unknown")
    return df


def split_dataset(
    df: pd.DataFrame, seed: int, val_size: float = 0.15, test_size: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train/validation/test sets.

    `stratify=df["label"]` means each split keeps roughly the same
    BENIGN vs ATTACK ratio as the full dataset — otherwise a random
    split could accidentally put almost all attacks in one split.
    """
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    # val_size is expressed as a fraction of the FULL dataset, so we
    # need to convert it to a fraction of what's left (train_val).
    val_relative = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_relative, random_state=seed, stratify=train_val["label"]
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def class_distribution(df: pd.DataFrame) -> dict:
    """Count how many BENIGN vs ATTACK rows are in a dataframe."""
    counts = df["label"].value_counts().to_dict()
    return {"BENIGN": int(counts.get("BENIGN", 0)), "ATTACK": int(counts.get("ATTACK", 0))}


def run_pipeline(
    input_path: str,
    output_dir: str = "data/processed",
    config_path: str = "config.yaml",
) -> dict:
    """Run the full Phase 2 pipeline end to end and return summary stats."""
    config = load_config(config_path)
    seed = config["experiment"]["random_seed"]
    dataset_cfg = config["dataset"]

    print(f"Loading raw data from: {input_path}")
    df_raw = load_raw_csv(input_path)
    print(f"Raw shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    print(f"Raw columns: {list(df_raw.columns)}")

    df = normalize_columns(df_raw)

    label_col = detect_label_column(df, dataset_cfg["label_column_candidates"])
    print(f"\nDetected label column: '{label_col}'")
    unique_vals = sorted(str(v) for v in df[label_col].dropna().unique())
    print(f"Original label values ({len(unique_vals)}): {unique_vals}")

    df, n_dropped_missing_label = drop_missing_label_rows(df, label_col)
    if n_dropped_missing_label:
        print(
            f"\nDropped {n_dropped_missing_label} rows with a missing/blank "
            f"'{label_col}' value (can't be labeled BENIGN or ATTACK)."
        )

    df = map_labels(df, label_col, dataset_cfg["benign_values"])

    df, dropped_cols = drop_unusable_columns(df)
    print(f"\nDropped {len(dropped_cols)} unusable columns: {dropped_cols}")
    print(f"Remaining feature columns ({df.shape[1] - 1}): {[c for c in df.columns if c != 'label']}")

    df = handle_missing_values(df)

    dist = class_distribution(df)
    print(f"\nClass distribution (full dataset) -> BENIGN: {dist['BENIGN']}, ATTACK: {dist['ATTACK']}")

    split_cfg = dataset_cfg["split"]
    train, val, test = split_dataset(
        df, seed=seed, val_size=split_cfg["val_size"], test_size=split_cfg["test_size"]
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "processed_full.csv", index=False)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    stats = {
        "input_file": str(input_path),
        "raw_rows": int(df_raw.shape[0]),
        "raw_columns": int(df_raw.shape[1]),
        "rows_dropped_missing_label": int(n_dropped_missing_label),
        "processed_rows": int(df.shape[0]),
        "processed_columns": int(df.shape[1]),
        "label_column_source": label_col,
        "dropped_columns": dropped_cols,
        "class_distribution_full": dist,
        "class_distribution_train": class_distribution(train),
        "class_distribution_val": class_distribution(val),
        "class_distribution_test": class_distribution(test),
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "random_seed": seed,
    }
    with open(out_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    results_dir = Path(config["paths"]["results_tables"])
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dist]).to_csv(results_dir / "class_distribution.csv", index=False)

    print(f"\nProcessed data written to: {out_dir}/")
    print("  - processed_full.csv, train.csv, val.csv, test.csv")
    print(f"  - dataset_stats.json")
    print(f"Class distribution table written to: {results_dir / 'class_distribution.csv'}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: preprocess the raw IoT dataset into a clean, split, reproducible form."
    )
    parser.add_argument("--input", required=True, help="Path to the raw dataset CSV")
    parser.add_argument("--output-dir", default="data/processed", help="Where to write processed files")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run_pipeline(args.input, args.output_dir, args.config)


if __name__ == "__main__":
    main()