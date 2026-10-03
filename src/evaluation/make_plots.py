"""
Phase 7: generates the comparison charts for the final report.

Reads the already-saved result CSVs (no API calls here) and produces three
PNGs in results/plots/:
  1. clean_metrics.png     -- accuracy/precision/recall/F1, single-shot vs ReAct
  2. asr_by_category.png   -- attack success rate by category, no defense
  3. asr_defenses.png      -- before/after ASR for each defense, ReAct and single-shot

Every chart is built only from rows where clean_correct was True (for ASR
charts) and only from dataset="real" by default, so synthetic rows never
silently blend into these numbers -- the same bug fixed in
adversarial_runner.summarize().

Run:
    python -m src.evaluation.make_plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PLOTS_DIR = Path("results/plots")


def plot_clean_metrics(path="results/tables/clean_metrics_summary.csv", dataset="real"):
    df = pd.read_csv(path)
    df = df[df["dataset"] == dataset]
    if df.empty:
        print(f"No clean metrics for dataset={dataset}, skipping clean_metrics.png")
        return

    metrics = ["accuracy", "precision_attack", "recall_attack", "f1_attack"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    agents = df["agent"].tolist()

    x = range(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, agent in enumerate(agents):
        row = df[df["agent"] == agent].iloc[0]
        values = [row[m] for m in metrics]
        offset = (i - (len(agents) - 1) / 2) * width
        bars = ax.bar([xi + offset for xi in x], values, width, label=agent)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"Clean-data metrics (dataset={dataset}, n={df['n'].iloc[0]} per agent)")
    ax.legend()
    fig.tight_layout()
    out = PLOTS_DIR / "clean_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def _asr_table(df: pd.DataFrame, by: list) -> pd.DataFrame:
    g = df.groupby(by).agg(
        clean_correct=("clean_correct", "sum"), successes=("attack_success", "sum")
    ).reset_index()
    g["ASR"] = [s / c if c else None for s, c in zip(g["successes"], g["clean_correct"])]
    return g


def plot_asr_by_category(path="results/tables/adversarial_results.csv", dataset="real", defense="none"):
    df = pd.read_csv(path, keep_default_na=False)
    for col in ["clean_correct", "attack_success"]:
        df[col] = df[col].astype(str) == "True"
    df = df[(df["dataset"] == dataset) & (df["defense"] == defense)]
    if df.empty:
        print(f"No rows for dataset={dataset}, defense={defense}, skipping asr_by_category.png")
        return

    g = _asr_table(df, ["agent", "category"])
    categories = sorted(g["category"].unique())
    agents = sorted(g["agent"].unique())

    x = range(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, agent in enumerate(agents):
        sub = g[g["agent"] == agent].set_index("category").reindex(categories)
        values = [v if pd.notna(v) else 0 for v in sub["ASR"]]
        n_vals = [
            sub.loc[c, "clean_correct"] if c in sub.index and pd.notna(sub.loc[c, "clean_correct"]) else 0
            for c in categories
        ]
        offset = (i - (len(agents) - 1) / 2) * width
        bars = ax.bar([xi + offset for xi in x], values, width, label=agent)
        for b, v, n in zip(bars, values, n_vals):
            label = f"{v:.0%}\n(n={int(n)})" if n else "n/a"
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, label, ha="center", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Attack Success Rate")
    ax.set_title(f"ASR by category, no defense (dataset={dataset})\n(n = cases agent got right when clean -- often small, see table)")
    ax.legend()
    fig.tight_layout()
    out = PLOTS_DIR / "asr_by_category.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_asr_defenses(path="results/tables/adversarial_results.csv", dataset="real"):
    df = pd.read_csv(path, keep_default_na=False)
    for col in ["clean_correct", "attack_success"]:
        df[col] = df[col].astype(str) == "True"
    df = df[df["dataset"] == dataset]
    if df.empty:
        print(f"No rows for dataset={dataset}, skipping asr_defenses.png")
        return

    g = _asr_table(df, ["agent", "defense"])
    defenses = ["none", "evidence_first", "verification_gate"]
    agents = sorted(g["agent"].unique())

    x = range(len(defenses))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, agent in enumerate(agents):
        sub = g[g["agent"] == agent].set_index("defense").reindex(defenses)
        values = [v if pd.notna(v) else None for v in sub["ASR"]]
        n_vals = [sub.loc[d, "clean_correct"] if d in sub.index and pd.notna(sub.loc[d, "clean_correct"]) else 0 for d in defenses]
        plot_values = [v if v is not None else 0 for v in values]
        offset = (i - (len(agents) - 1) / 2) * width
        bars = ax.bar([xi + offset for xi in x], plot_values, width, label=agent)
        for b, v, n in zip(bars, values, n_vals):
            label = f"{v:.0%}\n(n={int(n)})" if v is not None and n else "n/a"
            ax.text(b.get_x() + b.get_width() / 2, (v or 0) + 0.02, label, ha="center", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(defenses)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("Attack Success Rate")
    ax.set_title(f"ASR before/after each defense (dataset={dataset})\n(n = cases agent got right when clean -- often small, see table)")
    ax.legend()
    fig.tight_layout()
    out = PLOTS_DIR / "asr_defenses.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Phase 7: generate comparison charts from saved results.")
    parser.add_argument("--dataset", default="real")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_clean_metrics(dataset=args.dataset)
    plot_asr_by_category(dataset=args.dataset)
    plot_asr_defenses(dataset=args.dataset)


if __name__ == "__main__":
    main()
