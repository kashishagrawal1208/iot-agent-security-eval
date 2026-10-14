"""
Plots for the clean-data sample-size progression.

Reads the two CSVs written by sample_size_progression.py (no API calls) and
writes three PNGs to results/plots/:
  progression_accuracy_f1.png   accuracy (95% Wilson band) and F1 vs n
  progression_recall_fpr.png    ATTACK recall and false-positive rate (95% Wilson bands) vs n
  progression_mcnemar.png       McNemar p-value (single_shot vs react) vs n

Reading the plots: the subsets are nested, so neighbouring points share most of
their flows, and n changes only how precisely performance is measured -- the
agents are not trained, so this is not a learning curve.

Run:
    python -m src.evaluation.plot_progression
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METRICS = Path("results/tables/sample_size_progression_clean.csv")
PAIRED = Path("results/tables/sample_size_progression_mcnemar.csv")
PLOTS = Path("results/plots")
COLORS = {"react": "tab:blue", "single_shot": "tab:orange"}
LABELS = {"react": "ReAct", "single_shot": "Single-Shot"}
FOOTNOTE = ("Nested balanced subsets of one evaluated sample (adjacent points share data). "
            "Bands = 95% Wilson intervals. n changes measurement precision, not the agents.")


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    c = (k / n + z * z / (2 * n)) / d
    m = z / d * math.sqrt((k / n) * (1 - k / n) / n + z * z / (4 * n * n))
    return (max(0.0, c - m), min(1.0, c + m))


def _line(ax, df, col, lo=None, hi=None, pct=True):
    for agent, g in df.groupby("agent"):
        g = g.sort_values("n")
        y = g[col] * (100 if pct else 1)
        ax.plot(g["n"], y, marker="o", color=COLORS[agent], label=LABELS[agent])
        if lo is not None:
            ax.fill_between(g["n"], lo(g) * 100, hi(g) * 100, color=COLORS[agent], alpha=0.15)
        for x, v in zip(g["n"], y):
            offset = -14 if (agent == "react" and v >= 3) else 7  # ReAct labels below, Single-Shot above; near-zero values go above the point
            ax.annotate(f"{v:.0f}", (x, v), textcoords="offset points", xytext=(0, offset), ha="center",
                        fontsize=8, color=COLORS[agent])


def _style(ax, title, ylabel, sizes):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Sample size n (flows, 50% ATTACK / 50% BENIGN)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sizes)
    ax.grid(alpha=0.3)
    ax.legend()


def main() -> None:
    m = pd.read_csv(METRICS)
    p = pd.read_csv(PAIRED)
    sizes = sorted(m["n"].unique())
    PLOTS.mkdir(parents=True, exist_ok=True)

    m["recall_lo"], m["recall_hi"] = zip(*[wilson(r.tp, r.tp + r.fn) for r in m.itertuples()])
    m["fpr_lo"], m["fpr_hi"] = zip(*[wilson(r.fp, r.fp + r.tn) for r in m.itertuples()])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    _line(axes[0], m, "accuracy", lambda g: g["acc_ci_low"], lambda g: g["acc_ci_high"])
    _style(axes[0], "Accuracy vs sample size", "Accuracy (%)", sizes)
    _line(axes[1], m, "f1")
    _style(axes[1], "F1 (ATTACK) vs sample size", "F1 (%)", sizes)
    axes[0].set_ylim(40, 95)
    axes[1].set_ylim(20, 85)
    fig.text(0.5, 0.005, FOOTNOTE, ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(PLOTS / "progression_accuracy_f1.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    _line(axes[0], m, "recall", lambda g: g["recall_lo"], lambda g: g["recall_hi"])
    _style(axes[0], "Recall on ATTACK flows vs sample size", "Recall (%)", sizes)
    _line(axes[1], m, "FPR", lambda g: g["fpr_lo"], lambda g: g["fpr_hi"])
    _style(axes[1], "False-positive rate vs sample size", "FPR (%)", sizes)
    axes[0].set_ylim(0, 80)
    axes[1].set_ylim(0, 25)
    fig.text(0.5, 0.005, FOOTNOTE, ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(PLOTS / "progression_recall_fpr.png", dpi=150)
    plt.close(fig)

    p = p.sort_values("n")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(p["n"], p["p_value"], marker="o", color="tab:green")
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="p = 0.05")
    for r in p.itertuples():
        net = r.single_shot_only_correct - r.react_only_correct
        late = r.n >= 150  # late points: label below-left so it does not sit on the descending line
        ax.annotate(f"p={r.p_value:.3f}\nnet +{net}", (r.n, r.p_value), textcoords="offset points",
                    xytext=(-12, -28) if late else (0, 10), ha="right" if late else "center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlim(45, 195)
    ax.set_ylim(0.002, 6)
    _style(ax, "McNemar p-value, Single-Shot vs ReAct, vs sample size", "p-value (log scale)", sizes)
    fig.text(0.5, 0.005, "net +k = flows only Single-Shot got right minus flows only ReAct got right.\nNested samples: p-values at different n are not independent.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(PLOTS / "progression_mcnemar.png", dpi=150)
    plt.close(fig)
    print("Saved 3 plots to", PLOTS)


if __name__ == "__main__":
    main()