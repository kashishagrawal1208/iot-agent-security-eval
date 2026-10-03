"""
Phase 7+: final metric set (Tier A + Tier B).

Reads the three result CSVs already produced by earlier phases --
adversarial_results.csv, clean_evaluation.csv, clean_metrics_summary.csv --
and computes every metric that needs no new API calls and no code changes
to the agents/runners themselves. This is pure aggregation over data you
already collected.

Produces five output tables in results/tables/:
  final_table1_classification.csv   -- accuracy/precision/recall/F1/FNR/FPR
                                        + Wilson 95% CI on accuracy
  final_table2_adversarial.csv      -- ASR, accuracy drop, decision flip
                                        rate, by agent x category, no defense
  final_table3_defense.csv          -- ASR by agent x defense, with
                                        absolute/relative reduction vs "none"
  final_table4_behaviour.csv        -- tool reliance rate, tool-trust
                                        correlation (error propagation),
                                        parse/format compliance rate
  final_stats_tests.csv             -- McNemar's test (single_shot vs react,
                                        paired clean-data correctness)

Also prints a short keyword-based rationale-coding summary (Tier B,
qualitative -- not saved as a table, since it's meant to point you at
specific rows to quote in the discussion section, not a number to report).

Everything here respects the dataset tag -- "real" by default -- so
synthetic rows from earlier phases never silently blend into these
numbers (the same class of bug fixed in adversarial_runner.summarize()).

Run:
    python -m src.evaluation.final_metrics
    python -m src.evaluation.final_metrics --dataset synthetic   # sanity check only
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd

ADV_PATH = "results/tables/adversarial_results.csv"
CLEAN_EVAL_PATH = "results/tables/clean_evaluation.csv"
CLEAN_SUMMARY_PATH = "results/tables/clean_metrics_summary.csv"
OUT_DIR = Path("results/tables")

SHARED_CATEGORIES = ["direct_injection", "context_flooding"]
TOOL_CATEGORIES = ["tool_poisoning", "compound_poisoning"]

TRUST_KEYWORDS = [
    "trust", "allowlist", "allow-list", "verified", "soc", "safe",
    "benign", "false positive", "pre-approved", "cleared",
]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_adv(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(ADV_PATH, keep_default_na=False)
    for col in ["clean_correct", "attack_success", "clean_tool_called", "attacked_tool_called"]:
        if col in df.columns:
            df[col] = df[col].astype(str) == "True"
    return df[df["dataset"] == dataset].copy()


def _load_clean_eval(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(CLEAN_EVAL_PATH, keep_default_na=False)
    return df[df["dataset"] == dataset].copy()


def _load_clean_summary(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(CLEAN_SUMMARY_PATH, keep_default_na=False)
    return df[df["dataset"] == dataset].copy()


# ---------------------------------------------------------------------------
# Stats helpers (no extra dependency beyond scipy, already present via sklearn)
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a proportion. Returns (low, high).
    Chosen over a normal approximation because it stays well-behaved for
    the small sample sizes this project actually has (some n < 10)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar_test(table_2x2: list) -> dict:
    """Manual McNemar's test (no statsmodels dependency) with continuity
    correction. table_2x2 = [[both_correct, a_only], [b_only, both_wrong]]
    -- we only need the two discordant cells, b and c."""
    from scipy.stats import chi2

    b = table_2x2[0][1]  # single_shot correct, react wrong
    c = table_2x2[1][0]  # single_shot wrong, react correct
    n_discordant = b + c
    if n_discordant == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "note": "no discordant pairs"}
    statistic = (abs(b - c) - 1) ** 2 / n_discordant  # Yates' continuity correction
    p_value = 1 - chi2.cdf(statistic, df=1)
    return {"b": b, "c": c, "statistic": statistic, "p_value": p_value, "note": ""}


# ---------------------------------------------------------------------------
# Table 1: Classification Performance (clean data)
# ---------------------------------------------------------------------------

def table1_classification_performance(dataset: str) -> pd.DataFrame:
    summary = _load_clean_summary(dataset)
    if summary.empty:
        print(f"No clean_metrics_summary rows for dataset={dataset}")
        return pd.DataFrame()

    rows = []
    for _, r in summary.iterrows():
        tn, fp, fn, tp = r["tn"], r["fp"], r["fn"], r["tp"]
        fnr = fn / (fn + tp) if (fn + tp) else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        n = int(r["n"])
        correct = int(round(r["accuracy"] * n))
        ci_low, ci_high = wilson_ci(correct, n)
        rows.append({
            "agent": r["agent"], "n": n,
            "accuracy": r["accuracy"], "accuracy_ci_low": ci_low, "accuracy_ci_high": ci_high,
            "precision_attack": r["precision_attack"], "recall_attack": r["recall_attack"],
            "f1_attack": r["f1_attack"], "FNR": fnr, "FPR": fpr,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 2: Adversarial Robustness (attacked data, no defense, by category)
# ---------------------------------------------------------------------------

def table2_adversarial_robustness(dataset: str, defense: str = "none") -> pd.DataFrame:
    df = _load_adv(dataset)
    df = df[df["defense"] == defense]
    if df.empty:
        print(f"No adversarial rows for dataset={dataset}, defense={defense}")
        return pd.DataFrame()

    rows = []
    for (agent, category), g in df.groupby(["agent", "category"]):
        total = len(g)
        clean_correct_n = int(g["clean_correct"].sum())
        successes = int(g["attack_success"].sum())
        asr = successes / clean_correct_n if clean_correct_n else float("nan")

        clean_acc = (g["true_label"] == g["clean_pred"]).mean()
        attacked_acc = (g["true_label"] == g["attacked_pred"]).mean()
        drop = clean_acc - attacked_acc

        flips = (g["clean_pred"] != g["attacked_pred"]).mean()

        ci_low, ci_high = wilson_ci(successes, clean_correct_n) if clean_correct_n else (float("nan"), float("nan"))

        rows.append({
            "agent": agent, "category": category, "total_cases": total,
            "clean_correct_n": clean_correct_n, "ASR": asr,
            "ASR_ci_low": ci_low, "ASR_ci_high": ci_high,
            "accuracy_clean": clean_acc, "accuracy_attacked": attacked_acc,
            "accuracy_drop": drop, "decision_flip_rate": flips,
        })
    return pd.DataFrame(rows).sort_values(["agent", "category"])


# ---------------------------------------------------------------------------
# Table 3: Defense Effectiveness
# ---------------------------------------------------------------------------

def table3_defense_effectiveness(dataset: str) -> pd.DataFrame:
    df = _load_adv(dataset)
    df_shared = df[df["category"].isin(SHARED_CATEGORIES)]
    if df_shared.empty:
        print(f"No shared-category adversarial rows for dataset={dataset}")
        return pd.DataFrame()

    rows = []
    baseline_asr = {}
    for (agent, defense), g in df_shared.groupby(["agent", "defense"]):
        clean_correct_n = int(g["clean_correct"].sum())
        successes = int(g["attack_success"].sum())
        asr = successes / clean_correct_n if clean_correct_n else float("nan")
        rows.append({"agent": agent, "defense": defense, "clean_correct_n": clean_correct_n, "ASR": asr})
        if defense == "none":
            baseline_asr[agent] = asr

    out = pd.DataFrame(rows)
    out["ASR_reduction_abs"] = out.apply(
        lambda r: (baseline_asr.get(r["agent"], float("nan")) - r["ASR"])
        if r["defense"] != "none" and not pd.isna(baseline_asr.get(r["agent"]))
        else float("nan"),
        axis=1,
    )
    out["ASR_reduction_rel"] = out.apply(
        lambda r: (r["ASR_reduction_abs"] / baseline_asr[r["agent"]])
        if r["defense"] != "none" and baseline_asr.get(r["agent"], 0) not in (0, float("nan")) and not pd.isna(r["ASR_reduction_abs"])
        else float("nan"),
        axis=1,
    )
    return out.sort_values(["agent", "defense"])


# ---------------------------------------------------------------------------
# Table 4: Agent / Reasoning Behaviour
# ---------------------------------------------------------------------------

def table4_agent_behaviour(dataset: str) -> pd.DataFrame:
    df = _load_adv(dataset)
    rows = []

    # Tool reliance rate -- react only, clean vs attacked, per defense
    react = df[df["agent"] == "react"]
    for defense, g in react.groupby("defense"):
        rows.append({
            "metric": "tool_reliance_rate_clean", "scope": f"react / defense={defense}",
            "value": g["clean_tool_called"].mean(), "n": len(g),
        })
        rows.append({
            "metric": "tool_reliance_rate_attacked", "scope": f"react / defense={defense}",
            "value": g["attacked_tool_called"].mean(), "n": len(g),
        })

    # Error propagation / tool-trust correlation -- react, tool-related categories, no defense
    tool_df = react[(react["category"].isin(TOOL_CATEGORIES)) & (react["defense"] == "none")]
    for called, g in tool_df.groupby("attacked_tool_called"):
        rows.append({
            "metric": "attack_success_given_tool_called" if called else "attack_success_given_tool_not_called",
            "scope": "react / tool_poisoning+compound_poisoning / defense=none",
            "value": g["attack_success"].mean(), "n": len(g),
        })

    # Parse / format compliance rate -- both agents, across clean+attacked predictions
    clean_eval = _load_clean_eval(dataset)
    for agent in df["agent"].unique():
        preds = pd.concat([
            df[df["agent"] == agent]["clean_pred"],
            df[df["agent"] == agent]["attacked_pred"],
            clean_eval[clean_eval["agent"] == agent]["predicted_label"],
        ])
        if len(preds):
            compliance = (preds != "UNPARSEABLE").mean()
            rows.append({"metric": "parse_compliance_rate", "scope": agent, "value": compliance, "n": len(preds)})

    return pd.DataFrame(rows)


def rationale_coding_report(dataset: str) -> None:
    """Prints (does not save) which attack-success rows explicitly mention
    trust-related language in their rationale -- pointers for the write-up's
    discussion section, not a headline number."""
    df = _load_adv(dataset)
    hits = df[(df["attack_success"]) & (df["category"].isin(TOOL_CATEGORIES))]
    if hits.empty:
        print("\nNo tool-poisoning/compound attack successes to code.")
        return

    pattern = re.compile("|".join(TRUST_KEYWORDS), re.IGNORECASE)
    matched = hits[hits["attacked_rationale"].str.contains(pattern, na=False)]

    print(f"\n=== Rationale coding: attack successes citing trust-language ({dataset}) ===")
    print(f"{len(matched)}/{len(hits)} successful tool/compound attacks have a rationale "
          f"explicitly mentioning trust-related terms.\n")
    for _, r in matched.iterrows():
        print(f"  [{r['case_id']}/{r['agent']}] \"{r['attacked_rationale']}\"")


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------

def stats_tests(dataset: str) -> pd.DataFrame:
    clean = _load_clean_eval(dataset)
    pivot = clean.pivot_table(index="source_index", columns="agent", values="predicted_label", aggfunc="first")
    true = clean.drop_duplicates("source_index").set_index("source_index")["true_label"]

    if "single_shot" not in pivot.columns or "react" not in pivot.columns:
        print("Need both single_shot and react rows in clean_evaluation.csv for McNemar's test")
        return pd.DataFrame()

    paired = pivot.join(true).dropna(subset=["single_shot", "react"])
    ss_correct = paired["single_shot"] == paired["true_label"]
    re_correct = paired["react"] == paired["true_label"]

    both_correct = int((ss_correct & re_correct).sum())
    ss_only = int((ss_correct & ~re_correct).sum())
    re_only = int((~ss_correct & re_correct).sum())
    both_wrong = int((~ss_correct & ~re_correct).sum())

    result = mcnemar_test([[both_correct, ss_only], [re_only, both_wrong]])
    return pd.DataFrame([{
        "test": "McNemar (single_shot vs react, paired clean accuracy)",
        "n_pairs": len(paired), "both_correct": both_correct,
        "single_shot_only_correct": ss_only, "react_only_correct": re_only,
        "both_wrong": both_wrong, "statistic": result["statistic"],
        "p_value": result["p_value"], "significant_at_0.05": result["p_value"] < 0.05,
    }])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tier A + B final metrics, computed from existing result CSVs only.")
    parser.add_argument("--dataset", default="real")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t1 = table1_classification_performance(args.dataset)
    if not t1.empty:
        print(f"\n=== Table 1: Classification Performance ({args.dataset}) ===\n")
        print(t1.to_string(index=False))
        t1.to_csv(OUT_DIR / "final_table1_classification.csv", index=False)

    t2 = table2_adversarial_robustness(args.dataset)
    if not t2.empty:
        print(f"\n=== Table 2: Adversarial Robustness ({args.dataset}, no defense) ===\n")
        print(t2.to_string(index=False))
        t2.to_csv(OUT_DIR / "final_table2_adversarial.csv", index=False)

    t3 = table3_defense_effectiveness(args.dataset)
    if not t3.empty:
        print(f"\n=== Table 3: Defense Effectiveness ({args.dataset}, shared categories) ===\n")
        print(t3.to_string(index=False))
        t3.to_csv(OUT_DIR / "final_table3_defense.csv", index=False)

    t4 = table4_agent_behaviour(args.dataset)
    if not t4.empty:
        print(f"\n=== Table 4: Agent/Reasoning Behaviour ({args.dataset}) ===\n")
        print(t4.to_string(index=False))
        t4.to_csv(OUT_DIR / "final_table4_behaviour.csv", index=False)

    rationale_coding_report(args.dataset)

    t5 = stats_tests(args.dataset)
    if not t5.empty:
        print(f"\n=== Statistical tests ({args.dataset}) ===\n")
        print(t5.to_string(index=False))
        t5.to_csv(OUT_DIR / "final_stats_tests.csv", index=False)

    print(f"\nAll tables saved to {OUT_DIR}/final_table*.csv and final_stats_tests.csv")


if __name__ == "__main__":
    main()