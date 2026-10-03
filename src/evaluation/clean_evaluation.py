"""
Phase 7: clean-data evaluation.

Measures how good the agents actually are at the base classification task
-- no attacks, no defenses -- using accuracy/precision/recall/F1, the
metrics that give the adversarial results (Phase 5/6) their meaning. A
100% attack success rate means nothing if the agent's baseline accuracy
was already near zero on that category.

Samples real flows from data/processed/test.csv, EXCLUDING every row
already used in the adversarial corpus (tracked via its source_index
column), so this evaluation stays independent of the adversarial one --
otherwise a case that was hard for the agent would double-count as both
"low clean accuracy" and "high attack success," inflating both findings
from the same underlying rows.

No defenses are applied here -- defenses only make sense where there's an
attack to defend against; this measures the undefended baseline.

Results are saved incrementally, keyed on (source_index, agent, dataset),
so a crash or rate limit is safe to resume from with the same command.

Run:
    python -m src.evaluation.clean_evaluation --n-per-class 5      # cheap trial
    python -m src.evaluation.clean_evaluation                      # full run
    python -m src.evaluation.clean_evaluation --summary-only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from src.data.flow_to_text import flow_to_text


def sample_clean_flows(config_path: str = "config.yaml", n_per_class: int = 30) -> pd.DataFrame:
    """Sample a balanced set of real ATTACK/BENIGN flows, excluding rows
    already used in the adversarial corpus."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    seed = config["experiment"]["random_seed"]
    adv = config["adversarial"]

    test_df = pd.read_csv("data/processed/test.csv", low_memory=False)

    used_indices = set()
    corpus_path = Path(adv["corpus_path"])
    if corpus_path.exists():
        corpus = pd.read_csv(corpus_path, keep_default_na=False)
        if "source_index" in corpus.columns:
            used_indices = set(corpus["source_index"].tolist())

    available = test_df[~test_df.index.isin(used_indices)]
    print(
        f"Excluded {len(test_df) - len(available)} rows already used in the "
        f"adversarial corpus (kept independent of Phase 5/6 results)"
    )

    parts = []
    for label in ["ATTACK", "BENIGN"]:
        pool = available[available["label"] == label]
        n = min(n_per_class, len(pool))
        if n < n_per_class:
            print(f"WARNING: only {n} {label} flows available (wanted {n_per_class})")
        parts.append(pool.sample(n=n, random_state=seed))

    sample = pd.concat(parts).sample(frac=1, random_state=seed)  # shuffle
    print(f"Sampled {len(sample)} flows: {sample['label'].value_counts().to_dict()}")
    return sample


def run_clean_eval(
    sample: pd.DataFrame,
    classifiers: dict,
    results_path,
    excluded_cols: list,
    pause_seconds: float = 0.0,
    dataset: str = "real",
) -> None:
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if results_path.exists():
        prev = pd.read_csv(results_path, keep_default_na=False)
        done = set(zip(prev["source_index"].astype(str), prev["agent"], prev["dataset"]))

    for idx, row in sample.iterrows():
        row_dict = row.to_dict()
        flow_text = flow_to_text(row_dict, excluded_columns=excluded_cols)

        for agent_name, classify in classifiers.items():
            key = (str(idx), agent_name, dataset)
            if key in done:
                print(f"[{idx}] {agent_name}: already done, skipping")
                continue

            try:
                result = classify(flow_text)
            except Exception as e:
                raise SystemExit(
                    f"\nStopped at row {idx} / {agent_name}: {type(e).__name__}: {e}\n"
                    f"Progress so far is saved. Re-run the same command to resume."
                )
            time.sleep(pause_seconds)

            out = {
                "source_index": idx,
                "agent": agent_name,
                "dataset": dataset,
                "true_label": row_dict["label"],
                "predicted_label": result["label"],
                "rationale": result.get("rationale", ""),
            }
            pd.DataFrame([out]).to_csv(results_path, mode="a", header=not results_path.exists(), index=False)
            print(f"[{idx}] {agent_name}: true={out['true_label']} pred={out['predicted_label']}")


def summarize(results_path, dataset: str | None = None) -> None:
    """Print accuracy/precision/recall/F1 and a confusion matrix per agent."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    df = pd.read_csv(results_path, keep_default_na=False)
    available = sorted(df["dataset"].unique())
    if dataset:
        df = df[df["dataset"] == dataset]
        label = f"dataset={dataset}"
    else:
        label = f"ALL DATASETS BLENDED ({', '.join(available)}) -- pass --dataset to split these apart"

    print(f"\n=== Clean-data metrics ({label}) ===\n")
    summary_rows = []
    for agent_name, group in df.groupby("agent"):
        y_true = group["true_label"]
        y_pred = group["predicted_label"]
        acc = (y_true == y_pred).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=["ATTACK"], zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=["BENIGN", "ATTACK"])

        print(f"--- {agent_name} (n={len(group)}) ---")
        print(f"Accuracy:  {acc:.0%}")
        print(f"Precision (ATTACK): {precision[0]:.0%}")
        print(f"Recall (ATTACK):    {recall[0]:.0%}")
        print(f"F1 (ATTACK):        {f1[0]:.0%}")
        print("Confusion matrix (rows=true, cols=predicted, order=[BENIGN, ATTACK]):")
        print(cm)
        print()

        summary_rows.append({
            "agent": agent_name, "dataset": dataset or "blended", "n": len(group),
            "accuracy": acc, "precision_attack": precision[0], "recall_attack": recall[0],
            "f1_attack": f1[0], "tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1],
        })

    out_path = Path("results/tables/clean_metrics_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    print(f"Saved metrics summary to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7: clean-data accuracy/precision/recall/F1 evaluation.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-per-class", type=int, default=30, help="ATTACK and BENIGN flows to sample each")
    parser.add_argument("--agents", default="single_shot,react")
    parser.add_argument("--dataset", default="real")
    parser.add_argument("--results-path", default="results/tables/clean_evaluation.csv")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.summary_only:
        summarize(args.results_path, dataset=args.dataset)
        return

    with open(args.config) as f:
        config = yaml.safe_load(f)

    sample = sample_clean_flows(args.config, args.n_per_class)

    wanted = {a.strip() for a in args.agents.split(",")}
    classifiers = {}
    if "single_shot" in wanted:
        from src.agents.single_shot_agent import build_agent
        classifiers["single_shot"] = build_agent(args.config)
    if "react" in wanted:
        from src.agents.react_agent import build_react_agent
        react_classify = build_react_agent(args.config)
        classifiers["react"] = lambda flow_text: react_classify(flow_text)  # no tool override, no defense

    print(f"\nRunning clean evaluation against: {sorted(classifiers)}\n")
    run_clean_eval(
        sample, classifiers, args.results_path,
        config["agent"]["excluded_from_prompt"], config["adversarial"]["pause_seconds"], args.dataset,
    )
    summarize(args.results_path, dataset=args.dataset)


if __name__ == "__main__":
    main()