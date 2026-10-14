"""
Sample-size progression for the CLEAN-data evaluation.

What "n" means here: the number of real flows in the clean evaluation sample,
always balanced 50% ATTACK / 50% BENIGN (n=60 -> 30 + 30, n=180 -> 90 + 90).
The agents, prompts, model, temperature and metrics are identical for every
n; only the number of flows used to *measure* them changes. n does not change
the agents in any way (nothing is trained), so this experiment shows how the
measurement converges, NOT a learning curve.

How the subsets are built (nested, so every n reuses the smaller ones):
  1. A persisted "master order" (results/tables/clean_master_order.csv) ranks
     the flows within each class. Rank 1..30 = the original n=60 baseline set;
     ranks 31..75 = the flows added for n=150 (seeded shuffle); ranks 76..90 =
     new flows drawn for n=180 (excluding every flow already in the master list
     and every flow in the adversarial corpus).
  2. A subset for size n is simply "rank <= n/2 in each class".
  3. Only flows with no stored prediction yet are sent to the LLM; every other
     n is computed from predictions already on disk.

Run:
    python -m src.evaluation.sample_size_progression --plan-only    # no API calls
    python -m src.evaluation.sample_size_progression                # run missing flows, then summarize
    python -m src.evaluation.sample_size_progression --summary-only
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import yaml

SIZES = [60, 80, 100, 120, 150, 180]
N_BASELINE = 60
RESULTS_PATH = Path("results/tables/clean_evaluation.csv")
ORDER_PATH = Path("results/tables/clean_master_order.csv")
OUT_METRICS = Path("results/tables/sample_size_progression_clean.csv")
OUT_PAIRED = Path("results/tables/sample_size_progression_mcnemar.csv")
# Confusion counts (tn, fp, fn, tp) reported earlier for the original n=60 run, for a sanity check only.
EARLIER_N60 = {"single_shot": (28, 2, 13, 17), "react": (30, 0, 17, 13)}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = (z / d) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, c - m), min(1.0, c + m))


def migrate_model_column(path: Path, label: str = "unknown_mixed") -> None:
    """One-time: add a 'model' column to the existing results file so new rows (which record the model) stay aligned."""
    df = pd.read_csv(path, keep_default_na=False)
    if "model" not in df.columns:
        df["model"] = label
        df.to_csv(path, index=False)
        print(f"Added 'model' column to {path} (existing rows marked '{label}').")


def build_master_order(results_path: Path, dataset: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(results_path, keep_default_na=False)
    df = df[df["dataset"] == dataset]
    flows = df.drop_duplicates("source_index")[["source_index", "true_label"]].reset_index(drop=True)
    if len(flows) < N_BASELINE:
        raise SystemExit(f"Need at least {N_BASELINE} evaluated flows to build the master order, found {len(flows)}.")
    baseline, rest = flows.iloc[:N_BASELINE], flows.iloc[N_BASELINE:]
    counts = baseline["true_label"].value_counts()
    if counts.get("ATTACK", 0) != N_BASELINE // 2 or counts.get("BENIGN", 0) != N_BASELINE // 2:
        raise SystemExit(f"First {N_BASELINE} flows are not 30/30 balanced ({counts.to_dict()}); stop and inspect the file.")
    rows = []
    for label in ["ATTACK", "BENIGN"]:
        ordered = pd.concat([baseline[baseline["true_label"] == label],
                             rest[rest["true_label"] == label].sample(frac=1, random_state=seed)])
        for rank, sid in enumerate(ordered["source_index"], 1):
            rows.append({"source_index": int(sid), "label": label, "rank_in_class": rank,
                         "stage": "baseline" if rank <= N_BASELINE // 2 else "added"})
    return pd.DataFrame(rows)


def extend_master(master: pd.DataFrame, per_class: int, test_df: pd.DataFrame, corpus_ids: set, seed: int) -> pd.DataFrame:
    used = set(master["source_index"]) | corpus_ids
    rows = []
    for label in ["ATTACK", "BENIGN"]:
        have = int((master["label"] == label).sum())
        missing = per_class - have
        if missing <= 0:
            continue
        pool = test_df[(test_df["label"] == label) & ~test_df.index.isin(used)]
        new = pool.sample(n=missing, random_state=seed)
        for rank, sid in enumerate(new.index, have + 1):
            rows.append({"source_index": int(sid), "label": label, "rank_in_class": rank, "stage": "extension"})
    return pd.concat([master, pd.DataFrame(rows)], ignore_index=True) if rows else master


def subset_ids(master: pd.DataFrame, n: int) -> set:
    return set(master.loc[master["rank_in_class"] <= n // 2, "source_index"])


def confusion(g: pd.DataFrame) -> tuple:
    t, p = g["true_label"], g["predicted_label"]
    tp = int(((t == "ATTACK") & (p == "ATTACK")).sum())
    fn = int(((t == "ATTACK") & (p != "ATTACK")).sum())
    fp = int(((t == "BENIGN") & (p == "ATTACK")).sum())
    tn = int(((t == "BENIGN") & (p == "BENIGN")).sum())
    return tn, fp, fn, tp


def progression_tables(res: pd.DataFrame, master: pd.DataFrame, dataset: str, sizes=SIZES):
    res = res[res["dataset"] == dataset]
    metric_rows, paired_rows = [], []
    for n in sizes:
        ids = subset_ids(master, n)
        per_agent = {}
        for agent, g in res.groupby("agent"):
            g = g[g["source_index"].isin(ids)].drop_duplicates("source_index")
            if len(g) != n:
                print(f"  (n={n}, {agent}: only {len(g)} of {n} flows evaluated yet -- skipped)")
                continue
            tn, fp, fn, tp = confusion(g)
            correct = int((g["true_label"] == g["predicted_label"]).sum())
            lo, hi = wilson_ci(correct, n)
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            rec = tp / (tp + fn) if (tp + fn) else float("nan")
            f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and (prec + rec) else float("nan")
            metric_rows.append({"n": n, "agent": agent, "accuracy": correct / n, "acc_ci_low": lo, "acc_ci_high": hi,
                                "precision": prec, "recall": rec, "f1": f1,
                                "FNR": fn / (fn + tp) if (fn + tp) else float("nan"),
                                "FPR": fp / (fp + tn) if (fp + tn) else float("nan"),
                                "tn": tn, "fp": fp, "fn": fn, "tp": tp})
            per_agent[agent] = g.set_index("source_index")
        if {"single_shot", "react"} <= per_agent.keys():
            from scipy.stats import chi2
            a, b = per_agent["single_shot"], per_agent["react"].loc[per_agent["single_shot"].index]
            a_ok, b_ok = a["predicted_label"] == a["true_label"], b["predicted_label"] == b["true_label"]
            only_ss, only_re = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
            disc = only_ss + only_re
            stat = (abs(only_ss - only_re) - 1) ** 2 / disc if disc else 0.0
            paired_rows.append({"n": n, "single_shot_only_correct": only_ss, "react_only_correct": only_re,
                                "statistic": stat, "p_value": float(chi2.sf(stat, 1)) if disc else 1.0})
    return pd.DataFrame(metric_rows), pd.DataFrame(paired_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean-data sample-size progression (nested subsets).")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", default="real")
    ap.add_argument("--extend-to", type=int, default=max(SIZES))
    ap.add_argument("--plan-only", action="store_true", help="build/extend the master order and report the API calls needed; call nothing")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    seed = config["experiment"]["random_seed"]

    migrate_model_column(RESULTS_PATH)

    if args.summary_only:
        master = pd.read_csv(ORDER_PATH)
    else:
        test_df = pd.read_csv("data/processed/test.csv", low_memory=False)
        corpus_ids = set(pd.read_csv(config["adversarial"]["corpus_path"], keep_default_na=False)["source_index"])
        master = pd.read_csv(ORDER_PATH) if ORDER_PATH.exists() else build_master_order(RESULTS_PATH, args.dataset, seed)
        master = extend_master(master, args.extend_to // 2, test_df, corpus_ids, seed)
        master.to_csv(ORDER_PATH, index=False)
        overlap = set(master["source_index"]) & corpus_ids
        print(f"Master order: {len(master)} flows. Overlap with adversarial corpus: {len(overlap)} (must be 0).")
        if overlap:
            print("  WARNING -- overlapping flows:", sorted(overlap))

        res = pd.read_csv(RESULTS_PATH, keep_default_na=False)
        res = res[res["dataset"] == args.dataset]
        agents = ["single_shot", "react"]
        wanted = master.loc[master["rank_in_class"] <= args.extend_to // 2, "source_index"].tolist()
        done = set(zip(res["source_index"].astype(int), res["agent"]))
        missing = [sid for sid in wanted if any((sid, a) not in done for a in agents)]
        calls = sum(1 for sid in wanted for a in agents if (sid, a) not in done)
        print(f"Flows still to evaluate: {len(missing)}  (agent runs needed: {calls}; a react run can use up to 2 API calls)")
        if args.plan_only:
            return
        if missing:
            from src.agents.react_agent import build_react_agent
            from src.agents.single_shot_agent import build_agent
            from src.evaluation.clean_evaluation import run_clean_eval
            ss, re_ = build_agent(args.config), build_react_agent(args.config)
            classifiers = {"single_shot": ss, "react": lambda t: re_(t)}
            run_clean_eval(test_df.loc[missing], classifiers, RESULTS_PATH, config["agent"]["excluded_from_prompt"],
                           config["adversarial"]["pause_seconds"], args.dataset, model=config["agent"]["model"])

    res = pd.read_csv(RESULTS_PATH, keep_default_na=False)
    metrics, paired = progression_tables(res, master, args.dataset)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_METRICS, index=False)
    paired.to_csv(OUT_PAIRED, index=False)
    pd.set_option("display.width", 200)
    print("\n=== Clean-data progression (dataset=%s) ===\n" % args.dataset)
    print(metrics.round(3).to_string(index=False))
    print("\n=== Paired McNemar test, single_shot vs react ===\n")
    print(paired.round(4).to_string(index=False))
    base = metrics[metrics["n"] == N_BASELINE]
    for _, r in base.iterrows():
        if r["agent"] in EARLIER_N60:
            now = (int(r["tn"]), int(r["fp"]), int(r["fn"]), int(r["tp"]))
            tag = "MATCHES" if now == EARLIER_N60[r["agent"]] else "DIFFERS FROM"
            print(f"\nn=60 {r['agent']} (tn,fp,fn,tp)={now} {tag} the originally reported {EARLIER_N60[r['agent']]}")
    print(f"\nSaved {OUT_METRICS} and {OUT_PAIRED}")


if __name__ == "__main__":
    main()