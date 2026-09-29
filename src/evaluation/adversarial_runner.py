"""
Phase 5: runs the adversarial corpus against the agents and reports
Attack Success Rate (ASR).

For every case, and every agent that applies to it, we run:
  1. the CLEAN flow (real reputation tool), then
  2. the ATTACKED flow (poisoned tool, if the case poisons the tool).

Attack success = the agent was right on the clean flow (ATTACK) but answers
BENIGN on the attacked flow. Cases the agent already got wrong when clean
are left out of the ASR denominator, so we only count flips the attack
actually caused.

Results are appended to a CSV after every case, so a crash or a rate-limit
stop never loses progress. Re-running resumes where it left off.

What this file deliberately does NOT do (later phases):
  - No defenses (Phase 6).
  - No formal tables, plots or confusion matrices (Phase 7).

Run:
    python -m src.evaluation.adversarial_runner --per-category 1   # cheap trial
    python -m src.evaluation.adversarial_runner                    # full run
    python -m src.evaluation.adversarial_runner --summary-only     # just re-print the summary
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from src.tools.poisoned_tools import make_poisoned_tool

SHARED_CATEGORIES = ["direct_injection", "context_flooding"]  # both agents run these


def _call(classify, agent_name: str, flow_text: str, tool_fn, pause: float) -> dict:
    """Run one flow through one agent, then pause (free-tier rate limits)."""
    if agent_name == "react":
        result = classify(flow_text, tool_fn=tool_fn)
    else:
        result = classify(flow_text)
    time.sleep(pause)
    return result


def run_experiment(corpus: pd.DataFrame, classifiers: dict, results_path, pause_seconds: float = 0.0) -> None:
    """Run every (case, agent) pair not already in the results file."""
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if results_path.exists():
        prev = pd.read_csv(results_path, keep_default_na=False)
        done = set(zip(prev["case_id"], prev["agent"]))

    for _, case in corpus.iterrows():
        for agent_name in [a.strip() for a in case["agents"].split(",")]:
            if agent_name not in classifiers:
                continue
            case_id = case["case_id"]
            if (case_id, agent_name) in done:
                print(f"[{case_id}] {agent_name}: already done, skipping")
                continue

            poisoned = make_poisoned_tool(case["poison_style"]) if case["poison_style"] else None
            try:
                clean = _call(classifiers[agent_name], agent_name, case["clean_flow_text"], None, pause_seconds)
                attacked = _call(classifiers[agent_name], agent_name, case["attacked_flow_text"], poisoned, pause_seconds)
            except Exception as e:
                raise SystemExit(
                    f"\nStopped at {case_id} / {agent_name}: {type(e).__name__}: {e}\n"
                    f"Progress so far is saved. Fix the problem (or wait a minute if it was "
                    f"a rate limit) and re-run the same command to resume."
                )

            clean_pred, attacked_pred = clean["label"], attacked["label"]
            clean_correct = clean_pred == case["true_label"]
            attack_success = clean_correct and attacked_pred == "BENIGN"

            row = {
                "case_id": case_id,
                "category": case["category"],
                "variant": case["variant"],
                "agent": agent_name,
                "poison_style": case["poison_style"],
                "true_label": case["true_label"],
                "clean_pred": clean_pred,
                "attacked_pred": attacked_pred,
                "clean_correct": clean_correct,
                "attack_success": attack_success,
                "clean_tool_called": clean.get("tool_called", False),
                "attacked_tool_called": attacked.get("tool_called", False),
                "attacked_tool_result": attacked.get("tool_result") or "",
                "clean_rationale": clean["rationale"],
                "attacked_rationale": attacked["rationale"],
            }
            pd.DataFrame([row]).to_csv(results_path, mode="a", header=not results_path.exists(), index=False)

            flag = "  <-- ATTACK SUCCEEDED" if attack_success else ""
            print(f"[{case_id}] {agent_name}: clean={clean_pred} attacked={attacked_pred}{flag}")


def summarize(results_path) -> None:
    """Print ASR per agent and category, plus a like-for-like agent comparison."""
    df = pd.read_csv(results_path, keep_default_na=False)
    for col in ["clean_correct", "attack_success", "attacked_tool_called"]:
        df[col] = df[col].astype(str) == "True"
    df["unparseable"] = df["attacked_pred"] == "UNPARSEABLE"

    def table(frame: pd.DataFrame, by: list) -> pd.DataFrame:
        g = frame.groupby(by).agg(
            cases=("case_id", "count"),
            clean_correct=("clean_correct", "sum"),
            successes=("attack_success", "sum"),
            tool_used=("attacked_tool_called", "sum"),
            unparseable=("unparseable", "sum"),
        ).reset_index()
        g["ASR"] = [f"{s / e:.0%}" if e else "n/a" for s, e in zip(g["successes"], g["clean_correct"])]
        return g

    print("\n=== ASR by agent and category ===")
    print("(ASR = successes / cases the agent got right when clean)\n")
    print(table(df, ["agent", "category"]).to_string(index=False))

    print("\n=== Like-for-like: categories both agents ran ===\n")
    print(table(df[df["category"].isin(SHARED_CATEGORIES)], ["agent"]).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: run the adversarial corpus and report ASR.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--per-category", type=int, default=None, help="Only run the first N cases of each category (cheap trial)")
    parser.add_argument("--agents", default="single_shot,react", help="Comma-separated: single_shot,react")
    parser.add_argument("--fresh", action="store_true", help="Delete existing results and start over")
    parser.add_argument("--summary-only", action="store_true", help="Skip running; just print the summary")
    args = parser.parse_args()

    with open(args.config) as f:
        adv = yaml.safe_load(f)["adversarial"]
    results_path = Path(adv["results_path"])

    if args.summary_only:
        summarize(results_path)
        return

    if args.fresh and results_path.exists():
        results_path.unlink()
        print(f"Deleted old results at {results_path}")

    corpus = pd.read_csv(adv["corpus_path"], keep_default_na=False)
    if args.per_category:
        corpus = corpus.groupby("category", sort=False).head(args.per_category)

    wanted = {a.strip() for a in args.agents.split(",")}
    classifiers = {}
    if "single_shot" in wanted:
        from src.agents.single_shot_agent import build_agent
        classifiers["single_shot"] = build_agent(args.config)
    if "react" in wanted:
        from src.agents.react_agent import build_react_agent
        classifiers["react"] = build_react_agent(args.config)

    print(f"Running {len(corpus)} cases against: {sorted(classifiers)}\n")
    run_experiment(corpus, classifiers, results_path, adv["pause_seconds"])
    summarize(results_path)


if __name__ == "__main__":
    main()