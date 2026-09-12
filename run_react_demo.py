"""
Phase 4 demo/smoke test.

Runs the ReAct agent on a handful of real rows from data/processed/test.csv
and shows, for each one: whether it called the reputation tool, what the
tool returned, and its final verdict vs. the actual label.

This is NOT the formal evaluation (accuracy, F1, single-shot vs ReAct
comparison) -- that's Phase 7. This is just "does the multi-step loop
actually work end to end."

Usage:
    python run_react_demo.py --n 5
"""

import argparse

import pandas as pd
import yaml

from src.agents.react_agent import build_react_agent
from src.data.flow_to_text import flow_to_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of sample flows to test")
    parser.add_argument("--input", default="data/processed/test.csv")
    parser.add_argument("--seed", type=int, default=None, help="Override config's random seed")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else config["experiment"]["random_seed"]
    excluded_cols = config["agent"]["excluded_from_prompt"]

    df = pd.read_csv(args.input)
    sample = df.sample(n=min(args.n, len(df)), random_state=seed).reset_index(drop=True)

    classify = build_react_agent()

    correct = 0
    tool_calls = 0
    print(f"Running ReAct agent on {len(sample)} sample flows...\n")
    for i, row in sample.iterrows():
        row_dict = row.to_dict()
        actual_label = row_dict["label"]

        flow_text = flow_to_text(row_dict, excluded_columns=excluded_cols)
        result = classify(flow_text)

        is_correct = result["label"] == actual_label
        correct += int(is_correct)
        tool_calls += int(result["tool_called"])
        mark = "✓" if is_correct else "✗"

        print(f"[{i}] {mark} predicted={result['label']:<12} actual={actual_label}")
        print(f"     tool called: {result['tool_called']}")
        if result["tool_called"]:
            print(f"     tool result: {result['tool_result']}")
        print(f"     rationale: {result['rationale']}")
        print(f"     flow: {flow_text}")
        print()

    print(f"Result: {correct}/{len(sample)} correct. Tool used in {tool_calls}/{len(sample)} cases.")
    print("(This is a smoke test, not a formal evaluation -- that's Phase 7.)")


if __name__ == "__main__":
    main()