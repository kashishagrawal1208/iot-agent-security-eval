"""
Turns one row of processed flow data into a plain-English description
for the LLM agent to read.

Critically, this drops any column listed in config.yaml's
agent.excluded_from_prompt (label, attack_type) -- the agent must decide
using only the raw flow evidence, never the ground-truth answer.
"""

from __future__ import annotations


def flow_to_text(row: dict, excluded_columns: list[str]) -> str:
    """
    Build a one-sentence, human-readable description of a network flow.

    Example output:
        "Network flow observed with: ip.src_host=192.168.1.5;
         ip.dst_host=10.0.0.3; protocol=TCP; tcp.dstport=80; ..."

    `row` is a single row of data (e.g. from a pandas DataFrame, converted
    with .to_dict()). `excluded_columns` comes straight from config.yaml,
    so the exclusion list lives in one place, not scattered across files.
    """
    excluded = set(excluded_columns)
    parts = [
        f"{col}={value}"
        for col, value in row.items()
        if col not in excluded
    ]
    return "Network flow observed with: " + "; ".join(parts) + "."