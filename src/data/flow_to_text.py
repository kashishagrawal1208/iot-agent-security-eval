"""
Turns one row of processed flow data into a plain-English description
for the LLM agent to read.

Critically, this drops any column listed in config.yaml's
agent.excluded_from_prompt (label, attack_type, attack_label) -- the agent
must decide using only the raw flow evidence, never the ground-truth answer.

It also drops "filler" values -- fields that are 0, 0.0, or "unknown" for
protocols this flow doesn't use. Edge-IIoTset uses one shared row schema
across all protocols (ARP, ICMP, HTTP, DNS, MQTT, Modbus, ...), so any
single flow only has real values in the handful of fields for its own
protocol; everything else sits at a placeholder zero. Without trimming
those out, every prompt would be ~90% noise before any attack is even
added -- which would make it impossible to tell a real context-flooding
attack from the dataset's own formatting. This is a deliberate
simplification: a field that is genuinely, meaningfully zero (not just
an unused-protocol placeholder) gets dropped too, which is an accepted
trade-off for readable, on-topic prompts.
"""

from __future__ import annotations

_FILLER_VALUES = {"", "0", "0.0", "nan", "none", "unknown", "-"}


def _is_filler(value) -> bool:
    return str(value).strip().lower() in _FILLER_VALUES


def flow_to_text(row: dict, excluded_columns: list[str]) -> str:
    """
    Build a one-sentence, human-readable description of a network flow,
    keeping only fields with a real (non-filler) value.

    `row` is a single row of data (e.g. from a pandas DataFrame, converted
    with .to_dict()). `excluded_columns` comes straight from config.yaml,
    so the exclusion list lives in one place, not scattered across files.
    """
    excluded = set(excluded_columns)
    parts = [
        f"{col}={value}"
        for col, value in row.items()
        if col not in excluded and not _is_filler(value)
    ]
    if not parts:
        # Extremely unlikely (every field was filler), but keep this
        # function total rather than emitting an empty/confusing sentence.
        parts = ["no non-default fields observed"]
    return "Network flow observed with: " + "; ".join(parts) + "."