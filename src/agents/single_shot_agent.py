"""
Phase 3: Agent 1 -- the single-shot classifier.

Takes one flow's text description, sends it to Gemini in a single pass
(no tools, no multi-step reasoning -- that's Phase 4), and returns a
BENIGN/ATTACK verdict plus a short rationale.

What this file deliberately does NOT do (that's for later phases):
  - No tool calls / ReAct loop.
  - No adversarial handling or defenses.
  - No batch evaluation metrics (accuracy, F1, etc.) -- just single calls.
"""

from __future__ import annotations

import os
import re

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types
from src.agents.gemini_utils import generate_with_retry

SYSTEM_PROMPT = """You are an IoT network security analyst. You will be given \
a description of a single network flow's observed properties (IP addresses, \
ports, protocol, and similar fields).

Decide whether this flow is BENIGN (normal, legitimate traffic) or ATTACK \
(malicious or suspicious activity), based only on the evidence given.

Respond in EXACTLY this format, with nothing before or after it:
LABEL: BENIGN or ATTACK
RATIONALE: <one short sentence explaining your reasoning>
"""


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_client() -> genai.Client:
    """Create the Gemini client using the API key from .env."""
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found -- check your .env file.")
    return genai.Client(api_key=api_key)


def parse_response(text: str) -> dict:
    """
    Pull LABEL and RATIONALE out of the model's reply.

    If the model doesn't follow the format exactly, we don't crash --
    we return label=UNPARSEABLE so this is visible in results instead
    of silently miscounting it as BENIGN or ATTACK.
    """
    text = text or ""  # the model can return None when its reply has no text part
    label_match = re.search(r"LABEL:\s*(BENIGN|ATTACK)", text, re.IGNORECASE)
    rationale_match = re.search(r"RATIONALE:\s*(.+)", text, re.IGNORECASE)

    label = label_match.group(1).upper() if label_match else "UNPARSEABLE"
    rationale = rationale_match.group(1).strip() if rationale_match else text.strip()

    return {"label": label, "rationale": rationale, "raw_response": text}


def classify_flow(client: genai.Client, model: str, temperature: float, flow_text: str) -> dict:
    """Send one flow description to Gemini and return the parsed verdict."""
    response = generate_with_retry(
        client,
        model,
        flow_text,
        types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
        ),
    )
    return parse_response(response.text)


def build_agent(config_path: str = "config.yaml"):
    """
    Convenience wrapper: reads config.yaml once and returns a ready-to-use
    function classify(flow_text) -> dict, so calling code doesn't need to
    juggle the client/model/temperature separately.
    """
    config = load_config(config_path)
    agent_cfg = config["agent"]
    client = build_client()

    def classify(flow_text: str) -> dict:
        return classify_flow(client, agent_cfg["model"], agent_cfg["temperature"], flow_text)

    return classify