"""
Shared helper for calling the Gemini API with automatic retry on rate-limit
(429) errors. Both agents (single-shot and ReAct) use this so a temporary
free-tier quota hit pauses and retries instead of crashing the whole run.
"""

from __future__ import annotations

import time

from google.genai import errors


def generate_with_retry(client, model, contents, config, max_retries=5):
    """
    Call client.models.generate_content, retrying on 429 RESOURCE_EXHAUSTED
    errors. Waits however long Gemini says to wait (falls back to 10s if
    it doesn't say), up to max_retries times before giving up for real.
    """
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except errors.ClientError as e:
            if getattr(e, "code", None) != 429 or attempt == max_retries:
                raise
            wait_seconds = _extract_retry_delay(e) or 10
            print(f"  (rate limited, waiting {wait_seconds:.0f}s before retrying...)")
            time.sleep(wait_seconds)


def _extract_retry_delay(error) -> float | None:
    """Pull the suggested retry delay (in seconds) out of the API's error details, if present."""
    try:
        details = error.details.get("error", {}).get("details", [])
        for d in details:
            if d.get("@type", "").endswith("RetryInfo"):
                delay_str = d.get("retryDelay", "")  # e.g. "9s"
                return float(delay_str.rstrip("s"))
    except Exception:
        return None
    return None