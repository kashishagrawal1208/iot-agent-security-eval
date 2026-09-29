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
    Call client.models.generate_content, retrying on 429 (rate limit) and
    transient 5xx server errors. A DAILY quota error is different -- no
    amount of waiting a few seconds helps -- so we detect that specifically
    and fail immediately with a clear message instead of wasting retries.
    """
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except (errors.ClientError, errors.ServerError) as e:
            code = getattr(e, "code", None)

            if code == 429 and _is_daily_quota_error(e):
                raise RuntimeError(
                    f"Hit a DAILY free-tier quota for model '{model}'. This won't clear "
                    f"in seconds -- it resets at midnight Pacific time, or switch to a "
                    f"model with more daily headroom in config.yaml."
                ) from e

            is_rate_limit = code == 429
            is_server_hiccup = code is not None and code >= 500
            if not (is_rate_limit or is_server_hiccup) or attempt == max_retries:
                raise
            if is_rate_limit:
                wait_seconds = _extract_retry_delay(e) or 10
                print(f"  (rate limited, waiting {wait_seconds:.0f}s before retrying...)")
            else:
                wait_seconds = 20
                print(f"  (server error {code}, waiting {wait_seconds}s before retrying...)")
            time.sleep(wait_seconds)


def _is_daily_quota_error(error) -> bool:
    """Check whether a 429's quota metric is a per-DAY cap rather than per-minute."""
    try:
        details = error.details.get("error", {}).get("details", [])
        for d in details:
            if d.get("@type", "").endswith("QuotaFailure"):
                for v in d.get("violations", []):
                    if "PerDay" in v.get("quotaId", ""):
                        return True
    except Exception:
        pass
    return False

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