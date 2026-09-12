"""
Phase 4: Mock IP reputation lookup tool.

This is NOT a real external service (per the project's own rules) -- it's
a small deterministic function simulating what a reputation API might
return, so the ReAct agent has something to call as its "tool use" step.

Deterministic means: the same IP always produces the same result. This
matters for reproducibility, and later (Phase 5) for comparing this
"clean" tool against a deliberately poisoned version under the same
conditions.
"""

from __future__ import annotations

import hashlib


def lookup_ip_reputation(ip: str) -> dict:
    """Look up the reputation of an IP address.

    Args:
        ip: The IP address to check, e.g. 192.168.1.5
    """
    digest = hashlib.sha256(ip.encode()).hexdigest()
    score = int(digest[:8], 16) % 100  # deterministic pseudo-random 0-99

    if score < 15:
        status = "known_malicious"
        grade = "high_suspicion"
    elif score < 40:
        status = "unverified"
        grade = "moderate_suspicion"
    else:
        status = "trusted"
        grade = "low_suspicion"

    return {
        "ip": ip,
        "reputation_score": score,
        "trust_status": status,
        "suspiciousness_grade": grade,
    }