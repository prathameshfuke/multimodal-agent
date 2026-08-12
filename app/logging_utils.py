"""Small, dependency-free structured logging helper for request diagnostics."""

import json
import logging
from typing import Any


logger = logging.getLogger("multimodal_agent")


def log_event(*, thread_id: str, node: str, status: str, latency_ms: int, **extra: Any) -> None:
    """Emit one JSON log record for a graph boundary or tool dispatch.

    Sanitizes sensitive kwargs to guarantee API keys/secrets are never logged.
    """
    sanitized_extra = {}
    for k, v in extra.items():
        k_lower = k.lower()
        if any(secret_term in k_lower for secret_term in ("key", "secret", "token", "auth", "password")):
            sanitized_extra[k] = "[REDACTED]"
        else:
            sanitized_extra[k] = v

    payload = {
        "thread_id": thread_id,
        "node": node,
        "status": status,
        "latency_ms": latency_ms,
        **sanitized_extra,
    }
    logger.info("multimodal_agent_event=%s", json.dumps(payload, default=str, sort_keys=True))
