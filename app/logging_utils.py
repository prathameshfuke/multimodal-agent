"""Small, dependency-free structured logging helper for request diagnostics."""

import json
import logging
from typing import Any


logger = logging.getLogger("multimodal_agent")


def log_event(*, thread_id: str, node: str, status: str, latency_ms: int, **extra: Any) -> None:
    """Emit one JSON log record for a graph boundary or tool dispatch."""
    payload = {
        "thread_id": thread_id,
        "node": node,
        "status": status,
        "latency_ms": latency_ms,
        **extra,
    }
    logger.info("multimodal_agent_event=%s", json.dumps(payload, default=str, sort_keys=True))
