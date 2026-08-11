from pydantic import BaseModel
from typing import Literal

class TraceEvent(BaseModel):
    step_index: int
    tool_name: str
    input_summary: str
    output_summary: str
    latency_ms: int
    status: Literal["success", "retried", "failed", "partial"]
