from typing import TypedDict, Literal
from app.schemas.extraction import ExtractionResult
from app.schemas.plan import Plan
from app.schemas.trace import TraceEvent
from app.schemas.output import FinalOutput

class UploadedFileRef(TypedDict):
    filename: str
    content_type: str
    path: str  # temp storage path, not raw bytes in state

class AgentState(TypedDict):
    session_id: str
    user_query: str
    raw_files: list[UploadedFileRef]
    extraction_results: list[ExtractionResult]
    fused_context: str
    detected_urls: list[str]
    plan: Plan | None
    trace: list[TraceEvent]
    clarify_question: str | None
    final_output: FinalOutput | None
    # Declared here (not a scratch key) so MemorySaver checkpoints it on clarify-pause.
    # See decisions.md Decision 9.
    tool_outputs: dict
    status: Literal[
        "extracting", "planning", "executing",
        "awaiting_clarification", "done", "error"
    ]
