from pydantic import BaseModel

class ToolCall(BaseModel):
    tool_name: str
    args: dict
    reason: str  # one line, shown directly in the trace UI

class Plan(BaseModel):
    steps: list[ToolCall] = []
    clarify_question: str | None = None  # set instead of steps if ambiguous
