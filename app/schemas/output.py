from pydantic import BaseModel
from typing import Literal

class SummaryOutput(BaseModel):
    one_line: str
    bullets: list[str]  # exactly 3
    five_sentence: str

class SentimentOutput(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float
    justification: str  # one line

class FinalOutput(BaseModel):
    task_type: str
    summary: SummaryOutput | None = None
    sentiment: SentimentOutput | None = None
    raw_text: str | None = None  # for code-explain, comparisons, conversational
