from pydantic import BaseModel
from typing import Literal

class ExtractionResult(BaseModel):
    source_file: str
    modality: Literal["pdf", "image", "audio", "text"]
    text: str
    confidence: float  # 0-1
    low_confidence: bool
    warnings: list[str] = []
