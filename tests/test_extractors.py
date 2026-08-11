"""
Unit tests for multimodal extraction layer (pdf_extractor, image_extractor, audio_extractor).

Note for Developers:
Real sample files used by these unit tests are located in:
  tests/samples/
    ├── clean_sample.pdf     # Clean digital text PDF document
    ├── sample_image.png     # Sample PNG image file
    └── sample_audio.wav     # 1-second 16kHz mono audio clip

You can drop additional real production sample files into `tests/samples/` to run extended tests.
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from app.schemas.extraction import ExtractionResult
from app.extractors.pdf_extractor import extract_pdf
from app.extractors.image_extractor import extract_image
from app.extractors.audio_extractor import extract_audio

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
PDF_SAMPLE = SAMPLES_DIR / "clean_sample.pdf"
IMAGE_SAMPLE = SAMPLES_DIR / "sample_image.png"
AUDIO_SAMPLE = SAMPLES_DIR / "sample_audio.wav"


class DummyGeminiClient:
    """Mock Gemini client for fallback retry testing."""

    __slots__ = ("should_fail", "return_text", "call_count")

    def __init__(
        self, should_fail: bool = False, return_text: str = "Fallback extracted text"
    ):
        self.should_fail = should_fail
        self.return_text = return_text
        self.call_count = 0

    def generate_content(self, prompt_or_contents):
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError("Simulated Gemini API error")
        mock_response = MagicMock()
        mock_response.text = self.return_text
        return mock_response


def test_pdf_extractor_clean_sample():
    """Test PDF extractor on clean text PDF file."""
    assert PDF_SAMPLE.exists(), f"Sample PDF missing at {PDF_SAMPLE}"
    result = asyncio.run(extract_pdf(str(PDF_SAMPLE)))

    assert isinstance(result, ExtractionResult)
    assert result.modality == "pdf"
    assert result.source_file == str(PDF_SAMPLE)
    assert "clean sample PDF text" in result.text
    assert result.confidence > 0.8
    assert result.low_confidence is False


def test_pdf_extractor_missing_file():
    """Test PDF extractor handles missing file gracefully without crashing."""
    result = asyncio.run(extract_pdf("non_existent_file.pdf"))
    assert isinstance(result, ExtractionResult)
    assert result.low_confidence is True
    assert result.confidence == 0.0
    assert any("error" in w.lower() for w in result.warnings)


def test_image_extractor_sample():
    """Test image extractor with sample image and mock Gemini client."""
    assert IMAGE_SAMPLE.exists(), f"Sample image missing at {IMAGE_SAMPLE}"
    mock_client = DummyGeminiClient(return_text="Fallback vision image text")
    result = asyncio.run(
        extract_image(str(IMAGE_SAMPLE), gemini_client=mock_client)
    )

    assert isinstance(result, ExtractionResult)
    assert result.modality == "image"
    assert result.source_file == str(IMAGE_SAMPLE)


def test_image_extractor_fallback_retry():
    """Test that image extractor handles Gemini vision fallback failures gracefully."""
    failing_client = DummyGeminiClient(should_fail=True)
    result = asyncio.run(
        extract_image(str(IMAGE_SAMPLE), gemini_client=failing_client)
    )

    assert isinstance(result, ExtractionResult)
    assert result.low_confidence is True
    assert result.modality == "image"
    assert any("failed after 1 retry" in w for w in result.warnings)


def test_audio_extractor_sample():
    """Test audio extractor with sample WAV audio file and mock Gemini client."""
    assert AUDIO_SAMPLE.exists(), f"Sample audio missing at {AUDIO_SAMPLE}"
    mock_client = DummyGeminiClient(return_text="Cleaned transcript audio text.")
    result = asyncio.run(
        extract_audio(str(AUDIO_SAMPLE), gemini_client=mock_client, model_size="tiny")
    )

    assert isinstance(result, ExtractionResult)
    assert result.modality == "audio"
    assert any("audio_duration:" in w for w in result.warnings)
