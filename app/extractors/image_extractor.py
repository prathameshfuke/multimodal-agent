import asyncio
import os
from typing import Any
import pytesseract
from PIL import Image

from app.schemas.extraction import ExtractionResult

# Allow TESSERACT_CMD env var to override the default system PATH lookup.
# Windows devs: set TESSERACT_CMD=C:\path\to\tesseract.exe
# Docker/Linux: tesseract-ocr is installed via apt-get and available on PATH.
_tesseract_cmd = os.getenv("TESSERACT_CMD")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd


async def _call_gemini_vision_with_retry(
    image: Image.Image,
    gemini_client: Any = None,
    prompt: str = "Transcribe all visible text from this image accurately.",
) -> tuple[str, bool, bool]:
    """Wraps Gemini vision API call with 1 Reflexion-style retry. Returns (text, success, is_rate_limit)."""
    if gemini_client is None:
        return "", False, False

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            if hasattr(gemini_client, "aio") and hasattr(gemini_client.aio, "models"):
                response = await gemini_client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[image, prompt],
                )
                return (response.text or "").strip(), True, False
            elif hasattr(gemini_client, "generate_content"):
                res = gemini_client.generate_content([image, prompt])
                text = res.text if hasattr(res, "text") else str(res)
                return text.strip(), True, False
            else:
                return "", False, False
        except Exception as exc:
            last_exc = exc
            if attempt == 1:
                from app.logging_utils import is_rate_limit_error
                return "", False, is_rate_limit_error(last_exc)
            await asyncio.sleep(0.5)
    return "", False, False


async def extract_image(file_path: str, gemini_client: Any = None) -> ExtractionResult:
    warnings: list[str] = []

    try:
        image = Image.open(file_path)
        try:
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            conf_scores = [float(c) for c in ocr_data.get("conf", []) if int(c) >= 0]
            mean_conf = (sum(conf_scores) / len(conf_scores) / 100.0) if conf_scores else 0.0
            ocr_text = pytesseract.image_to_string(image).strip()
        except Exception as ocr_err:
            mean_conf = 0.0
            ocr_text = ""
            warnings.append(f"Local OCR unavailable ({ocr_err}); defaulting to Gemini vision fallback.")

        if mean_conf >= 0.60 and len(ocr_text) >= 20:
            return ExtractionResult(
                source_file=file_path,
                modality="image",
                text=ocr_text,
                confidence=round(mean_conf, 2),
                low_confidence=False,
                warnings=warnings,
            )

        # Trigger Gemini vision fallback if confidence < 0.60 or text < 20 chars
        warnings.append(
            f"Low local OCR confidence ({mean_conf:.2f}) or short text length ({len(ocr_text)} chars); triggering Gemini vision fallback."
        )

        vision_text, success, is_rate_limit = await _call_gemini_vision_with_retry(image, gemini_client)
        if success and vision_text:
            return ExtractionResult(
                source_file=file_path,
                modality="image",
                text=vision_text,
                confidence=0.85,
                low_confidence=True,
                warnings=warnings,
            )

        if is_rate_limit:
            warnings.append("Gemini vision fallback rate-limited (429 API quota limit reached). Returning low-confidence local OCR text.")
        else:
            warnings.append("Gemini vision fallback failed after 1 retry. Returning low-confidence local OCR text.")

        return ExtractionResult(
            source_file=file_path,
            modality="image",
            text=ocr_text,
            confidence=round(mean_conf, 2),
            low_confidence=True,
            warnings=warnings,
        )

    except Exception as e:
        warnings.append(f"Image processing error: {str(e)}")
        return ExtractionResult(
            source_file=file_path,
            modality="image",
            text="",
            confidence=0.0,
            low_confidence=True,
            warnings=warnings,
        )
