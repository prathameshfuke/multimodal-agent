import asyncio
from typing import Any
import pdfplumber
import pytesseract
from PIL import Image
from pathlib import Path

from app.schemas.extraction import ExtractionResult

# Auto-detect local project Tesseract binary if present in <project>/tesseract/
_PROJECT_TESSERACT = Path(__file__).resolve().parent.parent.parent / "tesseract" / "tesseract.exe"
if _PROJECT_TESSERACT.exists():
    pytesseract.pytesseract.tesseract_cmd = str(_PROJECT_TESSERACT)


async def _call_gemini_vision_with_retry(
    image: Image.Image,
    gemini_client: Any = None,
    prompt: str = "Transcribe all text from this scanned document page accurately.",
) -> tuple[str, bool]:
    """Wraps Gemini vision API call with 1 Reflexion-style retry."""
    for attempt in range(2):
        try:
            if gemini_client is not None:
                if hasattr(gemini_client, "aio") and hasattr(gemini_client.aio, "models"):
                    response = await gemini_client.aio.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[image, prompt],
                    )
                    return (response.text or "").strip(), True
                elif hasattr(gemini_client, "generate_content"):
                    res = gemini_client.generate_content([image, prompt])
                    text = res.text if hasattr(res, "text") else str(res)
                    return text.strip(), True
            else:
                from google import genai

                client = genai.Client()
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[image, prompt],
                )
                return (response.text or "").strip(), True
        except Exception:
            if attempt == 1:
                return "", False
            await asyncio.sleep(0.5)
    return "", False


async def extract_pdf(file_path: str, gemini_client: Any = None) -> ExtractionResult:
    warnings: list[str] = []
    page_texts: list[str] = []
    page_confidences: list[float] = []
    has_low_confidence_page = False

    try:
        with pdfplumber.open(file_path) as pdf:
            if not pdf.pages:
                warnings.append("PDF contains 0 pages.")
                return ExtractionResult(
                    source_file=file_path,
                    modality="pdf",
                    text="",
                    confidence=0.0,
                    low_confidence=True,
                    warnings=warnings,
                )

            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                text = (page.extract_text() or "").strip()

                if len(text) >= 20:
                    page_texts.append(text)
                    page_confidences.append(1.0)
                    continue

                page_img = page.to_image(resolution=150).original
                ocr_data = pytesseract.image_to_data(
                    page_img, output_type=pytesseract.Output.DICT
                )

                conf_scores = [
                    float(c) for c in ocr_data.get("conf", []) if int(c) >= 0
                ]
                mean_conf = (
                    (sum(conf_scores) / len(conf_scores) / 100.0)
                    if conf_scores
                    else 0.0
                )
                ocr_text = pytesseract.image_to_string(page_img).strip()

                if mean_conf >= 0.60 and len(ocr_text) >= 20:
                    page_texts.append(ocr_text)
                    page_confidences.append(mean_conf)
                    warnings.append(
                        f"Page {page_num}: Extracted via local Tesseract OCR (confidence {mean_conf:.2f})."
                    )
                else:
                    has_low_confidence_page = True
                    warnings.append(
                        f"Page {page_num}: Low local OCR confidence ({mean_conf:.2f}); triggering Gemini vision fallback."
                    )

                    vision_text, success = await _call_gemini_vision_with_retry(
                        page_img, gemini_client
                    )
                    if success and vision_text:
                        page_texts.append(vision_text)
                        page_confidences.append(0.85)
                    else:
                        page_texts.append(ocr_text)
                        page_confidences.append(mean_conf)
                        warnings.append(
                            f"Page {page_num}: Gemini vision fallback failed after 1 retry. Preserving local OCR text."
                        )

    except Exception as e:
        warnings.append(f"PDF processing error: {str(e)}")
        return ExtractionResult(
            source_file=file_path,
            modality="pdf",
            text="",
            confidence=0.0,
            low_confidence=True,
            warnings=warnings,
        )

    full_text = "\n\n".join(page_texts)
    overall_confidence = (
        (sum(page_confidences) / len(page_confidences))
        if page_confidences
        else 0.0
    )
    is_low_confidence = (
        has_low_confidence_page
        or (overall_confidence < 0.60)
        or (len(full_text) < 20)
    )

    return ExtractionResult(
        source_file=file_path,
        modality="pdf",
        text=full_text,
        confidence=round(overall_confidence, 2),
        low_confidence=is_low_confidence,
        warnings=warnings,
    )
