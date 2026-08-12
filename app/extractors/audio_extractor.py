import asyncio
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from app.schemas.extraction import ExtractionResult

PROMPT_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "prompts"
    / "audio_cleanup_v1.txt"
)


def _load_cleanup_prompt(raw_transcript: str) -> str:
    if PROMPT_FILE.exists():
        template = PROMPT_FILE.read_text(encoding="utf-8")
        return template.format(raw_transcript=raw_transcript)
    return f"Clean up surface errors without altering meaning: {raw_transcript}"


async def _call_gemini_cleanup_with_retry(
    prompt: str,
    gemini_client: Any = None,
) -> tuple[str, bool]:
    """Wrap Gemini LLM audio cleanup call with 1 Reflexion-style retry.

    Returns (cleaned_text, success). If gemini_client is None or every
    attempt fails, returns ("", False) so the caller falls back to the
    raw ASR transcript.
    """
    if gemini_client is None:
        return "", False

    for attempt in range(2):
        try:
            if hasattr(gemini_client, "aio") and hasattr(gemini_client.aio, "models"):
                response = await gemini_client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                return (response.text or "").strip(), True
            elif hasattr(gemini_client, "generate_content"):
                res = gemini_client.generate_content(prompt)
                text = res.text if hasattr(res, "text") else str(res)
                return text.strip(), True
            else:
                return "", False
        except Exception:
            if attempt == 1:
                return "", False
            await asyncio.sleep(0.5)
    return "", False


async def extract_audio(
    file_path: str,
    gemini_client: Any = None,
    model_size: str = "base",
) -> ExtractionResult:
    """Transcribe an audio file with faster-whisper and optionally clean the
    transcript with Gemini 2.5 Flash.

    Confidence is derived from the average log-probability across segments.
    The result is marked low_confidence if confidence < 0.60 or the
    transcript is shorter than 10 characters.
    """
    warnings: list[str] = []

    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(file_path, beam_size=5)

        segment_list = list(segments)
        raw_text = " ".join(
            seg.text.strip() for seg in segment_list if seg.text
        ).strip()

        duration = getattr(info, "duration", 0.0)
        warnings.append(f"audio_duration: {duration:.2f}s")

        avg_logprob = (
            sum(seg.avg_logprob for seg in segment_list) / len(segment_list)
            if segment_list
            else -1.0
        )
        # Map log-probability (typically -1 to 0) to a 0–1 confidence score.
        confidence = (
            max(0.0, min(1.0, 1.0 + (avg_logprob / 3.0))) if segment_list else 0.5
        )
        is_low_confidence = confidence < 0.60 or len(raw_text) < 10

        if not raw_text:
            warnings.append("Audio transcription yielded empty text.")
            return ExtractionResult(
                source_file=file_path,
                modality="audio",
                text="",
                confidence=0.0,
                low_confidence=True,
                warnings=warnings,
            )

        prompt = _load_cleanup_prompt(raw_text)
        cleaned_text, success = await _call_gemini_cleanup_with_retry(
            prompt, gemini_client
        )

        if success and cleaned_text:
            return ExtractionResult(
                source_file=file_path,
                modality="audio",
                text=cleaned_text,
                confidence=round(confidence, 2),
                low_confidence=is_low_confidence,
                warnings=warnings,
            )

        warnings.append(
            "Gemini audio cleanup pass failed after 1 retry. Returning raw ASR transcript."
        )
        return ExtractionResult(
            source_file=file_path,
            modality="audio",
            text=raw_text,
            confidence=round(confidence, 2),
            low_confidence=is_low_confidence,
            warnings=warnings,
        )

    except Exception as e:
        warnings.append(f"Audio processing error: {str(e)}")
        return ExtractionResult(
            source_file=file_path,
            modality="audio",
            text="",
            confidence=0.0,
            low_confidence=True,
            warnings=warnings,
        )
