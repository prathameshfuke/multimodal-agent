import json
from pathlib import Path
from typing import Any

from app.schemas.output import SummaryOutput

_PROMPT = Path(__file__).resolve().parent.parent.parent / "prompts" / "summarize_v1.txt"


def _load_prompt(text: str) -> str:
    return _PROMPT.read_text(encoding="utf-8").format(text=text)


def _enforce_three_bullets(bullets: list[str]) -> list[str]:
    """Truncate to 3 or pad with ellipsis — per Decision 3."""
    if len(bullets) > 3:
        return bullets[:3]
    while len(bullets) < 3:
        bullets.append("...")
    return bullets


async def summarize(text: str, gemini_client: Any) -> SummaryOutput:
    """Summarise text into a one-line summary, three bullets, and a paragraph.

    Makes up to 2 attempts (1 Reflexion retry) before returning a safe
    fallback SummaryOutput so the formatter never receives None.
    """
    prompt = _load_prompt(text)

    for attempt in range(2):
        try:
            if hasattr(gemini_client, "aio") and hasattr(gemini_client.aio, "models"):
                response = await gemini_client.aio.models.generate_content(
                    model="gemini-2.5-flash", contents=prompt
                )
                raw = (response.text or "").strip()
            else:
                res = gemini_client.generate_content(prompt)
                raw = (res.text if hasattr(res, "text") else str(res)).strip()

            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
                raw = raw.rstrip("`").strip()

            data = json.loads(raw)
            data["bullets"] = _enforce_three_bullets(data.get("bullets", []))
            return SummaryOutput(**data)

        except Exception as exc:
            if attempt == 1:
                from app.logging_utils import is_rate_limit_error
                if is_rate_limit_error(exc):
                    return SummaryOutput(
                        one_line="AI service rate-limited.",
                        bullets=["Quota limit reached", "Please wait a minute", "Or supply custom key in Settings"],
                        five_sentence="The AI service is temporarily rate-limited (429 API quota limit reached). Please wait a minute and try again or supply your own key in Settings.",
                    )
                return SummaryOutput(
                    one_line="Summary unavailable.",
                    bullets=["...", "...", "..."],
                    five_sentence="The summarization step failed after one retry.",
                )
