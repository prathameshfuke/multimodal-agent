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


def _call_gemini(prompt: str, gemini_client: Any) -> str:
    if hasattr(gemini_client, "aio"):
        raise TypeError("Use async version for async clients")
    res = gemini_client.generate_content(prompt)
    return res.text if hasattr(res, "text") else str(res)


async def summarize(text: str, gemini_client: Any) -> SummaryOutput:
    prompt = _load_prompt(text)

    for attempt in range(2):
        try:
            if hasattr(gemini_client, "aio") and hasattr(gemini_client.aio, "models"):
                response = await gemini_client.aio.models.generate_content(
                    model="gemini-2.5-flash", contents=prompt
                )
                raw = (response.text or "").strip()
            else:
                raw = _call_gemini(prompt, gemini_client)


            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
                raw = raw.rstrip("`").strip()

            data = json.loads(raw)
            data["bullets"] = _enforce_three_bullets(data.get("bullets", []))
            return SummaryOutput(**data)

        except Exception:
            if attempt == 1:
                return SummaryOutput(
                    one_line="Summary unavailable.",
                    bullets=["...", "...", "..."],
                    five_sentence="The summarization step failed after one retry.",
                )

    return SummaryOutput(
        one_line="Summary unavailable.",
        bullets=["...", "...", "..."],
        five_sentence="The summarization step failed after one retry.",
    )
