import json
from pathlib import Path
from typing import Any

from app.schemas.output import SentimentOutput

_PROMPT = Path(__file__).resolve().parent.parent.parent / "prompts" / "sentiment_v1.txt"


def _load_prompt(text: str) -> str:
    return _PROMPT.read_text(encoding="utf-8").format(text=text)


async def sentiment(text: str, gemini_client: Any) -> SentimentOutput:
    """Classify text sentiment as positive, negative, or neutral.

    Makes up to 2 attempts (1 Reflexion retry) before returning a
    neutral/zero-confidence fallback so the formatter never receives None.
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
                raw = res.text if hasattr(res, "text") else str(res)


            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:]).rstrip("`").strip()

            data = json.loads(raw)
            return SentimentOutput(**data)

        except Exception as exc:
            if attempt == 1:
                from app.logging_utils import is_rate_limit_error
                if is_rate_limit_error(exc):
                    return SentimentOutput(
                        label="neutral",
                        confidence=0.0,
                        justification="The AI service is temporarily rate-limited (429 API quota limit reached). Please wait a minute and try again.",
                    )
                return SentimentOutput(
                    label="neutral",
                    confidence=0.0,
                    justification="Sentiment analysis failed after one retry.",
                )

    return SentimentOutput(
        label="neutral",
        confidence=0.0,
        justification="Sentiment analysis failed after one retry.",
    )
