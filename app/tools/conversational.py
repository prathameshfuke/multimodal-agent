from pathlib import Path
from typing import Any

_PROMPT = Path(__file__).resolve().parent.parent.parent / "prompts" / "conversational_v1.txt"


def _load_prompt(query: str, context: str) -> str:
    return _PROMPT.read_text(encoding="utf-8").format(query=query, context=context)


async def conversational_answer(query: str, context: str, gemini_client: Any) -> str:
    prompt = _load_prompt(query, context)

    for attempt in range(2):
        try:
            if hasattr(gemini_client, "aio") and hasattr(gemini_client.aio, "models"):
                response = await gemini_client.aio.models.generate_content(
                    model="gemini-2.5-flash", contents=prompt
                )
                return (response.text or "").strip()
            else:
                res = gemini_client.generate_content(prompt)
                return (res.text if hasattr(res, "text") else str(res)).strip()

        except Exception:
            if attempt == 1:
                return "Conversational answer failed after one retry."

    return "Conversational answer failed after one retry."

