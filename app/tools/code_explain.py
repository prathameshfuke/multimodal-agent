from pathlib import Path
from typing import Any

_PROMPT = Path(__file__).resolve().parent.parent.parent / "prompts" / "code_explain_v1.txt"


def _load_prompt(text: str) -> str:
    return _PROMPT.read_text(encoding="utf-8").format(text=text)


async def code_explain(text: str, gemini_client: Any) -> str:
    """Explain what a code snippet does, identify bugs, and state complexity.

    Makes up to 2 attempts (1 Reflexion retry) before returning a
    plain-text fallback string.
    """
    prompt = _load_prompt(text)

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
                return "Code explanation failed after one retry."

    return "Code explanation failed after one retry."
