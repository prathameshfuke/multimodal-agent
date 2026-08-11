import json
from pathlib import Path
from typing import Any

_PROMPT = Path(__file__).resolve().parent.parent.parent / "prompts" / "cross_compare_v1.txt"


def _load_prompt(text_a: str, text_b: str) -> str:
    return _PROMPT.read_text(encoding="utf-8").format(text_a=text_a, text_b=text_b)


async def cross_compare(text_a: str, text_b: str, gemini_client: Any) -> dict:
    prompt = _load_prompt(text_a, text_b)

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
                raw = "\n".join(raw.split("\n")[1:]).rstrip("`").strip()

            return json.loads(raw)

        except Exception:
            if attempt == 1:
                return {
                    "same_topic": False,
                    "shared_themes": [],
                    "key_differences": [],
                    "comparative_summary": "Cross-comparison failed after one retry.",
                }

    return {
        "same_topic": False,
        "shared_themes": [],
        "key_differences": [],
        "comparative_summary": "Cross-comparison failed after one retry.",
    }
