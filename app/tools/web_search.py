import re
from typing import Any
import httpx

from app.logging_utils import is_rate_limit_error


async def _ddg_web_search(query: str) -> str:
    """Fetch live web search results via DuckDuckGo HTML API."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers=headers,
            follow_redirects=True,
        )
        titles = re.findall(r'<a class="result__a"[^>]*>(.*?)</a>', r.text, re.DOTALL)
        snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', r.text, re.DOTALL)

        results = []
        for title, snippet in zip(titles[:5], snippets[:5]):
            t_clean = re.sub(r"<[^>]+>", "", title).strip()
            s_clean = re.sub(r"<[^>]+>", "", snippet).strip()
            if t_clean and s_clean:
                results.append(f"Title: {t_clean}\nSnippet: {s_clean}")

        if results:
            return "\n\n---\n\n".join(results)
        return f"No search results found for query: '{query}'."


async def web_search(query: str, gemini_client: Any = None) -> str:
    """Search the web for up-to-date information on a given query.
    
    Uses Gemini Google Search Grounding when available, with a DuckDuckGo HTML search fallback.
    """
    if not query or not query.strip():
        return "Search query must not be empty."

    if gemini_client is not None:
        try:
            from google.genai import types
            config = types.GenerateContentConfig(tools=[{"google_search": {}}])
            prompt = f"Perform a web search and summarize current live search information for: {query}"
            
            if hasattr(gemini_client, "aio") and hasattr(gemini_client.aio, "models"):
                res = await gemini_client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=config,
                )
                if res and hasattr(res, "text") and res.text:
                    return res.text.strip()
            elif hasattr(gemini_client, "models"):
                res = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=config,
                )
                if res and hasattr(res, "text") and res.text:
                    return res.text.strip()
        except Exception as exc:
            if is_rate_limit_error(exc):
                return "The AI service is temporarily rate-limited (429 API quota limit reached). Please wait a minute and try again or supply your own key in Settings."

    try:
        return await _ddg_web_search(query)
    except Exception as e:
        return f"Web search failed: {str(e)}"
