"""
Tool registry: single source of truth for tool metadata and dispatch.

The planner LLM reads the output of get_tool_schema_for_planner() verbatim, so the
description and args_schema fields here are the canonical specification — not auto-derived
from signatures. See decisions.md Decision 7 for the rationale.
"""

import time
from typing import Any

from app.tools.summarize import summarize
from app.tools.sentiment import sentiment
from app.tools.code_explain import code_explain
from app.tools.youtube_transcript import fetch_youtube_transcript
from app.tools.cross_compare import cross_compare
from app.tools.conversational import conversational_answer
from app.logging_utils import log_event

TOOL_REGISTRY: dict[str, dict] = {
    "summarize": {
        "fn": summarize,
        "description": (
            "Summarise a block of text into a one-line summary, three key bullets, "
            "and a five-sentence paragraph. Use when the user asks to summarise, "
            "condense, or get an overview of a document or transcript."
        ),
        "args_schema": {"text": "str — the text to summarise"},
    },
    "sentiment": {
        "fn": sentiment,
        "description": (
            "Classify the overall sentiment of a text as positive, negative, or neutral, "
            "with a confidence score and a one-line justification. Use when the user asks "
            "about tone, opinion, or emotional content."
        ),
        "args_schema": {"text": "str — the text to analyse"},
    },
    "code_explain": {
        "fn": code_explain,
        "description": (
            "Explain what a code snippet does, list any bugs or issues, and state its "
            "time and space complexity. Use when the uploaded content contains source code "
            "or the user asks to explain, review, or analyse code."
        ),
        "args_schema": {"text": "str — the code to explain"},
    },
    "fetch_youtube_transcript": {
        "fn": fetch_youtube_transcript,
        "description": (
            "Fetch the caption transcript of a YouTube video given its URL. Returns the "
            "full transcript as plain text, or a clear fallback message if captions are "
            "unavailable. Use when a YouTube URL is detected and the user wants to work "
            "with the video content."
        ),
        "args_schema": {"url": "str — the full YouTube video URL"},
    },
    "cross_compare": {
        "fn": cross_compare,
        "description": (
            "Compare two texts and return their shared themes, key differences, and a "
            "comparative summary. Use when the user uploads two documents or asks to "
            "compare two pieces of content."
        ),
        "args_schema": {
            "text_a": "str — the first text",
            "text_b": "str — the second text",
        },
    },
    "conversational_answer": {
        "fn": conversational_answer,
        "description": (
            "Answer a conversational or factual question using the provided context. "
            "Use when the user asks a direct question that can be answered from the "
            "extracted document content without needing a specialised analysis tool."
        ),
        "args_schema": {
            "query": "str — the user's question",
            "context": "str — the relevant context to draw from",
        },
    },
}


def get_tool_schema_for_planner() -> list[dict]:
    """Returns the tool list in the format injected into planner_v1.txt {tool_schema}."""
    return [
        {
            "name": name,
            "description": entry["description"],
            "args": entry["args_schema"],
        }
        for name, entry in TOOL_REGISTRY.items()
    ]


async def dispatch_tool(
    tool_name: str,
    args: dict,
    gemini_client: Any = None,
    *,
    thread_id: str = "unknown",
) -> Any:
    """
    Central dispatch. Raises KeyError for unknown tool names so executor_node
    can record a clean failure rather than silently producing None.
    """
    if tool_name not in TOOL_REGISTRY:
        log_event(thread_id=thread_id, node="tool_dispatch", status="error", latency_ms=0, tool=tool_name)
        raise KeyError(f"Unknown tool: '{tool_name}'. Valid tools: {list(TOOL_REGISTRY)}")

    fn = TOOL_REGISTRY[tool_name]["fn"]
    started = time.monotonic()

    try:
        # Tools that don't need a gemini_client (fetch_youtube_transcript)
        if tool_name == "fetch_youtube_transcript":
            result = await fn(**args)
        else:
            result = await fn(**args, gemini_client=gemini_client)
        log_event(thread_id=thread_id, node="tool_dispatch", status="success", latency_ms=int((time.monotonic() - started) * 1000), tool=tool_name)
        return result
    except Exception as exc:
        log_event(thread_id=thread_id, node="tool_dispatch", status="error", latency_ms=int((time.monotonic() - started) * 1000), tool=tool_name, error=str(exc))
        raise
