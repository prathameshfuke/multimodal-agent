"""
Unit tests for the six tool functions.
Uses lightweight mock Gemini clients — no real API calls required.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

from app.tools.summarize import summarize
from app.tools.sentiment import sentiment
from app.tools.code_explain import code_explain
from app.tools.youtube_transcript import fetch_youtube_transcript
from app.tools.cross_compare import cross_compare
from app.tools.conversational import conversational_answer
from app.tools.registry import TOOL_REGISTRY, get_tool_schema_for_planner, dispatch_tool
from app.schemas.output import SummaryOutput, SentimentOutput


def _mock_client(response_text: str):
    client = MagicMock(spec=["generate_content"])
    response = MagicMock()
    response.text = response_text
    client.generate_content.return_value = response
    return client


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_returns_summary_output():
    payload = json.dumps({
        "one_line": "A quick overview.",
        "bullets": ["Point A", "Point B", "Point C"],
        "five_sentence": "One. Two. Three. Four. Five.",
    })
    client = _mock_client(payload)
    result = asyncio.run(summarize("some text", client))
    assert isinstance(result, SummaryOutput)
    assert len(result.bullets) == 3
    assert result.one_line == "A quick overview."


def test_summarize_enforces_three_bullets_when_four_returned():
    payload = json.dumps({
        "one_line": "Overview.",
        "bullets": ["A", "B", "C", "D"],  # 4 — should be truncated
        "five_sentence": "One. Two. Three. Four. Five.",
    })
    result = asyncio.run(summarize("text", _mock_client(payload)))
    assert len(result.bullets) == 3


def test_summarize_pads_bullets_when_two_returned():
    payload = json.dumps({
        "one_line": "Overview.",
        "bullets": ["A", "B"],  # 2 — should be padded
        "five_sentence": "One. Two. Three. Four. Five.",
    })
    result = asyncio.run(summarize("text", _mock_client(payload)))
    assert len(result.bullets) == 3
    assert result.bullets[2] == "..."


def test_summarize_fallback_on_llm_failure():
    client = MagicMock(spec=["generate_content"])
    client.generate_content.side_effect = RuntimeError("API error")
    result = asyncio.run(summarize("text", client))
    assert isinstance(result, SummaryOutput)
    assert result.one_line == "Summary unavailable."


# ---------------------------------------------------------------------------
# sentiment
# ---------------------------------------------------------------------------

def test_sentiment_returns_sentiment_output():
    payload = json.dumps({
        "label": "positive",
        "confidence": 0.91,
        "justification": "The text uses enthusiastic and affirming language.",
    })
    result = asyncio.run(sentiment("great product!", _mock_client(payload)))
    assert isinstance(result, SentimentOutput)
    assert result.label == "positive"
    assert result.confidence == 0.91


def test_sentiment_fallback_on_llm_failure():
    client = MagicMock(spec=["generate_content"])
    client.generate_content.side_effect = RuntimeError("fail")
    result = asyncio.run(sentiment("text", client))
    assert result.label == "neutral"
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# code_explain
# ---------------------------------------------------------------------------

def test_code_explain_returns_string():
    client = _mock_client("EXPLANATION\nDoes X.\n\nBUGS\nNone.\n\nCOMPLEXITY\nO(n).")
    result = asyncio.run(code_explain("def f(): pass", client))
    assert isinstance(result, str)
    assert len(result) > 0


def test_code_explain_fallback_on_failure():
    client = MagicMock(spec=["generate_content"])
    client.generate_content.side_effect = RuntimeError("fail")
    result = asyncio.run(code_explain("code", client))
    assert "failed" in result.lower()


# ---------------------------------------------------------------------------
# fetch_youtube_transcript
# ---------------------------------------------------------------------------

def test_youtube_transcript_invalid_url():
    result = asyncio.run(fetch_youtube_transcript("https://example.com/not-a-youtube-url"))
    assert "Could not extract" in result


def test_youtube_transcript_valid_id_format():
    # We can't make a real API call here; just verify the function doesn't raise
    # and returns a string when the API is unavailable or captions are disabled.
    result = asyncio.run(fetch_youtube_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
    assert isinstance(result, str)


def test_youtube_transcript_private_or_deleted_video_returns_fallback():
    with patch(
        "youtube_transcript_api.YouTubeTranscriptApi.fetch",
        side_effect=RuntimeError("private/deleted video response"),
    ):
        result = asyncio.run(
            fetch_youtube_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        )
    assert "Could not fetch transcript" in result
    assert "private/deleted" in result


# ---------------------------------------------------------------------------
# cross_compare
# ---------------------------------------------------------------------------

def test_cross_compare_returns_dict():
    payload = json.dumps({
        "same_topic": True,
        "shared_themes": ["theme1"],
        "key_differences": ["diff1"],
        "comparative_summary": "Both discuss X but differ on Y.",
    })
    result = asyncio.run(cross_compare("text A", "text B", _mock_client(payload)))
    assert isinstance(result, dict)
    assert "same_topic" in result
    assert "shared_themes" in result


def test_cross_compare_fallback_on_failure():
    client = MagicMock(spec=["generate_content"])
    client.generate_content.side_effect = RuntimeError("fail")
    result = asyncio.run(cross_compare("a", "b", client))
    assert result["same_topic"] is False
    assert "failed" in result["comparative_summary"].lower()


# ---------------------------------------------------------------------------
# conversational_answer
# ---------------------------------------------------------------------------

def test_conversational_answer_returns_string():
    client = _mock_client("The answer is 42.")
    result = asyncio.run(conversational_answer("What is the answer?", "context text", client))
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def test_tool_registry_has_six_tools():
    assert len(TOOL_REGISTRY) == 6


def test_get_tool_schema_for_planner_structure():
    schema = get_tool_schema_for_planner()
    assert len(schema) == 6
    for entry in schema:
        assert "name" in entry
        assert "description" in entry
        assert "args" in entry


def test_dispatch_tool_unknown_raises_key_error():
    try:
        asyncio.run(dispatch_tool("nonexistent_tool", {}))
        assert False, "Expected KeyError"
    except KeyError:
        pass


def test_dispatch_tool_youtube_no_gemini_client():
    # Should work without a gemini_client since fetch_youtube_transcript needs none
    result, succeeded = asyncio.run(
        dispatch_tool("fetch_youtube_transcript", {"url": "https://example.com/bad"})
    )
    assert isinstance(result, str)
    assert succeeded is False
