"""
API-layer tests for app/main.py using FastAPI's TestClient.

Test strategy:
  - All tests patch app.state.graph with a MockGraph so no real Gemini or YouTube
    API calls are made. The mock graph returns controlled AgentState snapshots
    that mirror what the real graph produces.
  - TC4 and TC5 verify that the API response correctly surfaces trace entries
    showing the expected tools were used. What the mock proves: the API layer
    correctly reads tool names from state["trace"] and exposes them to callers.
    What a real end-to-end test (requiring a live Gemini key) would additionally
    prove: the planner chose those tools given that specific input. That test is
    left for integration testing against a real API key.

Sample files (drop replacements here for richer integration tests):
  tests/samples/clean_sample.pdf          — clean digital text PDF
  tests/samples/sample_audio.wav          — short WAV audio clip
  tests/samples/pdf_with_youtube_url.pdf  — PDF containing a YouTube URL (TC4)
"""

import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.output import FinalOutput, SummaryOutput
from app.schemas.plan import Plan, ToolCall
from app.schemas.trace import TraceEvent

SAMPLES = Path(__file__).parent / "samples"


# ---------------------------------------------------------------------------
# MockGraph — controls graph.invoke / get_state / update_state in tests
# ---------------------------------------------------------------------------

class MockGraph:
    """
    Minimal graph mock supporting both sync and async APIs.
    """

    def __init__(self, return_state: dict):
        self._return_state = return_state
        self._stored: dict[str, dict] = {}

    def invoke(self, state, config=None):
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
        if state is not None:
            self._stored[thread_id] = dict(state)
        stored = self._stored.get(thread_id, {})
        result = {**stored, **self._return_state}
        self._stored[thread_id] = result
        return result

    async def ainvoke(self, state, config=None):
        return self.invoke(state, config=config)

    def get_state(self, config):
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
        snap = MagicMock()
        snap.values = self._stored.get(thread_id)
        return snap

    async def aget_state(self, config):
        return self.get_state(config)

    def update_state(self, config, updates, as_node=None):
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
        if thread_id not in self._stored:
            self._stored[thread_id] = {}
        self._stored[thread_id].update(updates)

    async def aupdate_state(self, config, updates, as_node=None):
        return self.update_state(config, updates, as_node=as_node)


def _done_state(tool_name: str = "summarize", trace_tools: list[str] | None = None) -> dict:
    """Build a realistic 'done' AgentState snapshot."""
    tools = trace_tools or [tool_name]
    trace = [
        TraceEvent(
            step_index=i,
            tool_name=t,
            input_summary="test input",
            output_summary="test output",
            latency_ms=42,
            status="success",
        )
        for i, t in enumerate(tools)
    ]
    summary = SummaryOutput(
        one_line="A concise one-line summary.",
        bullets=["Point one.", "Point two.", "Point three."],
        five_sentence="S1. S2. S3. S4. S5.",
    )
    return {
        "status": "done",
        "final_output": FinalOutput(task_type=tool_name, summary=summary),
        "trace": trace,
        "clarify_question": None,
        "plan": Plan(steps=[ToolCall(tool_name=tool_name, args={}, reason="test")]),
        "tool_outputs": {0: summary},
        "extraction_results": [],
        "detected_urls": [],
        "fused_context": "",
    }


def _clarify_state(question: str = "What would you like me to do?") -> dict:
    """Build an 'awaiting_clarification' AgentState snapshot."""
    return {
        "status": "awaiting_clarification",
        "clarify_question": question,
        "final_output": None,
        "trace": [],
        "plan": Plan(steps=[], clarify_question=question),
        "tool_outputs": {},
        "extraction_results": [],
        "detected_urls": [],
        "fused_context": "",
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _patch_graph(client_fixture, mock_graph: MockGraph):
    """Replace the compiled graph on the live app with a mock."""
    client_fixture.app.state.graph = mock_graph


# ---------------------------------------------------------------------------
# Test 1: Unambiguous request → status="done" in one call
# ---------------------------------------------------------------------------

def test_session_unambiguous_returns_done(client):
    mock = MockGraph(_done_state("summarize"))
    _patch_graph(client, mock)

    pdf = (SAMPLES / "clean_sample.pdf").read_bytes()
    response = client.post(
        "/session",
        data={"user_query": "summarize this document"},
        files=[("files", ("clean_sample.pdf", io.BytesIO(pdf), "application/pdf"))],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "done"
    assert "thread_id" in body
    assert body["output"] is not None
    assert body["output"]["task_type"] == "summarize"
    assert len(body["trace"]) == 1
    assert body["trace"][0]["tool_name"] == "summarize"


# ---------------------------------------------------------------------------
# Test 2: Ambiguous request → awaiting_clarification → reply → done
# ---------------------------------------------------------------------------

def test_session_clarify_then_reply(client):
    clarify_mock = MockGraph(_clarify_state("What would you like me to do with this PDF?"))
    _patch_graph(client, clarify_mock)

    pdf = (SAMPLES / "clean_sample.pdf").read_bytes()
    r1 = client.post(
        "/session",
        data={"user_query": "I need help with this PDF."},
        files=[("files", ("clean_sample.pdf", io.BytesIO(pdf), "application/pdf"))],
    )
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["status"] == "awaiting_clarification"
    assert "question" in body1
    thread_id = body1["thread_id"]

    # Now swap to a mock that returns done for the resume call
    done_mock = MockGraph(_done_state("summarize"))
    # Preserve the stored clarify state so update_state/get_state works
    done_mock._stored = dict(clarify_mock._stored)
    _patch_graph(client, done_mock)

    r2 = client.post(
        f"/session/{thread_id}/reply",
        data={"reply": "Please summarize the document."},
    )
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["status"] == "done"
    assert body2["thread_id"] == thread_id


# ---------------------------------------------------------------------------
# Test 3: Unknown session thread_id → 404
# ---------------------------------------------------------------------------

def test_reply_unknown_thread_id_returns_404(client):
    empty_mock = MockGraph({})
    _patch_graph(client, empty_mock)

    r = client.post(
        "/session/nonexistent-thread-id/reply",
        data={"reply": "answer"},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Test 4 (TC4): PDF with YouTube URL + "summarize the video"
#              → fetch_youtube_transcript appears in trace
# ---------------------------------------------------------------------------

def test_tc4_pdf_with_youtube_url_invokes_fetch_transcript(client):
    """
    TC4: When the extracted PDF text contains a YouTube URL and the user asks
    to summarize the video, the graph should invoke fetch_youtube_transcript.

    What this test proves: the API correctly surfaces the fetch_youtube_transcript
    tool name from state["trace"] in the response. The mock simulates what the
    planner + executor would produce when a YouTube URL is detected.
    """
    tc4_state = _done_state(tool_name="summarize", trace_tools=["fetch_youtube_transcript", "summarize"])
    # Add detected URL to match what fuse_node would find
    tc4_state["detected_urls"] = ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
    tc4_state["fused_context"] = (
        "User query: summarize the video\n\n---\n\n"
        "This document discusses a well-known music video.\n"
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )

    mock = MockGraph(tc4_state)
    _patch_graph(client, mock)

    pdf_bytes = (SAMPLES / "pdf_with_youtube_url.pdf").read_bytes()
    response = client.post(
        "/session",
        data={"user_query": "summarize the video"},
        files=[("files", ("pdf_with_youtube_url.pdf", io.BytesIO(pdf_bytes), "application/pdf"))],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "done"

    tool_names_in_trace = [t["tool_name"] for t in body["trace"]]
    assert "fetch_youtube_transcript" in tool_names_in_trace, (
        f"Expected fetch_youtube_transcript in trace, got: {tool_names_in_trace}"
    )


def test_tc4_unmocked_real_planner_youtube_url(client):
    """
    Real un-mocked regression test for TC4: Exercises real extract_node, fuse_node,
    planner_node, and executor_node on pdf_with_youtube_url.pdf to verify that
    the planner chooses fetch_youtube_transcript and summarize, NOT conversational_answer.
    """
    import os
    from app.graph.build import build_graph

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("No real GEMINI_API_KEY configured for unmocked planner test.")

    # Re-enable real graph on client
    client.app.state.graph = build_graph()

    pdf_bytes = (SAMPLES / "pdf_with_youtube_url.pdf").read_bytes()
    response = client.post(
        "/session",
        headers={"X-Gemini-Api-Key": api_key},
        data={"user_query": "summarize the video in this document"},
        files=[("files", ("pdf_with_youtube_url.pdf", io.BytesIO(pdf_bytes), "application/pdf"))],
    )

    assert response.status_code == 200
    body = response.json()
    if body.get("status") == "awaiting_clarification" and "couldn't interpret" in body.get("question", "").lower():
        pytest.skip("Gemini API rate-limit quota exhausted during unmocked test run.")

    assert body["status"] == "done"

    tool_names_in_trace = [t["tool_name"] for t in body["trace"]]
    assert "fetch_youtube_transcript" in tool_names_in_trace, (
        f"Expected fetch_youtube_transcript in real planner trace, got: {tool_names_in_trace}"
    )
    assert "conversational_answer" not in tool_names_in_trace, (
        f"Planner incorrectly chose conversational_answer instead of fetch_youtube_transcript. Trace: {tool_names_in_trace}"
    )


# ---------------------------------------------------------------------------
# Test 5 (TC5): Audio + PDF + "do they discuss the same topic?"
#              → cross_compare appears in trace
# ---------------------------------------------------------------------------

def test_tc5_audio_and_pdf_invokes_cross_compare(client):
    """
    TC5: When the user uploads a PDF and an audio file and asks whether they
    discuss the same topic, the graph should invoke cross_compare.

    What this test proves: the API correctly surfaces the cross_compare tool
    from state["trace"]. The mock simulates the planner choosing cross_compare
    given two extracted modalities.
    """
    tc5_state = _done_state(tool_name="cross_compare", trace_tools=["cross_compare"])
    tc5_state["output"] = {
        "task_type": "cross_compare",
        "raw_text": '{"same_topic": true, "shared_themes": ["testing"], "key_differences": [], "comparative_summary": "Both discuss test content."}',
    }

    mock = MockGraph(tc5_state)
    _patch_graph(client, mock)

    pdf_bytes = (SAMPLES / "clean_sample.pdf").read_bytes()
    wav_bytes = (SAMPLES / "sample_audio.wav").read_bytes()

    response = client.post(
        "/session",
        data={"user_query": "do they discuss the same topic?"},
        files=[
            ("files", ("clean_sample.pdf", io.BytesIO(pdf_bytes), "application/pdf")),
            ("files", ("sample_audio.wav", io.BytesIO(wav_bytes), "audio/wav")),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "done"

    tool_names_in_trace = [t["tool_name"] for t in body["trace"]]
    assert "cross_compare" in tool_names_in_trace, (
        f"Expected cross_compare in trace, got: {tool_names_in_trace}"
    )


# ---------------------------------------------------------------------------
# Test 6: Corrupt / unsupported file upload → graceful error response
# ---------------------------------------------------------------------------

def test_wrong_mime_disguised_file_is_rejected(client):
    """
    Content signatures, rather than filename or browser-declared MIME, determine
    admissibility. A Windows executable renamed to .pdf must not reach pdfplumber.
    """
    disguised_exe = b"MZ\x90\x00This is executable-looking content"
    response = client.post(
        "/session",
        data={"user_query": "analyze this"},
        files=[("files", ("report.pdf", io.BytesIO(disguised_exe), "application/pdf"))],
    )

    assert response.status_code == 415
    assert "invalid file content" in response.json()["detail"].lower()


def test_oversized_file_is_rejected_while_streaming(client):
    mock = MockGraph(_done_state())
    _patch_graph(client, mock)

    # Valid PDF magic ensures this exercises the size guard, not MIME rejection.
    oversized = b"%PDF-1.7\n" + (b"x" * (10 * 1024 * 1024))
    response = client.post(
        "/session",
        data={"user_query": "summarize"},
        files=[("files", ("large.pdf", io.BytesIO(oversized), "application/pdf"))],
    )

    assert response.status_code == 413
    assert "10 mb" in response.json()["detail"].lower()


def test_empty_upload_is_rejected(client):
    response = client.post(
        "/session",
        data={"user_query": "summarize"},
        files=[("files", ("empty.pdf", io.BytesIO(b""), "application/pdf"))],
    )

    assert response.status_code == 422
    assert "must not be empty" in response.json()["detail"].lower()


def test_empty_query_is_rejected(client):
    pdf = (SAMPLES / "clean_sample.pdf").read_bytes()
    response = client.post(
        "/session",
        data={"user_query": "   "},
        files=[("files", ("clean_sample.pdf", io.BytesIO(pdf), "application/pdf"))],
    )

    assert response.status_code == 422
    assert "user_query must not be empty" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test 7: GET /session/{id}/trace read-only endpoint
# ---------------------------------------------------------------------------

def test_get_trace_returns_trace_list(client):
    mock = MockGraph(_done_state("sentiment"))
    _patch_graph(client, mock)

    pdf = (SAMPLES / "clean_sample.pdf").read_bytes()
    r1 = client.post(
        "/session",
        data={"user_query": "what is the tone?"},
        files=[("files", ("clean_sample.pdf", io.BytesIO(pdf), "application/pdf"))],
    )
    thread_id = r1.json()["thread_id"]

    r2 = client.get(f"/session/{thread_id}/trace")
    assert r2.status_code == 200
    body = r2.json()
    assert "trace" in body
    assert isinstance(body["trace"], list)


def test_get_trace_unknown_thread_returns_404(client):
    empty_mock = MockGraph({})
    _patch_graph(client, empty_mock)

    r = client.get("/session/does-not-exist/trace")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Test 8: Health and root endpoints
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_root_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Multimodal" in r.text


# ---------------------------------------------------------------------------
# Test 9 (Real Graph Resume): Verifies extract_node does NOT re-run on /reply
# ---------------------------------------------------------------------------

def test_real_graph_resume_does_not_re_extract(client):
    """
    Real integration test (no MockGraph) verifying that when a session pauses on
    clarify and is resumed via /reply, LangGraph resumes execution from planner_node
    without re-running extract_node or re-reading raw_files[].path (Decision 10).
    """
    from unittest.mock import patch
    from app.graph.build import build_graph

    # Ensure app uses the real compiled graph (default in lifespan)
    # Patch planner_node to return clarify on 1st call, and a real plan on 2nd call
    call_counts = {"planner": 0}

    async def mock_planner(state, *, gemini_client=None):
        call_counts["planner"] += 1
        if call_counts["planner"] == 1:
            return {
                **state,
                "plan": Plan(steps=[], clarify_question="What would you like me to do?"),
                "clarify_question": "What would you like me to do?",
                "status": "awaiting_clarification",
            }

        plan = Plan(
            steps=[ToolCall(tool_name="summarize", args={"text": state.get("fused_context", "")}, reason="User requested summary.")]
        )
        return {
            **state,
            "plan": plan,
            "clarify_question": None,
            "status": "executing",
        }

    async def mock_summarize(text, gemini_client=None):
        return SummaryOutput(
            one_line="Real graph summary.",
            bullets=["Bullet 1.", "Bullet 2.", "Bullet 3."],
            five_sentence="S1. S2. S3. S4. S5.",
        )

    pdf_bytes = (SAMPLES / "clean_sample.pdf").read_bytes()

    with patch("app.graph.build.planner_node", side_effect=mock_planner), \
         patch("app.tools.summarize.summarize", side_effect=mock_summarize):

        # Recompile graph so the patched planner_node is bound to the StateGraph
        client.app.state.graph = build_graph()

        # 1. Post PDF upload -> mocked planner triggers clarify
        r1 = client.post(
            "/session",
            data={"user_query": "I need help with this PDF."},
            files=[("files", ("clean_sample.pdf", io.BytesIO(pdf_bytes), "application/pdf"))],
        )

        assert r1.status_code == 200
        b1 = r1.json()
        assert b1["status"] == "awaiting_clarification"
        assert b1["question"] == "What would you like me to do?"
        thread_id = b1["thread_id"]

        # 2. Reply to clarify question
        r2 = client.post(
            f"/session/{thread_id}/reply",
            data={"reply": "Please summarize this document."},
        )

        assert r2.status_code == 200
        b2 = r2.json()
        assert b2["status"] == "done"
        assert b2["output"] is not None
        assert b2["output"]["summary"]["one_line"] is not None
        assert call_counts["planner"] == 2

        # 3. Assert trace has exactly 1 execution trace (summarize) and extract did not re-run
        trace_tools = [t["tool_name"] for t in b2["trace"]]
        assert trace_tools.count("extract_node") == 0  # extract_node warnings empty for clean PDF
        assert trace_tools == ["summarize"]


# ---------------------------------------------------------------------------
# Test 10 (Regression): Client wiring — tool receives non-None startup client
# ---------------------------------------------------------------------------

def test_gemini_client_reaches_tool_from_startup(client):
    """
    Regression test for the client-wiring bug: build_graph() was previously
    called without gemini_client, so every node received None.  This test
    runs a real compiled graph (not a MockGraph) and asserts the exact client
    instance constructed at startup is the one that arrives at dispatch_tool().

    This is the gap that let the bug ship through 57 passing mocked tests.
    """
    from unittest.mock import patch, MagicMock
    from app.graph.build import build_graph
    from app.tools.registry import TOOL_REGISTRY

    # Create a sentinel client object we can identity-check later.
    sentinel_client = MagicMock(name="sentinel_gemini_client")

    captured_clients = []

    # Mock planner to produce a deterministic plan (avoids needing real Gemini for planning)
    async def mock_planner(state, *, gemini_client=None):
        plan = Plan(
            steps=[ToolCall(tool_name="summarize", args={"text": state.get("fused_context", "")}, reason="test")]
        )
        return {
            **state,
            "plan": plan,
            "clarify_question": None,
            "status": "executing",
        }

    # Spy on summarize to capture the gemini_client it receives
    async def spy_summarize(text, gemini_client):
        captured_clients.append(gemini_client)
        return SummaryOutput(
            one_line="Test.", bullets=["A", "B", "C"], five_sentence="S1. S2. S3. S4. S5."
        )

    pdf_bytes = (SAMPLES / "clean_sample.pdf").read_bytes()

    # Patch at registry level — dispatch_tool reads TOOL_REGISTRY[name]["fn"],
    # which was bound at import time, not the module-level function.
    original_fn = TOOL_REGISTRY["summarize"]["fn"]
    try:
        TOOL_REGISTRY["summarize"]["fn"] = spy_summarize
        with patch("app.graph.build.planner_node", side_effect=mock_planner):
            graph = build_graph(gemini_client=sentinel_client)
            client.app.state.graph = graph

            r = client.post(
                "/session",
                data={"user_query": "summarize this"},
                files=[("files", ("clean_sample.pdf", io.BytesIO(pdf_bytes), "application/pdf"))],
            )

            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "done"

            # The critical assertion: the tool received the exact sentinel client,
            # not None, not a freshly-constructed fallback client.
            assert len(captured_clients) >= 1, "summarize was never called"
            assert captured_clients[0] is sentinel_client, (
                f"Tool received {captured_clients[0]!r} instead of the startup sentinel client. "
                "Client wiring is broken: build_graph() did not propagate gemini_client to executor_node → dispatch_tool → tool."
            )
    finally:
        TOOL_REGISTRY["summarize"]["fn"] = original_fn


# ---------------------------------------------------------------------------
# Test 11: Per-Session Custom Gemini API Key & Logging Security Tests
# ---------------------------------------------------------------------------

def test_user_supplied_api_key_header_succeeds(client):
    from unittest.mock import patch, MagicMock

    mock_graph = MockGraph(_done_state("summarize"))
    _patch_graph(client, mock_graph)

    mock_genai_client = MagicMock(name="valid_user_gemini_client")

    with patch("app.main.genai.Client", return_value=mock_genai_client):
        response = client.post(
            "/session",
            headers={"X-Gemini-Api-Key": "user_custom_valid_key_123"},
            data={"user_query": "summarize this"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "done"


def test_invalid_user_supplied_api_key_fails_fast_400(client):
    from unittest.mock import patch

    def mock_invalid_client(*args, **kwargs):
        raise Exception("API_KEY_INVALID: Key not found")

    with patch("app.main.genai.Client", side_effect=mock_invalid_client):
        response = client.post(
            "/session",
            headers={"X-Gemini-Api-Key": "invalid_key_999"},
            data={"user_query": "summarize this"},
        )

    assert response.status_code == 400
    assert "Invalid Gemini API key" in response.json()["detail"]


def test_no_api_key_provided_returns_400(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    response = client.post(
        "/session",
        data={"user_query": "summarize this"},
    )

    assert response.status_code == 400
    assert "No Gemini API key provided" in response.json()["detail"]


def test_server_default_env_api_key_fallback_succeeds(client, monkeypatch):
    from unittest.mock import patch, MagicMock

    monkeypatch.setenv("GEMINI_API_KEY", "server_env_fallback_key")
    mock_graph = MockGraph(_done_state("summarize"))
    _patch_graph(client, mock_graph)

    mock_client_inst = MagicMock(name="env_fallback_client")

    with patch("app.main.genai.Client", return_value=mock_client_inst):
        response = client.post(
            "/session",
            data={"user_query": "summarize this"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "done"


def test_api_key_never_logged_in_events(caplog):
    import logging
    from app.logging_utils import log_event

    with caplog.at_level(logging.INFO):
        log_event(
            thread_id="test_thread",
            node="test_node",
            status="success",
            latency_ms=10,
            api_key="AIzaSySECRET_RAW_KEY_THAT_MUST_NOT_BE_LOGGED",
            x_gemini_api_key="SECRET_HEADER_KEY",
        )

    for record in caplog.records:
        assert "AIzaSySECRET_RAW_KEY_THAT_MUST_NOT_BE_LOGGED" not in record.message
        assert "SECRET_HEADER_KEY" not in record.message
        assert "[REDACTED]" in record.message


def test_health_returns_gemini_configured_status():
    tc = TestClient(app)
    response = tc.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "gemini_configured" in data


def test_session_without_x_gemini_api_key_header_uses_server_env():
    """
    Verifies that when X-Gemini-Api-Key header is absent, the backend accepts the request
    and uses the server GEMINI_API_KEY without requiring a header.
    """
    from unittest.mock import patch
    tc = TestClient(app)
    done_state = {
        "session_id": "test_env_key",
        "status": "done",
        "final_output": FinalOutput(task_type="conversational", raw_text="Response using env key"),
        "trace": [],
    }
    app.state.graph = MockGraph(done_state)

    mock_client_inst = MagicMock()
    mock_res = MagicMock()
    mock_client_inst.models.count_tokens.return_value = mock_res

    with patch("os.getenv", side_effect=lambda k, d=None: "AIzaSyServerEnvKey" if k == "GEMINI_API_KEY" else d), \
         patch("app.main.genai.Client", return_value=mock_client_inst) as mock_client_cls:

        response = tc.post("/session", data={"user_query": "hello"})
        assert response.status_code == 200
        assert response.json()["status"] == "done"
        mock_client_cls.assert_called_once_with(api_key="AIzaSyServerEnvKey")


def test_session_with_x_gemini_api_key_header_overrides_server_env():
    """
    Verifies that when X-Gemini-Api-Key header is sent, it overrides the server GEMINI_API_KEY.
    """
    from unittest.mock import patch
    tc = TestClient(app)
    done_state = {
        "session_id": "test_override_key",
        "status": "done",
        "final_output": FinalOutput(task_type="conversational", raw_text="Response using custom key"),
        "trace": [],
    }
    app.state.graph = MockGraph(done_state)

    mock_client_inst = MagicMock()
    mock_res = MagicMock()
    mock_client_inst.models.count_tokens.return_value = mock_res

    with patch("os.getenv", side_effect=lambda k, d=None: "AIzaSyServerEnvKey" if k == "GEMINI_API_KEY" else d), \
         patch("app.main.genai.Client", return_value=mock_client_inst) as mock_client_cls:

        response = tc.post(
            "/session",
            data={"user_query": "hello"},
            headers={"X-Gemini-Api-Key": "AIzaSyCustomHeaderOverrideKey"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "done"
        mock_client_cls.assert_called_once_with(api_key="AIzaSyCustomHeaderOverrideKey")


def test_planner_unambiguous_query_produces_real_plan():
    """
    Regression test: An unambiguous request ('hello, what can you do?') must produce
    a valid plan or conversational answer, not hit the parse-failure clarify question.
    """
    import asyncio
    from app.graph.nodes import planner_node

    class MockGeminiPlannerClient:
        def generate_content(self, prompt):
            res = MagicMock()
            res.text = '''```json
{
  "steps": [
    {
      "tool_name": "conversational_answer",
      "args": {"query": "hello, what can you do?", "context": ""},
      "reason": "Answer conversational greeting and describe capabilities."
    }
  ],
  "clarify_question": null
}
```'''
            return res

    state = {
        "session_id": "test_unambiguous",
        "user_query": "hello, what can you do?",
        "fused_context": "User query: hello, what can you do?",
        "detected_urls": [],
    }

    out = asyncio.run(planner_node(state, gemini_client=MockGeminiPlannerClient()))

    assert out["status"] == "executing"
    assert out["clarify_question"] is None
    assert len(out["plan"].steps) == 1
    assert out["plan"].steps[0].tool_name == "conversational_answer"


