"""
Unit tests for graph routing functions, fuse_node URL extraction,
formatter_node bullet enforcement, and MemorySaver checkpoint round-trip.
These test the plain Python logic without running the full compiled graph,
plus one test that exercises the checkpointer directly (Decision 9).
"""

import asyncio
from pathlib import Path

from app.graph.build import build_graph, route_after_plan, route_after_exec
from app.graph.nodes import extract_node, fuse_node, formatter_node
from app.schemas.output import FinalOutput, SummaryOutput
from app.schemas.plan import Plan, ToolCall


def _base_state(**overrides) -> dict:
    base = {
        "session_id": "test-session",
        "user_query": "",
        "raw_files": [],
        "extraction_results": [],
        "fused_context": "",
        "detected_urls": [],
        "plan": None,
        "trace": [],
        "clarify_question": None,
        "final_output": None,
        "tool_outputs": {},  # declared AgentState field, not a scratch key
        "status": "planning",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# route_after_plan
# ---------------------------------------------------------------------------

def test_route_after_plan_executes_when_ready():
    state = _base_state(status="executing")
    assert route_after_plan(state) == "execute"


def test_route_after_plan_clarifies_when_awaiting():
    state = _base_state(status="awaiting_clarification")
    assert route_after_plan(state) == "clarify"


def test_route_after_plan_planning_status_executes():
    state = _base_state(status="planning")
    assert route_after_plan(state) == "execute"


# ---------------------------------------------------------------------------
# route_after_exec
# ---------------------------------------------------------------------------

def test_route_after_exec_always_done():
    for status in ("done", "error", "executing", "partial"):
        assert route_after_exec(_base_state(status=status)) == "done"


# ---------------------------------------------------------------------------
# fuse_node — URL extraction
# ---------------------------------------------------------------------------

def test_fuse_node_extracts_urls_from_text():
    from app.schemas.extraction import ExtractionResult
    result = ExtractionResult(
        source_file="test.pdf",
        modality="pdf",
        text="Visit https://example.com and https://openai.com for more info.",
        confidence=0.9,
        low_confidence=False,
    )
    state = _base_state(
        user_query="check these out",
        extraction_results=[result],
    )
    out = asyncio.run(fuse_node(state))
    assert "https://example.com" in out["detected_urls"]
    assert "https://openai.com" in out["detected_urls"]


def test_fuse_node_deduplicates_urls():
    from app.schemas.extraction import ExtractionResult
    result = ExtractionResult(
        source_file="a.pdf",
        modality="pdf",
        text="See https://example.com — also at https://example.com",
        confidence=0.9,
        low_confidence=False,
    )
    state = _base_state(
        user_query="https://example.com",
        extraction_results=[result],
    )
    out = asyncio.run(fuse_node(state))
    assert out["detected_urls"].count("https://example.com") == 1


def test_fuse_node_builds_fused_context():
    from app.schemas.extraction import ExtractionResult
    r1 = ExtractionResult(
        source_file="a.pdf", modality="pdf", text="Hello", confidence=1.0, low_confidence=False
    )
    r2 = ExtractionResult(
        source_file="b.png", modality="image", text="World", confidence=1.0, low_confidence=False
    )
    state = _base_state(user_query="summarize", extraction_results=[r1, r2])
    out = asyncio.run(fuse_node(state))
    assert "Hello" in out["fused_context"]
    assert "World" in out["fused_context"]
    assert "summarize" in out["fused_context"]


def test_fuse_node_no_urls_returns_empty_list():
    from app.schemas.extraction import ExtractionResult
    result = ExtractionResult(
        source_file="a.pdf", modality="pdf", text="No links here at all.",
        confidence=0.9, low_confidence=False
    )
    state = _base_state(extraction_results=[result])
    out = asyncio.run(fuse_node(state))
    assert out["detected_urls"] == []


def test_extract_node_preserves_partial_result_when_gemini_fallback_times_out():
    """A failed vision fallback is traceable and does not discard the file result."""
    from app.extractors.image_extractor import extract_image
    class FailingGemini:
        def generate_content(self, _contents):
            raise TimeoutError("simulated Gemini timeout")

    image_path = Path(__file__).parent / "samples" / "sample_image.png"
    # Establish the fixture invokes the real fallback branch before asserting graph trace.
    direct = asyncio.run(extract_image(str(image_path), gemini_client=FailingGemini()))
    assert any("Gemini vision fallback failed" in warning for warning in direct.warnings)

    state = _base_state(raw_files=[{
        "filename": "sample_image.png", "content_type": "image/png", "path": str(image_path),
    }])
    out = asyncio.run(extract_node(state, gemini_client=FailingGemini()))

    assert len(out["extraction_results"]) == 1
    assert out["extraction_results"][0].low_confidence is True
    assert out["trace"][0].tool_name == "extract_node"
    assert out["trace"][0].status == "partial"
    assert "Gemini vision fallback failed" in out["trace"][0].output_summary


def test_extract_node_continues_after_genuinely_malformed_pdf(tmp_path):
    """pdfplumber parse failure for one file cannot abort a multi-file extraction."""
    valid_pdf = Path(__file__).parent / "samples" / "clean_sample.pdf"
    malformed_pdf = tmp_path / "malformed.pdf"
    malformed_pdf.write_bytes(b"%PDF-1.7\nthis is deliberately not a valid PDF structure")

    state = _base_state(raw_files=[
        {"filename": "good.pdf", "content_type": "application/pdf", "path": str(valid_pdf)},
        {"filename": "malformed.pdf", "content_type": "application/pdf", "path": str(malformed_pdf)},
    ])
    out = asyncio.run(extract_node(state))

    assert len(out["extraction_results"]) == 2
    assert "clean sample PDF text" in out["extraction_results"][0].text
    assert out["extraction_results"][1].low_confidence is True
    assert any("PDF processing error" in warning for warning in out["extraction_results"][1].warnings)
    assert any(event.status == "partial" for event in out["trace"])


# ---------------------------------------------------------------------------
# formatter_node — bullet enforcement
# ---------------------------------------------------------------------------

def test_formatter_node_truncates_four_bullets():
    summary = SummaryOutput(
        one_line="Short.",
        bullets=["A", "B", "C", "D"],
        five_sentence="1. 2. 3. 4. 5.",
    )
    plan = Plan(steps=[ToolCall(tool_name="summarize", args={}, reason="Summarise")])
    state = _base_state(plan=plan, tool_outputs={0: summary})
    out = asyncio.run(formatter_node(state))
    assert len(out["final_output"].summary.bullets) == 3


def test_formatter_node_pads_two_bullets():
    summary = SummaryOutput(
        one_line="Short.",
        bullets=["A", "B"],
        five_sentence="1. 2. 3. 4. 5.",
    )
    plan = Plan(steps=[ToolCall(tool_name="summarize", args={}, reason="Summarise")])
    state = _base_state(plan=plan, tool_outputs={0: summary})
    out = asyncio.run(formatter_node(state))
    assert len(out["final_output"].summary.bullets) == 3
    assert out["final_output"].summary.bullets[2] == "..."


def test_formatter_node_sets_task_type_from_plan():
    plan = Plan(steps=[ToolCall(tool_name="sentiment", args={}, reason="Analyse tone")])
    state = _base_state(plan=plan, tool_outputs={0: "some raw output"})
    out = asyncio.run(formatter_node(state))
    assert out["final_output"].task_type == "sentiment"


def test_formatter_node_produces_final_output():
    plan = Plan(steps=[ToolCall(tool_name="code_explain", args={}, reason="Explain")])
    state = _base_state(plan=plan, tool_outputs={0: "EXPLANATION\nDoes stuff."})
    out = asyncio.run(formatter_node(state))
    assert isinstance(out["final_output"], FinalOutput)
    assert out["final_output"].raw_text is not None


# ---------------------------------------------------------------------------
# MemorySaver checkpoint round-trip (Decision 9)
# ---------------------------------------------------------------------------

def test_tool_outputs_survives_checkpointer_round_trip():
    """
    Verifies that tool_outputs is preserved across a MemorySaver save + load cycle.

    Why this matters: the clarify-pause/resume path checkpoints AgentState mid-run.
    If tool_outputs were an undeclared TypedDict extra key, MemorySaver could
    silently drop it depending on how it serialises state. This test catches that
    failure mode before Phase 3 exercises the live clarify flow (Decision 9).
    """
    graph = build_graph()
    thread_id = "test-checkpoint-thread"
    config = {"configurable": {"thread_id": thread_id}}

    original_outputs = {0: "summarize output", 1: "sentiment output"}

    state = _base_state(
        session_id=thread_id,
        tool_outputs=original_outputs,
        status="planning",
    )

    asyncio.run(graph.ainvoke(state, config=config))
    snap = asyncio.run(graph.aget_state(config))

    restored_outputs = snap.values.get("tool_outputs")
    assert restored_outputs is not None, (
        "tool_outputs was dropped by MemorySaver — "
        "undeclared TypedDict keys are not safe across checkpointer cycles"
    )
    assert restored_outputs == original_outputs, (
        f"tool_outputs changed after checkpoint round-trip: {restored_outputs!r}"
    )
