from app.schemas.extraction import ExtractionResult
from app.schemas.plan import Plan, ToolCall
from app.schemas.trace import TraceEvent
from app.schemas.output import SummaryOutput, SentimentOutput, FinalOutput
from app.schemas.state import AgentState, UploadedFileRef


def test_extraction_result_schema():
    res = ExtractionResult(
        source_file="doc.pdf",
        modality="pdf",
        text="Extracted content",
        confidence=0.95,
        low_confidence=False,
        warnings=[],
    )
    assert res.modality == "pdf"
    assert res.confidence == 0.95


def test_plan_schema():
    call = ToolCall(
        tool_name="pdf_extractor",
        args={"file_path": "/tmp/test.pdf"},
        reason="Extract text from uploaded PDF",
    )
    plan = Plan(steps=[call], clarify_question=None)
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == "pdf_extractor"


def test_trace_event_schema():
    event = TraceEvent(
        step_index=0,
        tool_name="pdf_extractor",
        input_summary="test.pdf",
        output_summary="Extracted 10 lines",
        latency_ms=150,
        status="success",
    )
    assert event.status == "success"
    assert event.latency_ms == 150


def test_output_schemas():
    summary = SummaryOutput(
        one_line="Quick summary.",
        bullets=["Point 1", "Point 2", "Point 3"],
        five_sentence="Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5.",
    )
    sentiment = SentimentOutput(
        label="positive",
        confidence=0.9,
        justification="Tone is enthusiastic.",
    )
    final = FinalOutput(
        task_type="summary",
        summary=summary,
        sentiment=sentiment,
        raw_text=None,
    )
    assert final.task_type == "summary"
    assert final.summary.bullets == ["Point 1", "Point 2", "Point 3"]


def test_agent_state_schema():
    file_ref: UploadedFileRef = {
        "filename": "sample.jpg",
        "content_type": "image/jpeg",
        "path": "/tmp/uploads/sample.jpg",
    }
    state: AgentState = {
        "session_id": "session-123",
        "user_query": "Analyze image",
        "raw_files": [file_ref],
        "extraction_results": [],
        "fused_context": "",
        "detected_urls": [],
        "plan": None,
        "trace": [],
        "clarify_question": None,
        "final_output": None,
        "status": "extracting",
    }
    assert state["status"] == "extracting"
    assert state["raw_files"][0]["filename"] == "sample.jpg"
