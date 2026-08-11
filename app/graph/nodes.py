"""
LangGraph node functions. Each is a standalone async function (AgentState) -> AgentState.
All tool dispatch, retry, and branching logic lives here in plain Python — LangGraph
only sees coarse status transitions (Decision 1).
"""

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from app.schemas.extraction import ExtractionResult
from app.schemas.output import FinalOutput, SentimentOutput, SummaryOutput
from app.schemas.plan import Plan, ToolCall
from app.schemas.state import AgentState
from app.schemas.trace import TraceEvent
from app.logging_utils import log_event
from app.tools.registry import dispatch_tool, get_tool_schema_for_planner

_PLANNER_PROMPT = (
    Path(__file__).resolve().parent.parent.parent / "prompts" / "planner_v1.txt"
)

_URL_RE = re.compile(r"https?://[^\s<>\"\']+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mime_to_modality(content_type: str) -> str:
    if "pdf" in content_type:
        return "pdf"
    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("audio/") or "audio" in content_type:
        return "audio"
    return "text"


async def _extract_one(file_ref: dict, gemini_client: Any) -> ExtractionResult | None:
    modality = _mime_to_modality(file_ref.get("content_type", ""))
    path = file_ref["path"]

    try:
        if modality == "pdf":
            from app.extractors.pdf_extractor import extract_pdf
            return await extract_pdf(path, gemini_client=gemini_client)
        elif modality == "image":
            from app.extractors.image_extractor import extract_image
            return await extract_image(path, gemini_client=gemini_client)
        elif modality == "audio":
            from app.extractors.audio_extractor import extract_audio
            return await extract_audio(path, gemini_client=gemini_client)
        else:
            return ExtractionResult(
                source_file=path,
                modality="text",
                text=Path(path).read_text(encoding="utf-8", errors="replace"),
                confidence=1.0,
                low_confidence=False,
            )
    except Exception as e:
        return ExtractionResult(
            source_file=path,
            modality=modality,  # type: ignore[arg-type]
            text="",
            confidence=0.0,
            low_confidence=True,
            warnings=[f"Extraction failed: {e}"],
        )


def _summarise_result(result: ExtractionResult) -> str:
    """One-line input_summary for a TraceEvent."""
    return f"{result.source_file} ({result.modality}, {len(result.text)} chars)"


# ---------------------------------------------------------------------------
# extract_node
# ---------------------------------------------------------------------------


async def extract_node(state: AgentState, *, gemini_client: Any = None) -> AgentState:
    started = time.monotonic()
    thread_id = state.get("session_id", "unknown")
    raw_files = state.get("raw_files", [])
    tasks = [_extract_one(f, gemini_client) for f in raw_files]
    results: list[ExtractionResult | None] = await asyncio.gather(*tasks)

    good: list[ExtractionResult] = []
    trace = list(state.get("trace", []))

    for result in results:
        if result is None:
            continue
        good.append(result)
        if result.warnings:
            # Partial-failure files still continue — warnings carry the signal
            trace.append(
                TraceEvent(
                    step_index=len(trace),
                    tool_name="extract_node",
                    input_summary=result.source_file,
                    output_summary="; ".join(result.warnings),
                    latency_ms=0,
                    status="partial" if result.low_confidence else "success",
                )
            )

    output = {
        **state,
        "extraction_results": good,
        "trace": trace,
        "status": "planning",
    }
    log_event(
        thread_id=thread_id,
        node="extract_node",
        status="partial" if any(result.low_confidence for result in good) else "success",
        latency_ms=int((time.monotonic() - started) * 1000),
        files=len(raw_files),
    )
    return output


# ---------------------------------------------------------------------------
# fuse_node
# ---------------------------------------------------------------------------


def _extract_urls(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for text in texts:
        for url in _URL_RE.findall(text):
            # Strip trailing punctuation that the regex may capture
            url = url.rstrip(".,;:!?)")
            if url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


async def fuse_node(state: AgentState) -> AgentState:
    started = time.monotonic()
    results: list[ExtractionResult] = state.get("extraction_results", [])
    user_query = state.get("user_query", "")

    parts = [r.text for r in results if r.text]
    fused = "\n\n---\n\n".join(parts)
    if user_query:
        fused = f"User query: {user_query}\n\n---\n\n{fused}"

    detected_urls = _extract_urls([user_query] + [r.text for r in results])

    output = {
        **state,
        "fused_context": fused,
        "detected_urls": detected_urls,
    }
    log_event(
        thread_id=state.get("session_id", "unknown"), node="fuse_node", status="success",
        latency_ms=int((time.monotonic() - started) * 1000), urls=len(detected_urls),
    )
    return output


# ---------------------------------------------------------------------------
# planner_node
# ---------------------------------------------------------------------------


def _build_planner_prompt(state: AgentState) -> str:
    tool_schema_text = json.dumps(get_tool_schema_for_planner(), indent=2)
    template = _PLANNER_PROMPT.read_text(encoding="utf-8")
    return template.format(
        tool_schema=tool_schema_text,
        detected_urls=state.get("detected_urls", []),
        user_query=state.get("user_query", ""),
        fused_context=state.get("fused_context", ""),
    )


def _parse_plan(raw: str) -> Plan | None:
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rstrip("`").strip()
    try:
        data = json.loads(raw)
        return Plan(**data)
    except Exception:
        return None


async def planner_node(state: AgentState, *, gemini_client: Any = None) -> AgentState:
    started = time.monotonic()
    prompt = _build_planner_prompt(state)
    plan: Plan | None = None

    for attempt in range(2):
        try:
            if gemini_client is not None and hasattr(gemini_client, "aio"):
                response = await gemini_client.aio.models.generate_content(
                    model="gemini-2.5-flash", contents=prompt
                )
                raw = (response.text or "").strip()
            elif gemini_client is not None:
                res = gemini_client.generate_content(prompt)
                raw = (res.text if hasattr(res, "text") else str(res)).strip()
            else:
                from google import genai
                client = genai.Client()
                response = client.models.generate_content(
                    model="gemini-2.5-flash", contents=prompt
                )
                raw = (response.text or "").strip()

            plan = _parse_plan(raw)
            if plan is not None:
                break
        except Exception:
            if attempt == 1:
                break

    if plan is None:
        plan = Plan(
            steps=[],
            clarify_question="I couldn't interpret your request. Could you please restate what you'd like me to do?",
        )

    # Enforce invariant: steps XOR clarify_question, never both/neither
    if plan.steps and plan.clarify_question:
        plan = Plan(steps=plan.steps, clarify_question=None)
    elif not plan.steps and not plan.clarify_question:
        plan = Plan(
            steps=[],
            clarify_question="Please clarify what you'd like me to do with the provided content.",
        )

    new_status = (
        "awaiting_clarification" if plan.clarify_question else "executing"
    )
    output = {
        **state,
        "plan": plan,
        "clarify_question": plan.clarify_question,
        "status": new_status,
    }
    log_event(
        thread_id=state.get("session_id", "unknown"), node="planner_node", status=new_status,
        latency_ms=int((time.monotonic() - started) * 1000), steps=len(plan.steps),
    )
    return output


# ---------------------------------------------------------------------------
# executor_node
# ---------------------------------------------------------------------------


async def executor_node(state: AgentState, *, gemini_client: Any = None) -> AgentState:
    started = time.monotonic()
    thread_id = state.get("session_id", "unknown")
    plan: Plan | None = state.get("plan")
    if not plan or not plan.steps:
        output = {**state, "status": "done"}
        log_event(thread_id=thread_id, node="executor_node", status="success", latency_ms=0, steps=0)
        return output

    trace = list(state.get("trace", []))
    # Declared AgentState field — survives MemorySaver checkpoint on clarify-pause.
    tool_outputs: dict = dict(state.get("tool_outputs", {}))

    for i, step in enumerate(plan.steps):
        t0 = time.monotonic()
        step_status = "success"
        output: Any = None

        for attempt in range(2):
            try:
                output = await dispatch_tool(step.tool_name, step.args, gemini_client, thread_id=thread_id)
                break
            except Exception as e:
                if attempt == 0:
                    # Reflexion retry: feed the error back into args context
                    step.args["_retry_error_context"] = str(e)
                else:
                    step_status = "partial"
                    output = f"Tool '{step.tool_name}' failed after one retry: {e}"

        latency_ms = int((time.monotonic() - t0) * 1000)
        output_summary = (
            str(output)[:200] if output is not None else "No output"
        )

        trace.append(
            TraceEvent(
                step_index=len(trace),
                tool_name=step.tool_name,
                input_summary=f"{step.reason} | args: {list(step.args.keys())}",
                output_summary=output_summary,
                latency_ms=latency_ms,
                status=step_status,
            )
        )
        tool_outputs[i] = output

    output = {
        **state,
        "trace": trace,
        "tool_outputs": tool_outputs,
        "status": "done",
    }
    log_event(
        thread_id=thread_id, node="executor_node",
        status="partial" if any(event.status == "partial" for event in trace) else "success",
        latency_ms=int((time.monotonic() - started) * 1000), steps=len(plan.steps),
    )
    return output


# ---------------------------------------------------------------------------
# formatter_node
# ---------------------------------------------------------------------------


def _enforce_three_bullets(bullets: list[str]) -> list[str]:
    if len(bullets) > 3:
        return bullets[:3]
    while len(bullets) < 3:
        bullets.append("...")
    return bullets


def _derive_task_type(plan: Plan | None) -> str:
    if plan and plan.steps:
        return plan.steps[0].tool_name
    return "conversational"


async def formatter_node(state: AgentState) -> AgentState:
    started = time.monotonic()
    plan: Plan | None = state.get("plan")
    tool_outputs: dict = state.get("tool_outputs", {})
    task_type = _derive_task_type(plan)

    # Find the first non-None output to build FinalOutput from
    primary_output = next(
        (v for v in tool_outputs.values() if v is not None), None
    )

    summary: SummaryOutput | None = None
    sentiment_out: SentimentOutput | None = None
    raw_text: str | None = None

    if isinstance(primary_output, SummaryOutput):
        primary_output.bullets = _enforce_three_bullets(primary_output.bullets)
        summary = primary_output
    elif isinstance(primary_output, SentimentOutput):
        sentiment_out = primary_output
    elif isinstance(primary_output, (str, dict)):
        raw_text = (
            json.dumps(primary_output, indent=2)
            if isinstance(primary_output, dict)
            else primary_output
        )
    elif primary_output is None:
        raw_text = "No output was produced."

    final_output = FinalOutput(
        task_type=task_type,
        summary=summary,
        sentiment=sentiment_out,
        raw_text=raw_text,
    )

    output = {
        **state,
        "final_output": final_output,
        "status": "done",
    }
    log_event(
        thread_id=state.get("session_id", "unknown"), node="formatter_node", status="success",
        latency_ms=int((time.monotonic() - started) * 1000), task_type=task_type,
    )
    return output
