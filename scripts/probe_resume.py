"""
Async probe: test langgraph ainvoke/aupdate_state resume behavior
with async node functions.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.graph.build import route_after_plan, route_after_exec
from app.schemas.extraction import ExtractionResult
from app.schemas.output import FinalOutput, SummaryOutput
from app.schemas.plan import Plan, ToolCall
from app.schemas.state import AgentState
from app.schemas.trace import TraceEvent

extract_call_count = {"count": 0}
planner_call_count = {"count": 0}


async def controlled_extract(state):
    extract_call_count["count"] += 1
    print(f"  [extract_node] call #{extract_call_count['count']}")
    result = ExtractionResult(
        source_file="probe.pdf",
        modality="pdf",
        text="Probe document text.",
        confidence=0.95,
        low_confidence=False,
    )
    return {
        **state,
        "extraction_results": [result],
        "trace": state.get("trace", []),
        "status": "planning",
    }


async def controlled_fuse(state):
    user_query = state.get("user_query", "")
    texts = [r.text for r in state.get("extraction_results", []) if r.text]
    body = "\n\n---\n\n".join(texts)
    print(f"  [fuse_node] user_query={user_query!r}")
    return {
        **state,
        "fused_context": f"User query: {user_query}\n\n---\n\n{body}",
        "detected_urls": [],
    }


async def controlled_planner(state):
    planner_call_count["count"] += 1
    print(f"  [planner_node] call #{planner_call_count['count']}, status={state.get('status')!r}")
    if planner_call_count["count"] == 1:
        return {
            **state,
            "plan": None,
            "clarify_question": "What would you like me to do?",
            "status": "awaiting_clarification",
        }
    plan = Plan(
        steps=[ToolCall(tool_name="summarize", args={"text": "Probe document text."}, reason="User asked to summarize.")]
    )
    return {
        **state,
        "plan": plan,
        "clarify_question": None,
        "status": "executing",
    }


async def controlled_executor(state):
    plan = state.get("plan")
    tool_outputs = {}
    trace = list(state.get("trace", []))
    if plan and plan.steps:
        step = plan.steps[0]
        tool_outputs[0] = "Summarized content."
        trace.append(
            TraceEvent(
                step_index=len(trace),
                tool_name=step.tool_name,
                input_summary="probe",
                output_summary="Summarized content.",
                latency_ms=1,
                status="success",
            )
        )
    print(f"  [executor_node] ran tools: {[s.tool_name for s in (plan.steps if plan else [])]}")
    return {**state, "tool_outputs": tool_outputs, "trace": trace, "status": "done"}


async def controlled_formatter(state):
    summary = SummaryOutput(
        one_line="Probe summary.",
        bullets=["B1.", "B2.", "B3."],
        five_sentence="S1. S2. S3. S4. S5.",
    )
    print("  [formatter_node] producing FinalOutput")
    return {**state, "final_output": FinalOutput(task_type="summarize", summary=summary), "status": "done"}


def build_probe_graph():
    g = StateGraph(AgentState)
    g.add_node("extract", controlled_extract)
    g.add_node("fuse_context", controlled_fuse)
    g.add_node("plan", controlled_planner)
    g.add_node("execute_tools", controlled_executor)
    g.add_node("format_output", controlled_formatter)
    g.set_entry_point("extract")
    g.add_edge("extract", "fuse_context")
    g.add_edge("fuse_context", "plan")
    g.add_conditional_edges("plan", route_after_plan, {"clarify": END, "execute": "execute_tools"})
    g.add_conditional_edges("execute_tools", route_after_exec, {"done": "format_output"})
    g.add_edge("format_output", END)
    return g.compile(checkpointer=MemorySaver())


async def main():
    graph = build_probe_graph()
    config = {"configurable": {"thread_id": "probe-thread-001"}}

    initial_state: AgentState = {
        "session_id": "probe-thread-001",
        "user_query": "",
        "raw_files": [],
        "extraction_results": [],
        "fused_context": "",
        "detected_urls": [],
        "plan": None,
        "trace": [],
        "clarify_question": None,
        "final_output": None,
        "tool_outputs": {},
        "status": "extracting",
    }

    print("\n=== Phase 1: ainvoke with initial_state (bare upload, no query) ===")
    s1 = await graph.ainvoke(initial_state, config=config)
    print(f"  -> status={s1.get('status')!r}, clarify={s1.get('clarify_question')!r}")
    print(f"  -> extract_call_count={extract_call_count['count']}")
    assert s1["status"] == "awaiting_clarification", f"Expected clarification, got {s1['status']}"
    assert extract_call_count["count"] == 1

    print("\n=== Phase 2: aupdate_state(as_node='fuse_context') + ainvoke(None) ===")
    snap = await graph.aget_state(config)
    old_vals = snap.values
    texts = [r.text for r in old_vals.get("extraction_results", []) if r.text]
    body = "\n\n---\n\n".join(texts)
    reply = "Please summarize the document."
    new_fused = f"User query: {reply}\n\n---\n\n{body}" if body else f"User query: {reply}"

    await graph.aupdate_state(
        config,
        {
            "user_query": reply,
            "fused_context": new_fused,
            "clarify_question": None,
            "status": "planning",
        },
        as_node="fuse_context",
    )

    s2 = await graph.ainvoke(None, config=config)
    print(f"  -> status={s2.get('status')!r}, final_output={s2.get('final_output') is not None}")
    print(f"  -> extract_call_count={extract_call_count['count']} (expected: 1)")
    print(f"  -> planner_call_count={planner_call_count['count']} (expected: 2)")

    if extract_call_count["count"] == 1:
        print("\n✅ PASS: extract_node did NOT re-run on resume.")
        print("   aupdate_state(as_node='fuse_context') + ainvoke(None) correctly resumes from plan.")
        print("   Decision 10's cleanup-on-response safety argument is proven correct.")
        return True
    else:
        print(f"\n❌ FAIL: extract_node ran {extract_call_count['count']} times — it RE-RAN on resume.")
        print("   graph.ainvoke(None) restarted from extract, not from plan.")
        print("   Fix required: add interrupt_before=['plan'] to graph.compile() in app/graph/build.py.")
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
