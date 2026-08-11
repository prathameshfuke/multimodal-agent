"""
Assembles and compiles the LangGraph StateGraph.

Graph shape (Decision 1 — LangGraph as scheduler only):
  extract -> fuse_context -> plan
  plan -> route_after_plan -> clarify (END) | execute_tools
  execute_tools -> route_after_exec -> format_output -> END

The two routing functions are named, testable callables — not inline lambdas.
MemorySaver enables clarify-pause → resume within the same session thread.
"""

from functools import partial
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.schemas.state import AgentState
from app.graph.nodes import (
    extract_node,
    fuse_node,
    planner_node,
    executor_node,
    formatter_node,
)


# ---------------------------------------------------------------------------
# Routing functions — named so they are independently unit-testable
# ---------------------------------------------------------------------------


def route_after_plan(state: AgentState) -> str:
    """After plan: go to clarify (END) if waiting for user, else execute."""
    if state.get("status") == "awaiting_clarification":
        return "clarify"
    return "execute"


def route_after_exec(state: AgentState) -> str:
    """After execution: always format output (retry loop is inside executor_node)."""
    return "done"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(*, gemini_client: Any = None):
    """
    Compile and return the runnable StateGraph with MemorySaver checkpointer.
    Call once at app startup; reuse the returned object for all requests.
    """
    graph = StateGraph(AgentState)

    # LangGraph invokes nodes with state only, so bind the application-scoped
    # client here rather than allowing each node to receive None at runtime.
    graph.add_node("extract", partial(extract_node, gemini_client=gemini_client))
    graph.add_node("fuse_context", fuse_node)
    graph.add_node("plan", partial(planner_node, gemini_client=gemini_client))
    graph.add_node("execute_tools", partial(executor_node, gemini_client=gemini_client))
    graph.add_node("format_output", formatter_node)

    graph.set_entry_point("extract")

    graph.add_edge("extract", "fuse_context")
    graph.add_edge("fuse_context", "plan")

    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "clarify": END,
            "execute": "execute_tools",
        },
    )

    graph.add_conditional_edges(
        "execute_tools",
        route_after_exec,
        {
            "done": "format_output",
        },
    )

    graph.add_edge("format_output", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
