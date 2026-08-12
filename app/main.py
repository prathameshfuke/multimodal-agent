"""
FastAPI entrypoint for multimodal-agent.

Endpoints:
  POST /session                    — start a new session (upload files + query)
  POST /session/{thread_id}/reply  — resume a clarify-paused session
  GET  /session/{thread_id}/trace  — read current trace without invoking the graph
"""

import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from google import genai
except ImportError:
    genai = None

from app.graph.build import build_graph
from app.logging_utils import log_event
from app.schemas.state import AgentState, UploadedFileRef
from app.upload_validation import validate_and_store_upload

load_dotenv()

TEMP_DIR = Path(os.getenv("TEMP_STORAGE_DIR", str(Path(tempfile.gettempdir()) / "multimodal_agent")))
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Construct the Gemini client exactly ONCE at startup. Every node and
    # tool receives this same instance — no defensive per-tool reconstruction.
    gemini_client = None
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key and genai is not None:
        gemini_client = genai.Client(api_key=api_key)

    # Build once at startup; reuse across all requests (Decision 1 — graph is pure scheduler).
    app.state.graph = build_graph(gemini_client=gemini_client)
    app.state.gemini_client = gemini_client
    yield


app = FastAPI(
    title="Multimodal Agent API",
    description="LangGraph-orchestrated multimodal agentic assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cleanup_thread_dir(thread_id: str) -> None:
    """Delete the temp upload directory for a finished session."""
    target = TEMP_DIR / thread_id
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


def _serialize_response(final_state: dict, thread_id: str) -> dict:
    status = final_state.get("status")

    if status == "awaiting_clarification":
        return {
            "status": "awaiting_clarification",
            "thread_id": thread_id,
            "question": final_state.get("clarify_question", "Please clarify your request."),
        }

    final_output = final_state.get("final_output")
    trace = final_state.get("trace", [])

    return {
        "status": "done",
        "thread_id": thread_id,
        "output": final_output.model_dump() if final_output else None,
        "trace": [t.model_dump() for t in trace],
    }


# ---------------------------------------------------------------------------
# POST /session
# ---------------------------------------------------------------------------


@app.post("/session")
async def create_session(
    background_tasks: BackgroundTasks,
    user_query: str = Form(default=""),
    files: list[UploadFile] = File(default=[]),
):
    """
    Start a new agent session. Returns immediately with either:
      - {"status": "done", "thread_id": ..., "output": {...}, "trace": [...]}
      - {"status": "awaiting_clarification", "thread_id": ..., "question": "..."}
    """
    thread_id = str(uuid.uuid4())
    thread_dir = TEMP_DIR / thread_id
    thread_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not user_query or not user_query.strip():
            raise HTTPException(status_code=422, detail="user_query must not be empty.")

        raw_files: list[UploadedFileRef] = []
        for index, upload in enumerate(files):
            filename = upload.filename or "upload"
            # Keep user-controlled names from escaping the thread directory.
            dest = thread_dir / f"{index}_{Path(filename).name}"
            detected_mime = await validate_and_store_upload(upload, dest)
            raw_files.append(
                UploadedFileRef(
                    filename=filename,
                    content_type=detected_mime,
                    path=str(dest),
                )
            )

        initial_state: AgentState = {
            "session_id": thread_id,
            "user_query": user_query,
            "raw_files": raw_files,
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

        config = {"configurable": {"thread_id": thread_id}}
        final_state = await app.state.graph.ainvoke(initial_state, config=config)

        # Temp files are consumed once by extract_node and never re-read.
        # On-response cleanup is safe for both the normal and clarify-pause paths.
        # See decisions.md Decision 10 for the extractor-level audit that confirms this.
        background_tasks.add_task(_cleanup_thread_dir, thread_id)

        return _serialize_response(final_state, thread_id)

    except HTTPException:
        _cleanup_thread_dir(thread_id)
        raise
    except Exception as e:
        log_event(thread_id=thread_id, node="api.create_session", status="error", latency_ms=0, error=str(e))
        _cleanup_thread_dir(thread_id)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ---------------------------------------------------------------------------
# POST /session/{thread_id}/reply
# ---------------------------------------------------------------------------


@app.post("/session/{thread_id}/reply")
async def reply_to_session(
    thread_id: str,
    background_tasks: BackgroundTasks,
    reply: str = Form(...),
):
    """
    Resume a clarify-paused session with the user's answer.
    Loads the checkpointed state (extraction_results, fused_context, etc.),
    updates user_query and rebuilds fused_context, then resumes from planner_node.
    """
    try:
        if not reply or not reply.strip():
            raise HTTPException(status_code=422, detail="reply must not be empty.")
        config = {"configurable": {"thread_id": thread_id}}
        graph = app.state.graph
        snapshot = await graph.aget_state(config)

        if not snapshot or not snapshot.values:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{thread_id}' not found or has not been started.",
            )

        old_values = snapshot.values

        # Rebuild fused_context with the clarification as the new user_query.
        # This mirrors fuse_node's logic so the planner sees an accurate context.
        extraction_results = old_values.get("extraction_results", [])
        extracted_texts = [r.text for r in extraction_results if r.text]
        body = "\n\n---\n\n".join(extracted_texts)
        new_fused_context = (
            f"User query: {reply}\n\n---\n\n{body}" if body else f"User query: {reply}"
        )

        # Update state as if fuse_context node just produced this output.
        # LangGraph will continue from the edges after fuse_context → planner_node.
        await graph.aupdate_state(
            config,
            {
                "user_query": reply,
                "fused_context": new_fused_context,
                "clarify_question": None,
                "status": "planning",
            },
            as_node="fuse_context",
        )

        # Resume the graph from the planner node with no new top-level input.
        final_state = await graph.ainvoke(None, config=config)

        background_tasks.add_task(_cleanup_thread_dir, thread_id)
        return _serialize_response(final_state, thread_id)

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ---------------------------------------------------------------------------
# GET /session/{thread_id}/trace
# ---------------------------------------------------------------------------


@app.get("/session/{thread_id}/trace")
async def get_trace(thread_id: str):
    """Read the current trace for a session without invoking the graph."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await app.state.graph.aget_state(config)

        if not snapshot or not snapshot.values:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{thread_id}' not found.",
            )

        trace = snapshot.values.get("trace", [])
        return {"thread_id": thread_id, "trace": [t.model_dump() for t in trace]}

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ---------------------------------------------------------------------------
# Utility & Static Files
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check():
    # Deliberately avoid a Gemini network request: Render only needs a cheap
    # liveness/configuration signal, not an API quota-consuming readiness call.
    gemini_configured = bool(
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    return {"status": "healthy", "gemini_configured": gemini_configured}


# Mount static files at root (html=True serves index.html at /)
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
