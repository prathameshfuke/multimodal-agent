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
from typing import Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
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

    # StateGraph is pure and compiled once at startup.
    # Client instances are constructed per request from header or environment.
    app.state.graph = build_graph()
    yield


app = FastAPI(
    title="Multimodal Intelligence API",
    description="Production-grade LangGraph multimodal content processing system",
    version="1.0.0",
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


async def _get_or_create_gemini_client(x_gemini_api_key: str | None = None) -> Any:
    """
    Resolve Gemini API key from per-request header or server environment,
    construct a per-session client instance, and run a lightweight validation check.
    Raises HTTP 400 if no key is present or the provided key is invalid.
    """
    api_key = (x_gemini_api_key or "").strip() or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No Gemini API key provided. Please supply an API key in the Settings panel or configure GEMINI_API_KEY on the server.",
        )
    if genai is None:
        raise HTTPException(
            status_code=500,
            detail="google-genai SDK is not installed on the server.",
        )
    try:
        client = genai.Client(api_key=api_key)
        # Fast key validation (Requirement 5)
        if hasattr(client, "aio") and hasattr(client.aio, "models"):
            res = client.aio.models.count_tokens(model="gemini-2.5-flash", contents="ping")
            if hasattr(res, "__await__"):
                await res
        elif hasattr(client, "models") and hasattr(client.models, "count_tokens"):
            client.models.count_tokens(model="gemini-2.5-flash", contents="ping")
        return client
    except HTTPException:
        raise
    except Exception as exc:
        log_event(thread_id="validation", node="api_key_validation", status="error", latency_ms=0, error=str(exc))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Gemini API key provided: {exc}",
        )


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
    x_gemini_api_key: str | None = Header(default=None, alias="X-Gemini-Api-Key"),
):
    """
    Start a new agent session. Resolves per-session Gemini client from header or server env.
    Returns immediately with either:
      - {"status": "done", "thread_id": ..., "output": {...}, "trace": [...]}
      - {"status": "awaiting_clarification", "thread_id": ..., "question": "..."}
    """
    thread_id = str(uuid.uuid4())
    thread_dir = TEMP_DIR / thread_id
    thread_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not user_query or not user_query.strip():
            raise HTTPException(status_code=422, detail="user_query must not be empty.")

        # Construct and validate per-session Gemini client fast before running graph
        gemini_client = await _get_or_create_gemini_client(x_gemini_api_key)

        raw_files: list[UploadedFileRef] = []
        for index, upload in enumerate(files):
            filename = upload.filename or "upload"
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

        config = {"configurable": {"thread_id": thread_id, "gemini_client": gemini_client}}
        final_state = await app.state.graph.ainvoke(initial_state, config=config)

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
    x_gemini_api_key: str | None = Header(default=None, alias="X-Gemini-Api-Key"),
):
    """
    Resume a clarify-paused session with the user's answer.
    """
    try:
        if not reply or not reply.strip():
            raise HTTPException(status_code=422, detail="reply must not be empty.")

        gemini_client = await _get_or_create_gemini_client(x_gemini_api_key)

        config = {"configurable": {"thread_id": thread_id, "gemini_client": gemini_client}}
        graph = app.state.graph
        snapshot = await graph.aget_state(config)

        if not snapshot or not snapshot.values:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{thread_id}' not found or has not been started.",
            )

        old_values = snapshot.values

        extraction_results = old_values.get("extraction_results", [])
        extracted_texts = [r.text for r in extraction_results if r.text]
        body = "\n\n---\n\n".join(extracted_texts)
        new_fused_context = (
            f"User query: {reply}\n\n---\n\n{body}" if body else f"User query: {reply}"
        )

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
