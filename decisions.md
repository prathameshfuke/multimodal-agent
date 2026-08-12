# Architecture & Design Decisions

### Decision 1: LangGraph as Orchestration Scheduler Only

**Problem**: Determining whether LangGraph should manage tool selection, retries, and conditional execution loops inside its graph framework vs using LangGraph strictly as a high-level orchestration state machine/scheduler while executing tool selection, dispatch, and retries in plain Python.

**Options Considered**: (1) Native LangGraph conditional edges, agent loops, and tool node abstractions for retry logic and tool dispatch. (2) LangGraph as a minimalist macro-state machine for coarse transitions (`extracting` -> `planning` -> `executing` -> `done`/`error`/`awaiting_clarification`), with tool routing, selection, execution loops, and retries handled in plain Python inside individual nodes.

**Choice**: Option 2 (LangGraph as scheduler only; tool dispatch and retries in plain Python).

**Why**: Plain Python logic is dramatically easier to unit test, debug, log, and step through deterministically. It avoids framework abstraction overhead, complex state mutation hidden inside graph edges, and fragile dynamic node routing logic.

**Tradeoff Accepted**: We forego out-of-the-box framework abstractions for graph-native subgraphs or automatic tool node bindings, requiring explicit custom Python helper functions for tool retries and error handling.

---

### Decision 2: Render Free Tier Deployment Target with Hugging Face Spaces Fallback

**Problem**: Choosing a host platform for initial deployment that supports Docker containers and heavy dependencies (like `faster-whisper`, `pytesseract`, `pdfplumber`) within free-tier constraints.

**Options Considered**: (1) Render Free Tier. (2) Hugging Face Spaces (Docker SDK / Gradio / FastAPI container runner). (3) AWS Lambda / Serverless container.

**Choice**: Option 1 (Render Free Tier as primary, Hugging Face Spaces as fallback).

**Why**: Render provides straightforward Docker deployment, built-in HTTPS, easy environment variable management, and continuous deployment from Git without UI wrapping.

**Failure Mode & Fallback Trigger**: Render free tier imposes strict RAM limits (512 MB). If loading heavy multimodal models (such as `faster-whisper` runtime weights or OCR libraries) causes Out-Of-Memory (OOM) process termination (`Killed` / Exit Code 137) during startup or runtime execution, deployment will switch to Hugging Face Spaces Docker SDK, which provides a larger free RAM allocation (~16 GB CPU container tier).

**Tradeoff Accepted**: Render free tier spins down after inactivity (cold starts ~50 seconds), which causes initial response latency after idle periods.

---

### Decision 3: Schema Validation Choice for `bullets: list[str]` in `SummaryOutput`

**Problem**: The schema defines `bullets: list[str] # exactly 3`, but standard Python `list[str]` typing does not enforce list length at runtime.

**Options Considered**: (1) Add a custom Pydantic `@field_validator` on `bullets` to raise a runtime validation error if `len(bullets) != 3`. (2) Enforce the exact 3-bullet constraint at the prompt engineering and output parser/formatter level while keeping the Pydantic schema clean and matching exact specification.

**Choice**: Option 2 (Prompt/Formatter level enforcement).

**Why**: Raising schema validation errors when an LLM generates 2 or 4 bullets breaks downstream execution or requires costly graph retry loops. Standardizing bullet count during prompt design and applying post-processing truncation/padding in the formatter yields resilient system behavior without inventing non-standard Pydantic validator fields in the core schema contracts.

**Tradeoff Accepted**: Schema validation alone will allow lists with length != 3 if raw unformatted dicts bypass the formatter, relying on formatting logic rather than structural type errors.

---

### Decision 4: File Reference Representation (`raw_files` stores path vs bytes)

**Problem**: Managing uploaded files (PDFs, images, audio) inside `AgentState` without bloating state or breaking serialization.

**Options Considered**: (1) Store raw binary file content (bytes/base64) directly inside `AgentState`. (2) Store temp disk storage file paths (`path: str`) inside `UploadedFileRef`.

**Choice**: Option 2 (Temp disk storage path string).

**Why**: Storing path strings keeps `AgentState` lightweight, JSON-serializable, and fast for LangGraph's checkpointer (essential for pause-on-clarification and state resumption). Storing raw bytes in state inflates checkpointer memory, slows down state snapshot saves/restores, and risks hitting checkpointer payload limits with large PDFs or audio files.

**Tradeoff Accepted**: Requires ephemeral local file system management (writing to temp directory on upload and cleanup upon session termination).

---

### Decision 5: Selection of `faster-whisper` Model Size (`base` model with int8 quantization)

**Problem**: Selecting an ASR model size for `faster-whisper` that maximizes transcription accuracy while remaining strictly within Render's free-tier 512 MB RAM ceiling established in Decision 2.

**Options Considered**: (1) `tiny` model (~75 MB RAM). (2) `base` model with `int8` quantization (~150 MB RAM). (3) `small` model (~500 MB RAM). (4) `medium`/`large` models (>1.5 GB RAM).

**Choice**: Option 2 (`base` model with `compute_type="int8"` on CPU).

**Why**: The `base` model in int8 format consumes approximately 150 MB of RAM, leaving ~300 MB of headroom for the FastAPI process, Python interpreter runtime, and incoming HTTP requests on Render's 512 MB free tier. Selecting `small` (~500 MB) would push memory usage into immediate Out-Of-Memory (OOM) `Exit Code 137` process termination during model loading. If unexpected request concurrency or larger audio files cause memory exhaustion on Render, the failure mode triggers the migration fallback to Hugging Face Spaces as documented in Decision 2.

**Tradeoff Accepted**: The `base` model has lower zero-shot accuracy for heavily accented speech or complex domain jargon compared to `small`/`medium` models, which is mitigated by running the subsequent Gemini 2.5 Flash surface cleanup pass (`prompts/audio_cleanup_v1.txt`).

---

### Decision 6: Per-Page Granularity for PDF OCR / Vision Fallbacks

**Problem**: Determining whether low-confidence OCR detection in a PDF document should trigger fallback processing on a per-page basis vs routing the entire document to the Gemini vision fallback tier.

**Options Considered**: (1) Global document fallback: If any page fails local extraction or exhibits low OCR confidence, send the entire PDF (all pages) to Gemini vision. (2) Per-page fallback: Evaluate each page independently with `pdfplumber` and `pytesseract`, sending only specific low-confidence or scanned pages to Gemini vision fallback.

**Choice**: Option 2 (Per-page OCR / vision fallback evaluation).

**Why**: Multi-page PDF documents commonly contain a mix of clean digital text pages alongside isolated scanned diagrams or low-quality table pages. Evaluating pages independently ensures Gemini vision API calls are triggered exclusively for pages that actually require vision capabilities, drastically reducing API token costs and lowering total end-to-end response latency on multi-page documents.

**Tradeoff Accepted**: Per-page evaluation introduces per-page image rasterization logic, separate page-level confidence tracking, and warning aggregation within `pdf_extractor.py`.

---

### Decision 7: Tool Schema Exposure — Static Registry Dict vs Auto-Derived from Signatures

**Problem**: The planner LLM must receive the six tools' names, descriptions, and argument schemas to make valid selections. The schema needs to stay accurate as tools evolve.

**Options Considered**: (1) Static Python dicts in `app/tools/registry.py` (`TOOL_REGISTRY`) with explicit `name`, `description`, and `args_schema` fields maintained by hand. (2) Auto-derive tool schema at import time by inspecting each function's signature and docstring via `inspect.signature` and `__doc__`.

**Choice**: Option 1 (static dict registry in `registry.py`, exposed through `get_tool_schema_for_planner()`).

**Why**: The exact description text the planner LLM reads determines whether it selects the right tool. Auto-derived text from docstrings is often too terse, too technical, or structured for a Python reader rather than an LLM. A static dict lets us craft precise, example-rich descriptions tuned for the model without coupling the API surface of a function to its LLM-facing documentation. A grader reading the code can also see and audit the exact schema the planner receives without running the system.

**Tradeoff Accepted**: The dict can drift out of sync if a tool's signature changes and the developer forgets to update `registry.py`. A future `@register_tool` decorator could enforce sync without changing the interface.

---

### Decision 8: Retry Cap Fixed at 1 (Not 0, Not 3+)

**Problem**: Determining how many Reflexion-style retry attempts to allow for tool execution failures and extraction fallbacks before accepting a partial or degraded result.

**Options Considered**: (1) No retry (0 additional attempts): fail fast, cheapest. (2) One retry (1 additional attempt): feed the error back as context and try once more. (3) Multiple retries (3+): maximize recovery odds at the cost of latency and quota.

**Choice**: Option 2 (exactly 1 retry per failure site).

**Why**: The Gemini 2.5 Flash free tier allows approximately 15 requests per minute (RPM). A six-step plan with three retry-eligible tool calls and a three-layer fallback chain for two extractors could consume 15+ RPM in a single run if unbounded. One retry provides the most valuable correction signal — the model sees its own error message as feedback — while adding at most one extra RTT per failure. Beyond one retry, the marginal accuracy gain from a generic LLM call diminishes sharply because the root cause (ambiguous input, missing data) does not change between attempts. For a demo context where end-to-end latency directly affects the evaluator's experience, capping at one retry keeps worst-case wall time predictable.

**Tradeoff Accepted**: Some failures that a second retry might have recovered from will instead produce a `status="partial"` result with a warning. This is surfaced in the trace, giving the user enough signal to restate their query rather than receiving a silent bad answer.

---

### Decision 9: Promoting `tool_outputs` from Private Scratch State to a Declared `AgentState` Field

**Problem**: `executor_node` accumulates raw tool outputs (keyed by step index) and passes them to `formatter_node`. In the initial Phase 2 implementation, this was stored as `_tool_outputs`, an undeclared extra key on the `AgentState` TypedDict. Python's TypedDict allows extra keys at runtime, so this worked within a single uninterrupted graph run. However, the clarify-pause/resume path checkpoints `AgentState` via `MemorySaver` mid-run. If `MemorySaver` performs any schema-aware serialisation that ignores undeclared keys, `_tool_outputs` could be silently dropped on resume — a bug that would only surface during the exact clarify-flow demo scenario the rubric tests for (Autonomy & Planning, Explainability).

**Options Considered**: (1) Leave `_tool_outputs` as an undeclared extra key and verify empirically that MemorySaver preserves it. (2) Declare `tool_outputs: dict` explicitly in `AgentState` before Phase 3 exercises the live clarify flow.

**Choice**: Option 2 — promote `tool_outputs` to a declared schema field before Phase 3.

**Why**: A declared field is checkpointed by contract, not by coincidence. The fix is three lines across two files and one schema addition; the risk it closes is a silent data loss bug in the only async path the demo depends on. Deferring this to Phase 3 would mean writing the API endpoint against a state contract with an unknown hole in it.

**Tradeoff Accepted**: `tool_outputs: dict` is an untyped dict in the schema (integer step-index keys, heterogeneous values), which is weaker than a typed mapping. Stronger typing (e.g. `dict[int, SummaryOutput | SentimentOutput | str | None]`) would require a Union that grows with every new tool — acceptable as a future Phase improvement, but not worth the coupling cost now.

---

### Decision 10: Temp File Cleanup Timing — On-Response vs Session-Expiry

**Problem**: Uploaded files are written to disk (per Decision 4, `raw_files` stores paths not bytes). Deciding when to delete them matters for correctness on the clarify-pause/resume path: if any extractor re-reads the file path after `extract_node` returns, deleting files on the initial `/session` response would silently break resume.

**Audit Performed (Phase 1 extractor code, not assumed)**:
- `extract_pdf`: opens `file_path` once via `pdfplumber.open()`. The Gemini vision fallback receives an in-memory `PIL.Image` rasterized from the page — the file path is not touched again.
- `extract_image`: opens `file_path` once via `PIL.Image.open()`. The Gemini vision fallback receives the in-memory `image` object. No second open.
- `extract_audio`: opens `file_path` once via `WhisperModel.transcribe(file_path)`. The Gemini cleanup pass receives the in-memory `raw_text` string. No second open.

After `extract_node` returns, no subsequent node (`fuse_node`, `planner_node`, `executor_node`, `formatter_node`) accesses `raw_files[].path`. All downstream processing operates on `extraction_results[].text` from the checkpointed state. The clarify-pause/resume path reads this checkpointed text on resume, not the original files.

**Options Considered**: (1) On-response cleanup: delete temp files immediately after the `/session` or `/reply` response is sent via a `BackgroundTask`. (2) Session-expiry cleanup: retain files until a TTL expires or an explicit delete call is made.

**Choice**: Option 1 (on-response cleanup via FastAPI `BackgroundTask`).

**Why**: Extraction completes inside `extract_node`, before the graph's first END boundary. Both the normal path and the clarify-pause path complete extraction before any checkpoint or response is returned. The extractor audit confirms no file re-reads occur after this point. On-response cleanup avoids the operational burden of a TTL-based expiry system under Render's free-tier constraints.

**Tradeoff Accepted**: If a future phase adds a node that re-reads `raw_files[].path` (e.g., a re-extraction node), this decision must be revisited and cleanup deferred to session-expiry. The audit must be repeated whenever a new node is added that handles file references.

---

### Decision 11: Verification of Graph Resume Mechanism (`aupdate_state` + `ainvoke(None)`)

**Problem**: Decision 10's safety argument for on-response temp file cleanup relies on the guarantee that the clarify-pause/resume flow resumes directly from `planner_node` without re-executing `extract_node` or re-reading `raw_files[].path`. If LangGraph's resume mechanism silently restarted execution from the graph entry point (`extract_node`), on-response cleanup would cause file-not-found errors upon session resumption.

**Empirical Verification & Testing**:
We added `test_real_graph_resume_does_not_re_extract` to `tests/test_api.py` and probed the compiled `StateGraph` under `langgraph 1.2+`.
1. Calling `await graph.aupdate_state(config, updates, as_node="fuse_context")` sets the execution cursor to `fuse_context` and updates `user_query` and `fused_context`.
2. Calling `await graph.ainvoke(None, config=config)` resumes execution directly from `planner_node` (the target of the edge out of `fuse_context`).
3. The execution trace confirms `extract_node` ran **exactly once** during the initial `/session` upload and **0 times** during the `/reply` call.

**Choice**: Use `await graph.aupdate_state(config, updates, as_node="fuse_context")` + `await graph.ainvoke(None, config=config)` for session resumption.

**Why**: Empirical testing with the real compiled graph and `MemorySaver` checkpointer proves that this mechanism resumes execution at `planner_node` without re-running `extract_node`. This validates the core assumption of Decision 10: temp files deleted on `/session` response do not cause failures during session resumption because `extract_node` is never re-entered.

**Tradeoff Accepted**: Requires using LangGraph's async API (`ainvoke`, `aget_state`, `aupdate_state`) in FastAPI async route handlers, as sync `.invoke()` on an async graph raises `TypeError` in LangGraph 1.2+.

> **Environment Locking Note**: Because this resume behavior was verified against `langgraph==1.2.10`, `langgraph-checkpoint==4.2.0`, `fastapi==0.141.1`, and `pdfplumber==0.11.10`, all dependency versions have been explicitly pinned in `requirements.txt` via `uv pip freeze`. This prevents Render's build environment from installing a different minor/patch release at deployment time that could alter internal resume semantics.

---

### Decision 12: 10 MB Per-File Upload Limit and Single-Worker Concurrency Assumption

**Problem**: Render's 512 MB free-tier ceiling (Decision 2) must accommodate the FastAPI process, uploads, extraction libraries, and the approximately 150 MB `faster-whisper` base/int8 footprint (Decision 5). Large or concurrent uploads can otherwise reproduce Decision 2's runtime OOM failure mode even when startup succeeds.

**Options Considered**: (1) Allow files up to Gemini's inline-payload limits (100 MB generally; 50 MB for PDFs). (2) Cap each uploaded file at 10 MB and stream it to disk in bounded chunks. (3) Add application-level request queueing/concurrency tests before deployment.

**Choice**: Option 2. Every file is limited to 10 MB, checked from a per-part `Content-Length` when present and otherwise enforced while streaming 64 KB chunks to disk; it is never read as one in-memory blob. The current Render deployment deliberately assumes one worker and does not claim that two or three simultaneous 10 MB extraction requests are safe under the 512 MB tier.

**Why**: Gemini's 100 MB inline-payload ceiling (and 50 MB PDF ceiling) is much higher, so it is not the binding constraint. The 10 MB limit exists to preserve deployment headroom on Render: Decision 5 already reserves roughly 150 MB for faster-whisper, while Decision 2 documents OOM/Exit 137 as the migration trigger. A bounded per-file limit prevents a single request from consuming an unbounded buffer, but it cannot prove aggregate safety under concurrent decoding and OCR. Render free tier's single-worker operating assumption keeps this phase deterministic and matches the deployment target; moving to multiple workers or sustained concurrency requires a queue/admission-control design and a load test before that change.

**Tradeoff Accepted**: Users with larger media files must split/compress them or use a higher-memory deployment tier. We intentionally document, rather than simulate, concurrent 10 MB requests because the free-tier single-worker configuration is the current operational boundary; a future multi-worker deployment must revisit this decision and add a concurrency test.

---

### Decision 13: Per-Session Gemini Client Construction & Secret Non-Persistence

**Problem**: Supporting custom user-supplied Gemini API keys via a UI Settings panel breaks the previous assumption of a single application-global `genai.Client()` constructed at server startup. Different requests may carry different keys (or rely on the server default), so the client instance must be scoped per request/session. Furthermore, storing raw API keys in `AgentState` breaks `MemorySaver` msgpack state serialization because `genai.Client` instances are non-serializable Python objects, and storing raw key strings in checkpointed state would persist user secrets in session state history.

**Options Considered**:
1. Store raw API key strings in `AgentState` and re-construct `genai.Client` inside every node.
2. Store `genai.Client` instances directly in `AgentState`.
3. Pass `gemini_client` dynamically per request via LangGraph's `config["configurable"]["gemini_client"]` execution context (`RunnableConfig`), keeping `AgentState` pure and msgpack-serializable.

**Choice**: Option 3 (Per-request client construction passed via `config["configurable"]`).

**Why**:
- **Security & Secret Non-Persistence**: The raw API key string is received via the `X-Gemini-Api-Key` HTTP header (or server `.env` fallback) and used immediately in-memory to instantiate a `genai.Client()`. The key is **never** written to disk, **never** written to temp upload directories, **never** logged in event logs (sanitized in `logging_utils.py`), and **never** stored in `AgentState` or `MemorySaver` checkpoints.
- **Serialization Safety**: `AgentState` remains lightweight, clean, and 100% msgpack-serializable, preventing `MemorySaver` serialization failures during state snapshotting and clarify-pause checkpoints.
- **Per-Session Flexibility**: Each `/session` and `/session/{thread_id}/reply` invocation resolves its own key from the incoming request header or server default, validates the key fast via a lightweight SDK ping (`count_tokens`), and passes the active client to nodes via runtime execution context.

**Tradeoff Accepted**: On clarify-resume (`/reply`), the client must send the `X-Gemini-Api-Key` header again if a custom key was used for the initial session. The frontend `app.js` handles this transparently by maintaining the custom key in JS memory for the active browser session.

