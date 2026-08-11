# Graph Execution Flow

## Node Sequence

```
extract_node  →  fuse_node  →  planner_node
                                    │
              ┌─────────────────────┤ route_after_plan
              │                     │
        clarify (END)         executor_node
   [user sees question,              │
    session suspended]        route_after_exec
                                     │
                             formatter_node → END
```

## Node Responsibilities

### 1. `extract_node`
Dispatches all uploaded files concurrently via `asyncio.gather`. Each file is routed
to `extract_pdf`, `extract_image`, or `extract_audio` based on its `content_type`.
Partial extraction failures are recorded as `TraceEvent(status="partial")` and do not
abort the run — the node always transitions to `status="planning"`.

### 2. `fuse_node`
Joins all `ExtractionResult.text` values with `\n\n---\n\n` separators, prepends the
`user_query`, and stores the result as `fused_context`. Runs URL regex
(`https?://[^\s<>"']+`) over all extracted text and the user query; deduplicates
results into `detected_urls`. No LLM call.

### 3. `planner_node`
Makes a single structured-output Gemini 2.5 Flash call with `planner_v1.txt`.
The prompt receives `fused_context`, `detected_urls`, and the full tool schema from
`get_tool_schema_for_planner()`. Returns a `Plan` with either `steps` (ordered
`ToolCall` list) or `clarify_question` — never both, never neither. On two consecutive
parse failures it falls back to a `clarify_question`.

### 4. `executor_node`
Iterates `plan.steps` in order, dispatching each to `dispatch_tool()`. Records a
`TraceEvent` per step. On first-attempt failure: one Reflexion-style retry with the
error message appended to `args["_retry_error_context"]`. On second failure: records
`status="partial"` and continues to the next step.

### 5. `formatter_node`
Reads `tool_outputs` (declared `AgentState` field populated by `executor_node`), identifies
the primary output type (`SummaryOutput`, `SentimentOutput`, or raw string/dict), and
normalises it into `FinalOutput`. Enforces the 3-bullet rule via truncation/padding
(Decision 3). Sets `task_type` explicitly from `plan.steps[0].tool_name`.

---

## Conditional Edges

### `route_after_plan(state) -> str`
- Returns `"clarify"` → `END` when `state["status"] == "awaiting_clarification"`.
- Returns `"execute"` → `execute_tools` otherwise.

### `route_after_exec(state) -> str`
- Always returns `"done"` → `format_output`.
  The retry loop is inside `executor_node` (plain Python, not a graph edge) per Decision 1.

---

## Clarify-Pause & Resume (End-to-End)

1. **User uploads files** → graph starts on a given `thread_id`.
2. **`planner_node`** cannot determine the primary task → sets
   `status="awaiting_clarification"`, `clarify_question="<question>"`.
3. **`route_after_plan`** returns `"clarify"` → the graph reaches `END`.
   MemorySaver checkpoints the full `AgentState` under `thread_id`.
4. **API layer** (Phase 3) returns `{"status": "awaiting_clarification", "question": "..."}` to the client.
5. **User answers** → client sends a follow-up request with the same `thread_id` and
   the answer appended as the new `user_query`.
6. **Graph resumes** from checkpoint using the same `thread_id`; `planner_node` now
   has the clarification and produces a `steps` plan.
7. Execution continues: `execute_tools` → `format_output` → `END`.

What persists across the pause: everything in `AgentState` — `extraction_results`,
`fused_context`, `detected_urls`, `trace`. The files are not re-extracted.

---

## Frontend User Experience & Interactive States (What the User Sees)

The single-page web UI (`app/static/index.html`, `style.css`, `app.js`) presents a clean 3-panel workspace:

1. **Initial Upload & Media Entry (`POST /session`)**:
   - The user drags and drops files (PDF, PNG/JPG images, WAV/MP3 audio) into the dropzone or selects them via file browser.
   - Selected file badges appear with filename and file size.
   - The user enters an optional text query or leaves it blank for automatic task detection.
   - Upon submitting, the status pill changes to **Processing** with a spinner.

2. **Unambiguous Run State (`status: "done"`)**:
   - The agent card renders the structured output:
     - **Summary**: One-line highlight, 3 key bullet points, and 5-sentence paragraph.
     - **Sentiment**: Label (`POSITIVE`, `NEGATIVE`, `NEUTRAL`), confidence percentage, and justification.
     - **Code Explanation**: Sectioned plain-text review.
   - A collapsible **"Plan & Tool Trace"** drawer expands to reveal execution telemetry:
     - Ordered planned steps.
     - Step-by-step trace table (`Step #`, `Tool Name`, `Status Badge`, `Latency ms`, `Output Summary`).
   - The right-hand inspector panel populates **Fused Extracted Context** and raw JSON trace (`GET /session/{thread_id}/trace`).

3. **Clarification Pause State (`status: "awaiting_clarification"`)**:
   - An amber clarification banner appears inline in the chat thread with the agent's specific question (e.g., *"What would you like me to do with this document?"*).
   - The status pill displays **Awaiting Clarification**.
   - The query input switches to **Clarification Reply** mode and locks the file dropzone.
   - The user types their answer and submits. The request is dispatched to `POST /session/{thread_id}/reply`.
   - Execution resumes directly from `planner_node` using checkpointed extraction state (0 file re-reads).

4. **Error Handling Path (`status: "error"`)**:
   - An error alert card renders distinctly from completed responses, displaying the error message and resetting the UI state safely for another attempt.

---

## Mermaid Architecture Diagram

> Generated by `scripts/export_graph.py` from the compiled graph.
> Run `python scripts/export_graph.py` to regenerate after any graph change.

*(See `flow_diagram.md` in the project root for the auto-generated Mermaid output.)*

