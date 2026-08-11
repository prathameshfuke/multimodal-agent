# Multimodal Agent (`multimodal-agent`)

A FastAPI-backed, LangGraph-orchestrated multimodal agentic assistant designed to process multi-format inputs (PDF documents, images, audio files, raw text), construct structured tool execution plans, execute tool workflows, and produce validated final outputs.

---

## 🏗️ Repository Layout

```
multimodal-agent/
├── app/
│   ├── schemas/          # Pydantic models & TypedDict AgentState
│   │   ├── extraction.py
│   │   ├── plan.py
│   │   ├── trace.py
│   │   ├── output.py
│   │   ├── state.py
│   │   └── __init__.py
│   ├── extractors/       # PDF, Image, Audio extraction modules
│   ├── graph/             # LangGraph state nodes & macro scheduler
│   ├── tools/             # Tool implementations & registry
│   └── main.py            # FastAPI entrypoint
├── tests/                # Pytest unit & integration tests
├── prompts/              # Versioned prompt templates (one file per tool)
├── decisions.md          # Architectural Decision Records (ADRs)
├── flow.md               # Graph execution flow documentation
├── requirements.txt      # Phase 0 & 1 Python dependencies
├── Dockerfile            # Container build specification (Python 3.11 + Tesseract + FFmpeg)
└── .env.example          # Environment variables template
```

---

## ⚡ Quickstart

### Local Setup

1. **Create and activate a virtual environment (Python 3.12.13)**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   ```bash
   cp .env.example .env
   ```

4. **Run FastAPI Server**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

   Access the interactive Swagger documentation at `http://localhost:8000/docs`.

---

## 🧪 Testing

Run pytest suite:
```bash
pytest tests/
```

---

## 🐳 Docker Deployment

Build and run container locally:
```bash
docker build -t multimodal-agent .
docker run -p 8000:8000 --env-file .env multimodal-agent
```

---

## 📜 Decisions & Architecture

Key architectural choices are logged in [`decisions.md`](file:///d:/Project/multiragas/multimodal-agent/decisions.md). Highlights:
- **LangGraph as Scheduler Only**: State machine transitions are governed by LangGraph; tool execution loops and retries are kept in plain Python for testability and clarity.
- **Render Free Tier & HF Fallback**: Render free tier is the default deployment target. Hugging Face Spaces is designated as the fallback if memory constraints cause container OOM (`Exit Code 137`).
- **State Serialization**: `UploadedFileRef` stores disk file paths rather than raw byte buffers to preserve lightweight state serialization for LangGraph checkpointers.
