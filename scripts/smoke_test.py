"""Run the live deployment checks: python scripts/smoke_test.py https://service.onrender.com"""

import argparse
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "tests" / "samples"
TIMEOUT_SECONDS = 180


def _post_session(base_url: str, query: str, uploads: list[tuple[str, Path]]) -> dict[str, Any]:
    with ExitStack() as stack:
        files = [
            ("files", (name, stack.enter_context(path.open("rb")), _mime_for(path)))
            for name, path in uploads
        ]
        response = requests.post(
            f"{base_url}/session", data={"user_query": query}, files=files, timeout=TIMEOUT_SECONDS
        )
    response.raise_for_status()
    return response.json()


def _mime_for(path: Path) -> str:
    return {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".wav": "audio/wav",
    }[path.suffix.lower()]


def _has_tool(body: dict[str, Any], tool_name: str) -> bool:
    return body.get("status") == "done" and any(
        event.get("tool_name") == tool_name for event in body.get("trace", [])
    )


def _case(name: str, run) -> bool:
    try:
        passed, detail = run()
        print(f"{'PASS' if passed else 'FAIL'}: {name} — {detail}")
        return passed
    except requests.RequestException as exc:
        print(f"FAIL: {name} — HTTP error: {exc}")
        return False
    except Exception as exc:
        print(f"FAIL: {name} — unexpected error: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Exercise the deployed multimodal-agent service.")
    parser.add_argument("base_url", nargs="?", default=os.getenv("SMOKE_BASE_URL"))
    args = parser.parse_args()
    if not args.base_url:
        parser.error("Pass the Render base URL or set SMOKE_BASE_URL.")
    base_url = args.base_url.rstrip("/")

    pdf = SAMPLES / "clean_sample.pdf"
    image = SAMPLES / "sample_image.png"
    audio = SAMPLES / "sample_audio.wav"
    youtube_pdf = SAMPLES / "pdf_with_youtube_url.pdf"

    cases = [
        ("health", lambda: ((r := requests.get(f"{base_url}/health", timeout=15)).ok, r.text)),
        ("TC1 PDF summary", lambda: _expect_done(_post_session(base_url, "Summarize this document.", [(pdf.name, pdf)]))),
        ("TC2 image analysis", lambda: _expect_done(_post_session(base_url, "Summarize the text in this image.", [(image.name, image)]))),
        ("TC3 PDF sentiment", lambda: _expect_tool(_post_session(base_url, "What is the sentiment of this document?", [(pdf.name, pdf)]), "sentiment")),
        ("TC4 YouTube transcript", lambda: _expect_tool(_post_session(base_url, "Summarize the linked YouTube video.", [(youtube_pdf.name, youtube_pdf)]), "fetch_youtube_transcript")),
        ("TC5 audio/PDF comparison", lambda: _expect_tool(_post_session(base_url, "Do these discuss the same topic?", [(pdf.name, pdf), (audio.name, audio)]), "cross_compare")),
        ("ambiguity clarification", lambda: _clarify_flow(base_url, pdf)),
    ]
    # Materialise results so one failed live case never hides later failures.
    results = [_case(name, run) for name, run in cases]
    return 0 if all(results) else 1


def _expect_done(body: dict[str, Any]) -> tuple[bool, str]:
    return body.get("status") == "done" and body.get("output") is not None, str(body.get("status"))


def _expect_tool(body: dict[str, Any], tool_name: str) -> tuple[bool, str]:
    return _has_tool(body, tool_name), str([event.get("tool_name") for event in body.get("trace", [])])


def _clarify_flow(base_url: str, pdf: Path) -> tuple[bool, str]:
    initial = _post_session(base_url, "Help me decide what to do with this document.", [(pdf.name, pdf)])
    if initial.get("status") != "awaiting_clarification":
        return False, f"expected awaiting_clarification, got {initial.get('status')}"
    thread_id = initial["thread_id"]
    response = requests.post(
        f"{base_url}/session/{thread_id}/reply",
        data={"reply": "Please summarize the document."},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    resumed = response.json()
    return resumed.get("status") == "done", str(resumed.get("status"))


if __name__ == "__main__":
    sys.exit(main())
