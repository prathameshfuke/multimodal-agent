"""Streaming upload validation; never materialise an entire upload in memory."""

from pathlib import Path

from fastapi import HTTPException, UploadFile


MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
SNIFF_BYTES = 4096
CHUNK_SIZE = 64 * 1024


def sniff_mime(header: bytes) -> str | None:
    """Recognise the binary types the extractors support from content signatures."""
    if header.startswith(b"%PDF-"):
        return "application/pdf"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return "audio/wav"
    if header.startswith(b"fLaC"):
        return "audio/flac"
    if header.startswith(b"OggS"):
        return "audio/ogg"
    if header.startswith(b"ID3") or header[:2] == b"\xff\xfb":
        return "audio/mpeg"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    return None


async def validate_and_store_upload(upload: UploadFile, destination: Path) -> str:
    """Validate one upload while streaming it to disk and return its sniffed MIME type."""
    declared_size = upload.headers.get("content-length")
    if declared_size and declared_size.isdigit() and int(declared_size) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(413, detail="Each uploaded file must be 10 MB or smaller.")

    total = 0
    header = bytearray()
    try:
        with destination.open("xb") as output:
            while chunk := await upload.read(CHUNK_SIZE):
                total += len(chunk)
                if total > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(413, detail="Each uploaded file must be 10 MB or smaller.")
                if len(header) < SNIFF_BYTES:
                    header.extend(chunk[: SNIFF_BYTES - len(header)])
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    if total == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(422, detail="Uploaded files must not be empty.")

    mime = sniff_mime(bytes(header))
    if mime is None:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            415,
            detail="Unsupported or invalid file content. Upload a PDF, supported image, or supported audio file.",
        )
    return mime
