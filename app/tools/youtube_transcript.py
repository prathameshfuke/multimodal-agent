import re


def _extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
        r"(?:shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


async def fetch_youtube_transcript(url: str) -> str:
    video_id = _extract_video_id(url)
    if not video_id:
        return f"Could not extract a YouTube video ID from the URL: {url}"

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            TranscriptsDisabled,
            NoTranscriptFound,
            VideoUnavailable,
        )

        # youtube-transcript-api 1.x exposes a client instance and fetch(); older
        # get_transcript() examples would fail for every otherwise-valid URL.
        transcript = YouTubeTranscriptApi().fetch(video_id)
        return " ".join(snippet.text for snippet in transcript)

    except Exception as e:
        error_name = type(e).__name__
        if "TranscriptsDisabled" in error_name:
            return f"Transcripts are disabled for video {video_id}. No captions are available."
        if "NoTranscriptFound" in error_name:
            return f"No transcript found for video {video_id}. The video may not have captions."
        if "VideoUnavailable" in error_name:
            return f"Video {video_id} is unavailable or private."
        return f"Could not fetch transcript for video {video_id}: {str(e)}"
