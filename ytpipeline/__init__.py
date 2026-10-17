"""yt-pipeline: local YouTube transcription and summarization pipeline.

Merges the former yt-transcribe and yt-summarize tools into one portable
executable:

    YouTube URL -> yt-dlp (mp3) -> whisper.cpp transcript -> local-LLM summary

Everything runs locally; no cloud APIs, no API keys. Only the Python
standard library is used, so the whole tool ships as a single zipapp file.
"""

VERSION = "2.2.0"
