"""Batch processing: run a pipeline stage over a list of videos with a small
thread pool. All the heavy lifting happens in subprocesses (yt-dlp,
whisper-cli) and HTTP requests (Ollama), so plain threads are enough.

Transcription (whisper.cpp, CPU-bound) is typically faster than
summarization (local LLM), so even two workers overlap the stages nicely:
while one video is still being summarized, the next one is already
downloading/transcribing. A worker failure (bad id, dead video, empty
transcript, ...) is contained: the video is reported as failed and the rest
of the batch carries on; the process exit code reflects the failures.
"""

import concurrent.futures
import time

from .common import fmt_duration, log, set_log_tag, warn


def _guarded(video_id, fn, tag):
    """Run fn(video_id) for one worker thread, tagging its log lines."""
    if tag:
        set_log_tag(video_id)
    try:
        return fn(video_id)
    except SystemExit as e:  # die() in a worker must not kill the batch
        msg = e.code if isinstance(e.code, str) else f"aborted (exit {e.code})"
        raise RuntimeError(msg) from e
    finally:
        set_log_tag(None)


def _fmt_took(seconds):
    return fmt_duration(seconds) if seconds >= 60 else f"{seconds:.1f}s"


def run_batch(videos, fn, jobs, intro=None):
    """Run fn(video_id) for every video with at most `jobs` workers.

    Returns ({video_id: result}, [(video_id, error), ...]). With a single
    video the output is untagged (identical to the one-video command).
    """
    tag = len(videos) > 1
    if intro:
        log(intro)
    results, failures = {}, []
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_guarded, v, fn, tag): v for v in videos}
        for fut in concurrent.futures.as_completed(futures):
            video_id = futures[fut]
            try:
                results[video_id] = fut.result()
            except Exception as e:
                failures.append((video_id, e))
    if tag:
        if failures:
            warn(f"{len(failures)}/{len(videos)} video(s) failed: "
                 + ", ".join(v for v, _ in failures))
        log(f"* batch finished in {_fmt_took(time.monotonic() - t0)}: "
            f"{len(results)}/{len(videos)} video(s) ok")
    return results, failures
