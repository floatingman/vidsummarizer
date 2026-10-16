"""Transcription stage, ported from yt-transcribe 1.0.0.

Pipeline: YouTube URL/id -> yt-dlp (mp3 audio) -> whisper.cpp (whisper-cli)
-> plain-text transcript.

Output is plain UTF-8 text designed for consumption by language models:
a short `key: value` metadata header, a blank line, then the transcript as
clean paragraphs (no timestamps by default, no markup). See README.md for
the exact format specification.

External dependencies (must be on PATH, checked at runtime): yt-dlp,
ffmpeg, whisper-cli. The whisper GGML model is downloaded automatically on
first use if missing.
"""

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from . import VERSION
from .common import die, fmt_duration, fmt_ts, log, parse_video_id, require_tools, run, watch_url

DEFAULT_MODEL_NAME = "ggml-small.bin"
MODEL_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

# Paragraphing rules (documented in README.md)
PARAGRAPH_GAP_SECONDS = 1.5   # silence gap between segments that starts a new paragraph
MAX_PARAGRAPH_WORDS = 280     # soft cap: after this, break at the next sentence-ending segment
HARD_PARAGRAPH_WORDS = 500    # absolute cap for speech with no sentence punctuation
SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")


def ytdlp_base_args():
    """yt-dlp needs a JS runtime for reliable YouTube extraction; enable node if no deno."""
    import shutil
    if shutil.which("deno"):
        return []
    if shutil.which("node"):
        return ["--js-runtimes", "node"]
    return []


def fetch_metadata(video_id, ytdlp_extra):
    cmd = ["yt-dlp", "-J", "--no-playlist", *ytdlp_base_args(), *ytdlp_extra, watch_url(video_id)]
    proc = run(cmd, "yt-dlp metadata lookup")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        die("yt-dlp returned metadata that could not be parsed as JSON")


def download_audio(video_id, audio_dir, ytdlp_extra, redownload):
    audio_dir.mkdir(parents=True, exist_ok=True)
    mp3 = audio_dir / f"{video_id}.mp3"
    if mp3.exists() and mp3.stat().st_size > 0 and not redownload:
        log(f"* reusing existing audio: {mp3}")
        return mp3
    out_tmpl = str(audio_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-q", "--no-warnings", "--no-playlist",
        *ytdlp_base_args(),
        "-x", "--audio-format", "mp3", "--audio-quality", "0",
        "-o", out_tmpl,
        *ytdlp_extra,
        watch_url(video_id),
    ]
    for attempt in range(1, 4):  # YouTube occasionally throttles with transient HTTP 403
        try:
            run(cmd, f"yt-dlp audio download (attempt {attempt}/3)")
            break
        except RuntimeError as e:
            log(f"* {e}")
            if attempt == 3:
                raise SystemExit(1)
            log("* waiting 8s before retrying...")
            time.sleep(8)
    if not mp3.exists():
        die(f"expected audio file was not created: {mp3}")
    return mp3


def ensure_model(model_path, model_url=None):
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path
    url = model_url or f"{MODEL_BASE_URL}/{model_path.name}"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_suffix(model_path.suffix + ".part")
    log(f"Downloading whisper model {model_path.name} (one-time setup)...")

    def progress(count, block_size, total):
        if total > 0:
            pct = min(100, count * block_size * 100 // total)
            print(f"\r  {pct:3d}% ({count * block_size // (1024 * 1024)} MiB)", end="",
                  file=sys.stderr)

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=progress)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        die(f"model download failed from {url}: {e}")
    print(file=sys.stderr)
    tmp.rename(model_path)
    return model_path


def run_whisper(model, audio, workdir, lang, threads):
    """Transcribe with whisper-cli; returns the parsed JSON result."""
    workdir.mkdir(parents=True, exist_ok=True)

    def attempt(media, fatal=True):
        outbase = workdir / media.stem
        cmd = [
            "whisper-cli",
            "-m", str(model),
            "-f", str(media),
            "-l", lang,
            "-t", str(threads),
            "-oj", "-of", str(outbase),
        ]
        run(cmd, f"whisper-cli on {media.name}", fatal=fatal)
        out = outbase.with_suffix(".json")
        if not out.exists():
            msg = f"whisper-cli did not produce expected output: {out}"
            if fatal:
                die(msg)
            raise RuntimeError(msg)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        out.unlink(missing_ok=True)  # keep only the final transcript as an artifact
        return data

    try:
        return attempt(audio, fatal=False)  # whisper.cpp (miniaudio) reads mp3 directly
    except RuntimeError as e:
        # Fallback: transcode to 16 kHz mono WAV and retry once.
        log(f"* {e}")
        log("* mp3 decode failed, retrying with 16 kHz WAV conversion...")
        wav = workdir / f"{audio.stem}.16k.wav"
        run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio),
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
            "ffmpeg mp3->wav conversion",
        )
        return attempt(wav)


def normalize_segments(result):
    """Map whisper-cli JSON (new and legacy schemas) to [{start, end, text}] in seconds."""
    segs = []
    for seg in result.get("transcription") or []:  # whisper.cpp >= 1.7 unified JSON
        if isinstance(seg, dict):
            off = seg.get("offsets", {})
            segs.append({
                "start": (off.get("from") or 0) / 1000.0,
                "end": (off.get("to") or 0) / 1000.0,
                "text": seg.get("text", ""),
            })
    if not segs:  # legacy schema: result.segments[{start, end, text}]
        segs = result.get("result", {}).get("segments", [])
    return segs


def build_paragraphs(segments, timestamps):
    """Turn whisper segments into clean paragraphs.

    - New paragraph when the silence gap between consecutive segments is
      >= PARAGRAPH_GAP_SECONDS, or when the current paragraph has passed the
      soft word cap and the last segment ended a sentence (so paragraphs
      never split mid-sentence unless the HARD cap is hit).
    - Segment texts are whitespace-collapsed and joined with single spaces.
    """
    paras = []  # [start_time, [texts], word_count]
    prev_end = None
    for seg in segments:
        text = re.sub(r"\s+", " ", str(seg.get("text", ""))).strip()
        if not text:
            continue
        start, end = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        words = len(text.split())
        gap = start - prev_end if prev_end is not None else None
        cur = paras[-1] if paras else None
        new_para = (
            not paras
            or (gap is not None and gap >= PARAGRAPH_GAP_SECONDS)
            or (cur[2] >= MAX_PARAGRAPH_WORDS and SENTENCE_END.search(cur[1][-1].strip()))
            or cur[2] >= HARD_PARAGRAPH_WORDS
        )
        if new_para:
            paras.append([start, [text], words])
        else:
            cur[1].append(text)
            cur[2] += words
        prev_end = end

    out = []
    for start, texts, _ in paras:
        para = " ".join(texts)
        out.append(f"[{fmt_ts(start)}] {para}" if timestamps else para)
    return out


def build_output(meta, video_id, language, model, paragraphs):
    header = [
        ("title", str(meta.get("title") or "").strip()),
        ("channel", str(meta.get("channel") or meta.get("uploader") or "").strip()),
        ("upload_date", str(meta.get("upload_date") or "")),
        ("duration", fmt_duration(meta.get("duration"))),
        ("video_id", video_id),
        ("url", watch_url(video_id)),
        ("language", language),
        ("model", model.name),
        ("tool", f"yt-pipeline {VERSION} (local whisper.cpp)"),
    ]
    if len(header[2][1]) == 8:  # YYYYMMDD -> YYYY-MM-DD
        d = header[2][1]
        header[2] = ("upload_date", f"{d[:4]}-{d[4:6]}-{d[6:]}")
    lines = [f"{k}: {v}" for k, v in header if v]
    return "\n".join(lines) + "\n\n" + "\n\n".join(paragraphs) + "\n"


# --------------------------------------------------------------------------- #
# Core entry points (shared by the 'fetch', 'transcribe' and 'run' commands)
# --------------------------------------------------------------------------- #

def fetch_video_audio(t, workdir):
    """Download (or reuse) the mp3 for t.video; returns (meta, audio_path)."""
    require_tools(["yt-dlp", "ffmpeg"])
    video_id = parse_video_id(t.video)
    ytdlp_extra = re.findall(r"\S+", t.ytdlp) if t.ytdlp else []
    meta = fetch_metadata(video_id, ytdlp_extra)
    log(f"* {meta.get('title', video_id)} "
        f"[{fmt_duration(meta.get('duration')) or '?'}] -> downloading audio")
    audio = download_audio(video_id, Path(t.audio_dir or (workdir / "audio")),
                           ytdlp_extra, t.redownload)
    return meta, audio


def transcribe_video(t, workdir):
    """Full transcription pass; returns the Path of the written transcript.

    t must have fields: video, out (None ok), audio_dir (None ok),
    keep_audio, model (None ok), model_url, lang, threads, timestamps,
    redownload, ytdlp.
    """
    require_tools(["yt-dlp", "ffmpeg", "whisper-cli"])
    video_id = parse_video_id(t.video)
    ytdlp_extra = re.findall(r"\S+", t.ytdlp) if t.ytdlp else []

    meta = fetch_metadata(video_id, ytdlp_extra)
    log(f"* {meta.get('title', video_id)} "
        f"[{fmt_duration(meta.get('duration')) or '?'}] -> downloading audio")
    audio_dir = Path(t.audio_dir) if t.audio_dir else workdir / "audio"
    audio = download_audio(video_id, audio_dir, ytdlp_extra, t.redownload)

    model = ensure_model(Path(t.model) if t.model
                         else workdir / "models" / DEFAULT_MODEL_NAME, t.model_url)
    log(f"* transcribing {audio.name} with {model.name} (lang={t.lang})")
    t0 = time.monotonic()
    result = run_whisper(model, audio, audio_dir, t.lang, t.threads)
    log(f"* transcription finished in {time.monotonic() - t0:.1f}s")

    res = result.get("result", {})
    segments = normalize_segments(result)
    if not segments:
        die("whisper-cli returned no segments (is the audio silent or unreadable?)")
    detected = str(res.get("language") or "").strip()
    language = detected or (t.lang if t.lang != "auto" else "")

    paragraphs = build_paragraphs(segments, t.timestamps)
    output = build_output(meta, video_id, language, model, paragraphs)

    out = Path(t.out) if t.out else workdir / "transcripts" / f"{video_id}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(output, encoding="utf-8")

    if not t.keep_audio:
        audio.unlink(missing_ok=True)

    words = sum(len(p.split()) for p in paragraphs)
    log(f"* wrote {out} ({words} words, {len(paragraphs)} paragraphs)")
    return out
