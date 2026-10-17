"""Pipeline orchestration: the 'run' (end-to-end), 'fetch' (audio only),
'transcribe' (audio -> transcript) and 'doctor' (environment check)
commands. All video-taking commands accept a list (comma-separated
URLs/ids, several arguments, and/or a --batch file) and process it with a
small worker pool (see batch.py)."""

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from . import VERSION
from .batch import run_batch
from .common import (EXTERNAL_TOOLS, PROG, collect_videos, die, log,
                     require_tools, resolve_workdir)
from .summarize import Ollama, normalize_host, summarize_targets
from .transcribe import (DEFAULT_MODEL_NAME, ensure_model, fetch_video_audio,
                         transcribe_video)

TRANSCRIBE_TOOLS = ["yt-dlp", "ffmpeg", "whisper-cli"]
DEFAULT_WHISPER_THREADS = 8


# --------------------------------------------------------------------------- #
# Batch helpers
# --------------------------------------------------------------------------- #

def resolve_threads(args, jobs):
    """Whisper CPU threads per job. Unless -t was given explicitly, the
    default of 8 is split across the jobs that can actually run at once
    (at least 2 each) so a batch run keeps total CPU use near that of a
    single-video run."""
    if getattr(args, "threads", None) is not None:
        return args.threads
    return DEFAULT_WHISPER_THREADS if jobs <= 1 \
        else max(2, DEFAULT_WHISPER_THREADS // jobs)


def _in_flight(jobs, videos):
    """Jobs that can actually run concurrently for this video list."""
    return min(jobs, len(videos))


def _batch_intro(what, videos, jobs, threads=None):
    msg = f"* batch: {what} {len(videos)} video(s), {jobs} at a time"
    if threads:
        msg += f", {threads} whisper thread(s) each"
    return msg


def _transcribe_ns(args, video_id, threads, out):
    return SimpleNamespace(
        video=video_id,
        out=out,
        audio_dir=args.audio_dir,
        keep_audio=args.keep_audio,
        model=args.model,
        model_url=args.model_url,
        lang=args.lang,
        threads=threads,
        timestamps=args.timestamps,
        redownload=args.redownload,
        ytdlp=args.ytdlp,
    )


# --------------------------------------------------------------------------- #
# fetch: YouTube URL -> mp3 (no transcription)
# --------------------------------------------------------------------------- #

def cmd_fetch(args):
    workdir = resolve_workdir(args.workdir)
    videos = collect_videos(args.videos, args.batch)
    jobs = args.jobs

    def one(video_id):
        t = SimpleNamespace(
            video=video_id, audio_dir=args.audio_dir,
            redownload=args.redownload, ytdlp=args.ytdlp,
        )
        meta, audio = fetch_video_audio(t, workdir)
        log(f"* audio ready: {audio} "
            f"({audio.stat().st_size // (1024 * 1024)} MiB)")
        return str(audio)

    if len(videos) == 1 and jobs == 1:
        one(videos[0])
        return 0
    require_tools(["yt-dlp", "ffmpeg"])
    _, failures = run_batch(videos, one, jobs,
                            intro=_batch_intro("fetching", videos, jobs))
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# transcribe: YouTube URL -> transcript only
# --------------------------------------------------------------------------- #

def cmd_transcribe(args):
    workdir = resolve_workdir(args.workdir)
    videos = collect_videos(args.videos, args.batch)
    jobs = args.jobs
    threads = resolve_threads(args, _in_flight(jobs, videos))
    if args.out and len(videos) > 1:
        die("-o/--out writes one transcript file; drop it for batch runs "
            "(each video gets transcripts/<video_id>.txt)")

    def one(video_id):
        return str(transcribe_video(_transcribe_ns(args, video_id, threads,
                                                   args.out), workdir))

    if len(videos) == 1 and jobs == 1:
        one(videos[0])
        return 0
    require_tools(TRANSCRIBE_TOOLS)
    ensure_model(Path(args.model) if args.model
                 else workdir / "models" / DEFAULT_MODEL_NAME, args.model_url)
    _, failures = run_batch(
        videos, one, jobs,
        intro=_batch_intro("transcribing", videos, jobs,
                           threads if threads != DEFAULT_WHISPER_THREADS else None)
        if len(videos) > 1 else None)
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# run: YouTube URL -> mp3 -> transcript -> summary, end to end
# --------------------------------------------------------------------------- #

def cmd_run(args):
    workdir = resolve_workdir(args.workdir)
    videos = collect_videos(args.videos, args.batch)
    jobs = args.jobs
    if args.transcript_out and len(videos) > 1:
        die("-o/--transcript-out writes one transcript file; drop it for "
            "batch runs (each video gets transcripts/<video_id>.txt)")
    if args.stdout and len(videos) > 1 and jobs > 1:
        log("* --stdout with several videos: using --jobs 1 so the notes "
            "print in order")
        jobs = 1
    threads = resolve_threads(args, _in_flight(jobs, videos))

    def summarize_opts():
        return SimpleNamespace(
            out=args.out or str(workdir / "summaries"),
            model=args.llm_model,
            host=args.host,
            timeout=args.timeout,
            no_autostart=args.no_autostart,
            think=args.think,
            max_words=args.max_words,
            lang=args.summary_lang,
            chunk_words=args.chunk_words,
            notes_words=args.notes_words,
            max_num_ctx=args.max_num_ctx,
            force=args.force,
            stdout=args.stdout,
        )

    def one(video_id, ollama=None):
        t = _transcribe_ns(args, video_id, threads, args.transcript_out)
        transcript = transcribe_video(t, workdir)
        return summarize_targets([str(transcript)], summarize_opts(),
                                  ollama=ollama)

    if len(videos) == 1 and jobs == 1:  # exact single-video behavior of 2.0.x
        return one(videos[0])

    # Batch pre-flight, once, before any downloads start: external tools,
    # the whisper model (so only one worker can ever download it), and the
    # Ollama server + model (shared client; no autostart race between jobs).
    require_tools(TRANSCRIBE_TOOLS)
    ensure_model(Path(args.model) if args.model
                 else workdir / "models" / DEFAULT_MODEL_NAME, args.model_url)
    opts = summarize_opts()
    ollama = Ollama(opts.host, opts.model, opts.timeout,
                    no_autostart=opts.no_autostart)
    ollama.check_model()

    def work(video_id):
        return one(video_id, ollama=ollama)

    results, failures = run_batch(
        videos, work, jobs,
        intro=_batch_intro("processing", videos, jobs,
                           threads if threads != DEFAULT_WHISPER_THREADS else None)
        if len(videos) > 1 else None)
    # summarize_targets reports per-video failures as a non-zero return value
    soft_failures = [v for v, rc in results.items() if rc]
    return 1 if failures or soft_failures else 0


# --------------------------------------------------------------------------- #
# doctor: report what this machine has and what is missing
# --------------------------------------------------------------------------- #

def _tool_version(cmd):
    """Best-effort version string for an external CLI, or None."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              errors="replace", timeout=15)
    except Exception:
        return None
    text = (proc.stdout or "") + (proc.stderr or "")
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    if proc.returncode == 0:
        return first or None
    return None  # tools without a --version flag are skipped


def cmd_doctor(args):
    host = normalize_host(args.host)
    llm_model = args.llm_model
    workdir = resolve_workdir(args.workdir)
    model_path = workdir / "models" / DEFAULT_MODEL_NAME

    def ok(present):
        return "[AUTO]" if present is None else ("[ OK ]" if present else "[MISS]")
    problems = []

    print(f"{PROG} {VERSION} doctor: checking this machine for the local pipeline")
    print()

    # -- core ---------------------------------------------------------------
    py = sys.version_info
    py_ok = py >= (3, 9)
    print("Core:")
    print(f"  {ok(py_ok)} Python {sys.version.split()[0]} ({sys.executable})"
          + ("" if py_ok else "  needs 3.9+"))
    if not py_ok:
        problems.append("python")
    print()

    # -- audio + transcription ----------------------------------------------
    print("Audio + transcription (commands: run, fetch, transcribe):")
    for name in ("yt-dlp", "ffmpeg", "whisper-cli"):
        path = shutil.which(name)
        if not path:
            problems.append(name)
            purpose, hint = EXTERNAL_TOOLS[name]
            print(f"  {ok(False)} {name} ({purpose})")
            print(f"         install: {hint}")
        else:
            ver = None
            if name == "yt-dlp":
                ver = _tool_version(["yt-dlp", "--version"])
            elif name == "ffmpeg":
                v = _tool_version(["ffmpeg", "-version"]) or ""
                # "ffmpeg version 7.1.1-..." -> "7.1.1"
                m = v.split()
                ver = m[2] if "version" in m and len(m) > 2 else v[:24]
            print(f"  {ok(True)} {name}{' ' + ver if ver else ''}"
                  f"  ({path})")
    if model_path.exists():
        size = model_path.stat().st_size
        print(f"  {ok(True)} whisper model {model_path.name} "
              f"({size / (1024 * 1024):.0f} MiB) at {model_path}")
    else:
        print(f"  {ok(None)} whisper model {model_path.name} at {model_path}")
        print("         (downloaded automatically on first use, "
              "~466 MiB from huggingface.co/ggerganov/whisper.cpp)")
    print()

    # -- summaries ----------------------------------------------------------
    print("Summaries (commands: run, summarize):")
    sum_missing = False
    opath = shutil.which("ollama")
    if not opath:
        sum_missing = True
        problems.append("ollama")
        purpose, hint = EXTERNAL_TOOLS["ollama"]
        print(f"  {ok(False)} ollama ({purpose})")
        print(f"         install: {hint}")
    else:
        ver = (_tool_version(["ollama", "--version"]) or "").replace(
            "ollama version is ", "").strip()
        print(f"  {ok(True)} ollama{' ' + ver if ver else ''}  ({opath})")
        # server reachable?
        server_ok, server_ver, model_ok = False, None, None
        try:
            with urllib.request.urlopen(host + "/api/version", timeout=3) as r:
                server_ver = json.load(r).get("version")
            server_ok = True
        except Exception as e:
            print(f"  {ok(False)} Ollama server at {host}")
            print(f"         not reachable ({e}); it is started automatically "
                  f"by 'run'/'summarize' via 'ollama serve'")
        if server_ok:
            print(f"  {ok(True)} Ollama server at {host} "
                  f"(v{server_ver}, {'running' if opath else 'remote'})")
            try:
                with urllib.request.urlopen(host + "/api/tags", timeout=5) as r:
                    names = [m.get("name", "") for m in json.load(r).get("models", [])]
                model_ok = any(n == llm_model or n.split(":")[0] == llm_model
                               for n in names)
                if model_ok:
                    print(f"  {ok(True)} Ollama model '{llm_model}' installed")
                else:
                    problems.append(f"model:{llm_model}")
                    others = f" (installed: {', '.join(names)})" if names else ""
                    print(f"  {ok(False)} Ollama model '{llm_model}'{others}")
                    print(f"         install: ollama pull {llm_model}")
            except Exception as e:
                print(f"  {ok(False)} could not list Ollama models: {e}")
    print()

    # -- data ---------------------------------------------------------------
    print("Data:")
    try:
        workdir.mkdir(parents=True, exist_ok=True)
        writable = os.access(workdir, os.W_OK)
    except OSError:
        writable = False
    print(f"  {ok(writable)} workdir {workdir} "
          f"({'writable' if writable else 'NOT writable; use --workdir DIR'})")
    print("         audio/, transcripts/, summaries/, models/ live here")
    if not writable:
        problems.append("workdir")
    print()

    # -- verdict ------------------------------------------------------------
    if not problems:
        print("Result: everything present; the full pipeline can run on this machine.")
        return 0
    missing = [p for p in problems if p in EXTERNAL_TOOLS or p.startswith("model:")]
    can = []
    if not {"yt-dlp", "ffmpeg"} & set(missing):
        can.append("fetch")
    if not ({"yt-dlp", "ffmpeg", "whisper-cli"} & set(missing)):
        can.append("transcribe")
    if not sum_missing:
        can.append("summarize")
    print(f"Result: missing {', '.join(problems)}.")
    if can:
        print(f"Until then only these commands can run: {', '.join(can)}.")
    else:
        print("No pipeline stage can run until the items above are installed.")
    print("Fix the [MISS] items (install hints above) and re-run "
          f"'{PROG} doctor'.")
    return 1
