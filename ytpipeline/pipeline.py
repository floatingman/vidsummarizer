"""Pipeline orchestration: the 'run' (end-to-end), 'fetch' (audio only) and
'doctor' (environment check) commands."""

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from types import SimpleNamespace

from . import VERSION
from .common import EXTERNAL_TOOLS, PROG, log, resolve_workdir
from .summarize import normalize_host
from .transcribe import DEFAULT_MODEL_NAME, fetch_video_audio, transcribe_video


# --------------------------------------------------------------------------- #
# fetch: YouTube URL -> mp3 (no transcription)
# --------------------------------------------------------------------------- #

def cmd_fetch(args):
    workdir = resolve_workdir(args.workdir)
    t = SimpleNamespace(
        video=args.video, audio_dir=args.audio_dir,
        redownload=args.redownload, ytdlp=args.ytdlp,
    )
    meta, audio = fetch_video_audio(t, workdir)
    log(f"* audio ready: {audio} "
        f"({audio.stat().st_size // (1024 * 1024)} MiB)")
    return 0


# --------------------------------------------------------------------------- #
# run: YouTube URL -> mp3 -> transcript -> summary, end to end
# --------------------------------------------------------------------------- #

def cmd_run(args):
    workdir = resolve_workdir(args.workdir)
    t = SimpleNamespace(
        video=args.video,
        out=args.transcript_out,
        audio_dir=args.audio_dir,
        keep_audio=args.keep_audio,
        model=args.model,
        model_url=args.model_url,
        lang=args.lang,
        threads=args.threads,
        timestamps=args.timestamps,
        redownload=args.redownload,
        ytdlp=args.ytdlp,
    )
    transcript = transcribe_video(t, workdir)

    opts = SimpleNamespace(
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
    from .summarize import summarize_targets
    return summarize_targets([str(transcript)], opts)


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
