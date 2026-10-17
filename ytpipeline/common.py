"""Shared helpers for yt-pipeline: logging, subprocess wrapper, video-id
parsing, external-tool detection, and work-directory resolution."""

import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from . import VERSION

PROG = "yt-pipeline"
VERSION_LINE = f"{PROG} {VERSION}"

# External tools that cannot be bundled into the executable.
# name -> (what it does in the pipeline, how to install it on the target machine)
EXTERNAL_TOOLS = {
    "yt-dlp": (
        "downloads audio and metadata from YouTube",
        "pipx install yt-dlp  (or: pip install --user yt-dlp, or download the "
        "standalone binary from https://github.com/yt-dlp/yt-dlp)",
    ),
    "ffmpeg": (
        "encodes the downloaded audio to mp3 and converts it to wav when needed",
        "install with your system package manager, e.g.  sudo apt install ffmpeg",
    ),
    "whisper-cli": (
        "transcribes the audio locally (whisper.cpp)",
        "build whisper.cpp from https://github.com/ggml-org/whisper.cpp and put "
        "the whisper-cli binary on PATH (cmake -B build && cmake --build build "
        "-j --target whisper-cli)",
    ),
    "ollama": (
        "serves the local LLM that writes the summaries",
        "install from https://ollama.com, then pull a model with: "
        "ollama pull qwen3.6",
    ),
}

ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


# --------------------------------------------------------------------------- #
# Logging (thread-safe; batch mode tags each line with the video id)
# --------------------------------------------------------------------------- #

_print_lock = threading.Lock()
_log_tag = threading.local()


def set_log_tag(tag):
    """Tag every log/warn/error line from the calling thread with `tag`
    (batch mode uses the video id so concurrent videos stay readable)."""
    _log_tag.value = tag


def _tagged(msg):
    tag = getattr(_log_tag, "value", None)
    if tag and not msg.startswith("["):
        return f"[{tag}] {msg}"
    return msg


def die(msg, code=1):
    with _print_lock:
        print(f"{PROG}: error: {_tagged(msg)}", file=sys.stderr)
    sys.exit(code)


def warn(msg):
    with _print_lock:
        print(f"{PROG}: warning: {_tagged(msg)}", file=sys.stderr)


def log(msg):
    with _print_lock:
        print(_tagged(msg), file=sys.stderr)


# --------------------------------------------------------------------------- #
# Subprocess helper
# --------------------------------------------------------------------------- #

def run(cmd, what, tail=15, fatal=True):
    """Run a command; on failure print the tail of its stderr and exit (or raise)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    except FileNotFoundError:
        die(f"{what}: command not found: {cmd[0]}")
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        msg = f"{what} failed:\n  " + "\n  ".join(err[-tail:])
        if fatal:
            die(msg)
        raise RuntimeError(msg)
    return proc


# --------------------------------------------------------------------------- #
# Video ids
# --------------------------------------------------------------------------- #

def parse_video_id(arg):
    """Accept a bare 11-char video id or any common YouTube URL form."""
    arg = arg.strip()
    if ID_RE.match(arg):
        return arg
    m = (
        re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", arg)
        or re.search(r"[?&]v=([A-Za-z0-9_-]{11})", arg)
        or re.search(r"youtube\.com/(?:shorts|live|embed|v)/([A-Za-z0-9_-]{11})", arg)
    )
    if m:
        return m.group(1)
    die(f"could not extract an 11-character YouTube video id from: {arg!r}")


def watch_url(video_id):
    return f"https://www.youtube.com/watch?v={video_id}"


# --------------------------------------------------------------------------- #
# Video lists (batch input)
# --------------------------------------------------------------------------- #

def split_video_tokens(tokens):
    """Flatten comma-separated URL/id tokens into a list of non-empty ones."""
    out = []
    for tok in tokens or []:
        for part in str(tok).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def read_batch_file(path):
    """Read a batch file: one video URL/id per line (commas also separate),
    blank lines and # comments ignored. '-' reads from stdin."""
    if path == "-":
        text = sys.stdin.read()
    else:
        try:
            text = Path(path).expanduser().read_text(encoding="utf-8")
        except OSError as e:
            die(f"cannot read batch file {path}: {e}")
    lines = (ln.split("#", 1)[0] for ln in text.splitlines())  # strip comments
    return split_video_tokens(lines)


def collect_videos(positional, batch_files):
    """Merge positional tokens and --batch files into a validated list of
    video ids (dies early on junk so a typo costs no downloads; duplicate
    entries are dropped with a warning)."""
    tokens = split_video_tokens(positional)
    for bf in batch_files or []:
        tokens += read_batch_file(bf)
    if not tokens:
        die("no videos given: pass URL(s)/id(s) (comma-separated or as several "
            "arguments) and/or --batch FILE with one video per line")
    ids = [parse_video_id(t) for t in tokens]
    seen, out, dupes = set(), [], []
    for vid in ids:
        if vid in seen:
            if vid not in dupes:
                dupes.append(vid)
            continue
        seen.add(vid)
        out.append(vid)
    if dupes:
        warn(f"ignoring duplicate video(s): {', '.join(dupes)}")
    return out


# --------------------------------------------------------------------------- #
# Time formatting
# --------------------------------------------------------------------------- #

def fmt_ts(seconds):
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def fmt_duration(seconds):
    if seconds is None:
        return ""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


# --------------------------------------------------------------------------- #
# External-tool detection
# --------------------------------------------------------------------------- #

def missing_tools(names):
    return [n for n in names if shutil.which(n) is None]


def require_tools(names):
    """Exit with a clear message about every missing external tool."""
    missing = missing_tools(names)
    if not missing:
        return
    lines = ["this machine is missing required external tool(s):"]
    for n in missing:
        purpose, hint = EXTERNAL_TOOLS[n]
        lines.append(f"  - {n} ({purpose})")
        lines.append(f"    install: {hint}")
    lines.append("(all tools must be on PATH; run "
                 f"'{PROG} doctor' for a full environment check)")
    die("\n".join(lines))


# --------------------------------------------------------------------------- #
# Work directory (where audio/, transcripts/, summaries/, models/ live)
# --------------------------------------------------------------------------- #

def base_dir():
    """Directory containing the single-file executable (zipapp), or the repo
    root when running from a source checkout."""
    argv0 = sys.argv[0] if sys.argv and sys.argv[0] else ""
    if argv0:
        p = Path(argv0).resolve()
        try:
            if p.is_file() and p.name != "__main__.py" and p.suffix != ".py":
                return p.parent
        except OSError:
            pass
    return Path(__file__).resolve().parent.parent


def fallback_data_dir():
    xdg = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(xdg).expanduser() / "yt-pipeline"


def resolve_workdir(explicit=None):
    """--workdir > $YT_WORKDIR > directory of the executable > XDG data dir.

    The old tools kept audio/, transcripts/, summaries/ and models/ next to
    their script in the project checkout; the executable's own directory is
    the direct equivalent, so a copied yt-pipeline file plus its data folder
    behaves exactly like the old checkouts. If that directory is read-only
    (e.g. the binary was installed into /usr/local/bin), fall back to
    ~/.local/share/yt-pipeline so the tool still works anywhere.
    """
    if explicit:
        return Path(explicit).expanduser()
    if os.environ.get("YT_WORKDIR"):
        return Path(os.environ["YT_WORKDIR"]).expanduser()
    wd = base_dir()
    if not os.access(wd, os.W_OK):
        alt = fallback_data_dir()
        log(f"* {wd} is not writable; storing data in {alt}")
        log("* (override with --workdir DIR or $YT_WORKDIR)")
        return alt
    return wd
