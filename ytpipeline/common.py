"""Shared helpers for yt-pipeline: logging, subprocess wrapper, video-id
parsing, title-based output naming, external-tool detection, and
work-directory resolution."""

import contextlib
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix (no flock)
    fcntl = None

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
# Title-based output naming (transcripts/<Title>.txt, summaries/<Title>.md)
# --------------------------------------------------------------------------- #

# Characters hostile to filenames: path separators and wildcards on Linux,
# the set Windows forbids, plus what Obsidian disallows in note names.
UNSAFE_FILENAME = re.compile(r'[\\/:*?"<>|#^\[\]\x00-\x1f]')

# The video id embedded in a transcript's `key: value` header or a summary
# note's YAML frontmatter (quoted there); either way near the top of the file.
VIDEO_ID_HEAD_RE = re.compile(r'^video_id:\s*"?([A-Za-z0-9_-]{11})"?', re.M)

MAX_TITLE_CHARS = 90   # sane cap for a filename built from a runaway title
MAX_TITLE_BYTES = 200  # stays under Linux/macOS 255-byte filename limits
                        # even for multibyte titles, with room for the
                        # " (<id>)" collision suffix and the extension

# One output-name resolution at a time within a process: batch workers
# transcribing videos that share a sanitized title must not race the
# exists-check against the write (which would silently overwrite the first
# video's file). output_dir_lock() adds the cross-process equivalent.
naming_lock = threading.Lock()


def safe_name(title, max_len=MAX_TITLE_CHARS):
    """Filesystem-safe file stem for a video title: drop characters that
    are hostile to filenames on Linux (and reasonably portable elsewhere),
    collapse whitespace, trim surrounding dots (Windows) and cap the length
    in characters and UTF-8 bytes."""
    name = UNSAFE_FILENAME.sub("", str(title))
    name = re.sub(r"\s+", " ", name).strip().strip(".")
    name = name[:max_len].strip()
    while name and len(name.encode("utf-8")) > MAX_TITLE_BYTES:
        name = name[:-1].rstrip()
    return name


def video_id_in_file(path):
    """The video id embedded in a transcript's or summary note's header,
    or None when the file carries none (only the first 4 KiB are read)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.read(4096)
    except OSError:
        return None
    m = VIDEO_ID_HEAD_RE.search(head)
    return m.group(1) if m else None


def titled_path(directory, base, video_id, ext):
    """Path for a title-named output file, so two different videos never
    silently overwrite each other:

    - <base><ext> when that name is free, or when the existing file is this
      same video's own output (re-running a video updates its file in place);
    - <base> (<video_id>)<ext> when another video already owns the plain
      name: videos sharing a sanitized title each keep their own file. The
      suffixed name belongs to this video by construction (ids are unique),
      so re-runs keep updating it. In the contrived case where even that
      name is owned by yet another video (a title that literally ends in
      " (<id>)"), a " (2)", " (3)" ... counter is appended until a name
      is free or this video's own.

    Callers hold output_dir_lock() around resolution and write, so the
    exists-checks cannot race another writer.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{base}{ext}"
    if video_id:
        attempt = 0
        while path.exists() and video_id_in_file(path) != video_id:
            attempt += 1
            stem = f"{base} ({video_id})" if attempt == 1 \
                else f"{base} ({video_id}) ({attempt})"
            path = directory / f"{stem}{ext}"
            if attempt > 99:  # pathological: stop looking, take the name
                break
    return path


@contextlib.contextmanager
def output_dir_lock(directory):
    """Serialize output-name resolution + write against every other
    writer of the same directory, in this process (naming_lock) and in any
    other yt-pipeline process (an OS-level flock on the directory's
    .yt-pipeline.lock file, released automatically even if a process
    crashes). Without it, two processes transcribing different videos with
    the same sanitized title could interleave titled_path()'s exists-check
    with each other's write and silently drop a transcript; the pre-2.3
    id-based names could never collide, so that race is new with titles.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # "a" never truncates; the file is only a lock token, its content stays
    # empty and it is deliberately never deleted (unlink races re-create it)
    with open(directory / ".yt-pipeline.lock", "a") as lock_file:
        unlocked = None
        try:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                unlocked = fcntl.LOCK_UN
            with naming_lock:
                yield
        finally:
            if unlocked is not None:
                try:
                    fcntl.flock(lock_file.fileno(), unlocked)
                except OSError:  # pragma: no cover - fd already gone
                    pass


def write_text_atomic(path, content):
    """Write content to path via a temp file + atomic rename, so a reader
    in another process (summarize --all, a shell) never sees a half-written
    transcript or note; on failure the temp file is cleaned up and the
    previous content of `path`, if any, survives."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)  # atomic on the same directory/filesystem
    finally:
        tmp.unlink(missing_ok=True)  # no-op after a successful rename


def _transcript_entries(directory):
    """Yield (video_id, rank, path) for every transcript in a directory:
    title-named files are matched by the video_id in their header, legacy
    <id>.txt files by their name. rank orders a video's files: title-named
    beats legacy, then newest first."""
    if not os.path.isdir(directory):
        return
    for fn in sorted(os.listdir(directory)):
        path = Path(directory) / fn
        if not fn.endswith(".txt") or not path.is_file():
            continue
        vid = video_id_in_file(path)
        if not vid:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        yield vid, (path.stem != vid, mtime), path


def transcripts_by_id(directory):
    """{video_id: transcript path} for a transcripts directory, keeping the
    best file per video (title-named over the legacy <id>.txt, then the
    newest)."""
    best = {}
    for vid, rank, path in _transcript_entries(directory):
        if vid not in best or rank > best[vid][0]:
            best[vid] = (rank, path)
    return {vid: path for vid, (_, path) in best.items()}


def owned_transcripts(directory, video_id):
    """Every transcript file of one video in a directory: title-named files
    (matched via their header) and the legacy <id>.txt alike."""
    owned = [path for vid, _, path in _transcript_entries(directory)
             if vid == video_id]
    legacy = Path(directory) / f"{video_id}.txt"
    if legacy.is_file() and legacy not in owned:
        owned.append(legacy)
    return owned


def find_transcript(video_id, directories):
    """The transcript file for a video id, searching the directories in
    order. Understands both the current title-based names (matched via the
    header) and the pre-2.3 id-based <id>.txt files (matched by name, even
    when they carry no header). Prefers the title-named file when a video
    has both."""
    for d in directories:
        hit = transcripts_by_id(d).get(video_id)
        if hit is None:
            legacy = Path(d) / f"{video_id}.txt"
            if legacy.is_file():
                hit = legacy
        if hit is not None:
            return hit
    return None


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
