"""Command-line interface: subcommand parsing and dispatch.

    vidsummarizer <url>[,<url>...]       full pipeline (same as 'run')
    vidsummarizer run <urls/ids>...      mp3 -> transcript -> summary
    vidsummarizer fetch <urls/ids>...    mp3 only
    vidsummarizer transcribe <urls/ids>  transcript only (the old yt-transcribe)
    vidsummarizer summarize <id|url|file|'--all'>   (the old yt-summarize)
    vidsummarizer benchmark [variants]   score prompt variants, compare, report
    vidsummarizer prompts <list|show|new|promote>  manage prompt variants
    vidsummarizer doctor                 check this machine for missing tools

Every video-taking command accepts several videos at once: as several
arguments, comma-separated, from a --batch FILE (one per line), or any mix.
With more than one video the work is overlapped by a small worker pool
(-j/--jobs, default 2)."""

import argparse
import os
import sys
from pathlib import Path

from . import VERSION
from .common import PROG, die, resolve_workdir

COMMANDS = ("run", "fetch", "transcribe", "summarize", "benchmark",
            "prompts", "doctor")

EPILOG = """examples:
  vidsummarizer https://youtu.be/iCvmsMzlF7o           full pipeline in one command
  vidsummarizer run iCvmsMzlF7o --timestamps          same, explicit 'run'
  vidsummarizer url1,url2,jNQXAC9IVRw                 several videos, comma-separated
  vidsummarizer run --batch videos.txt -j 3           ...or one per line from a file
  vidsummarizer fetch jNQXAC9IVRw                      only download the mp3
  vidsummarizer transcribe jNQXAC9IVRw --lang de       only the transcript
  vidsummarizer summarize iCvmsMzlF7o                  only the summary
  vidsummarizer summarize --all --out ~/vault/notes    summarize every transcript
  vidsummarizer summarize iCvmsMzlF7o --style short   same video, other style
  vidsummarizer benchmark                               score every prompt variant
  vidsummarizer prompts new tighter                     add a prompt variant
  vidsummarizer prompts promote tighter                 make a variant the default
  vidsummarizer doctor                                 check required tools

data (audio/, transcripts/, summaries/, models/) lives next to the
executable by default; change it with --workdir DIR or $YT_WORKDIR."""


def add_workdir(p):
    p.add_argument("--workdir", metavar="DIR", default=None,
                   help="directory for audio/, transcripts/, summaries/ and "
                        "models/ (default: $YT_WORKDIR, else next to the "
                        "executable)")


def _jobs_type(s):
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer")
    if v < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return v


def env_jobs():
    raw = (os.environ.get("YT_JOBS") or "").strip()
    if not raw:
        return 2
    try:
        return _jobs_type(raw)
    except argparse.ArgumentTypeError:
        die(f"invalid $YT_JOBS value {raw!r} (expected an integer >= 1)")


def add_batch_opts(p):
    """List input + parallelism (shared by 'run', 'fetch', 'transcribe')."""
    p.add_argument("-b", "--batch", metavar="FILE", action="append",
                   default=None,
                   help="file with one video URL/id per line (blank lines and "
                        "# comments ignored; '-' reads stdin; may be given "
                        "more than once)")
    p.add_argument("-j", "--jobs", type=_jobs_type, default=env_jobs(),
                   metavar="N",
                   help="how many videos to process at once, overlapping "
                        "download/transcription of later videos with the "
                        "summarization of earlier ones (default: 2, env "
                        "YT_JOBS; 1 = strictly one after another)")


def add_transcribe_opts(p, out_flag=("-o", "--out"), out_help=None):
    """Options of the transcription stage (shared by 'transcribe' and 'run')."""
    p.add_argument("videos", nargs="*", metavar="video",
                   help="YouTube video URL(s) or 11-character id(s); commas "
                        "separate several, or use --batch FILE")
    p.add_argument(*out_flag, type=str, default=None, metavar="FILE",
                   help=out_help or "output transcript file "
                        "(default: <workdir>/transcripts/<Title>.txt, named "
                        "after the sanitized video title; one video only)")
    p.add_argument("--audio-dir", type=str, default=None,
                   help="directory for downloaded mp3 files "
                        "(default: <workdir>/audio)")
    p.add_argument("--keep-audio", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="keep the downloaded mp3 (default: --keep-audio)")
    p.add_argument("-m", "--model", type=str, default=None,
                   help=f"path to whisper GGML model (default: "
                        f"<workdir>/models/ggml-small.bin; auto-downloaded "
                        f"if missing)")
    p.add_argument("--model-url", default=None,
                   help="override URL used to auto-download a missing model")
    p.add_argument("-l", "--lang", default="auto",
                   help="spoken language, e.g. en, de (default: auto-detect)")
    p.add_argument("-t", "--threads", type=int, default=None,
                   help="whisper CPU threads per job (default: 8; when several "
                        "jobs can run at once and -t is not given, split "
                        "across them, min 2 each, so a batch stays within one "
                        "video's CPU budget)")
    p.add_argument("--timestamps", action="store_true",
                   help="prefix each transcript paragraph with a [MM:SS] marker")
    p.add_argument("--redownload", action="store_true",
                   help="re-download audio even if the mp3 already exists")
    p.add_argument("--ytdlp", default="",
                   help="extra yt-dlp arguments, e.g. "
                        "--ytdlp '--cookies-from-browser firefox'")
    add_batch_opts(p)


def add_fetch_opts(p):
    p.add_argument("videos", nargs="*", metavar="video",
                   help="YouTube video URL(s) or 11-character id(s); commas "
                        "separate several, or use --batch FILE")
    p.add_argument("--audio-dir", type=str, default=None,
                   help="directory for downloaded mp3 files "
                        "(default: <workdir>/audio)")
    p.add_argument("--redownload", action="store_true",
                   help="re-download audio even if the mp3 already exists")
    p.add_argument("--ytdlp", default="",
                   help="extra yt-dlp arguments, e.g. "
                        "--ytdlp '--cookies-from-browser firefox'")
    add_batch_opts(p)


def add_prompts_dir_opt(p):
    p.add_argument("--prompts-dir", metavar="DIR", default=None,
                   help="directory with prompt variant *.json files "
                        "(default: $YT_PROMPTS_DIR, else <workdir>/prompts "
                        "if it has any, else the bundled copy)")


def add_summarize_opts(p, run_mode=False):
    """Options of the summarization stage (shared by 'summarize' and 'run').

    In run_mode the summary-language option is --summary-lang (because --lang
    is the audio language there) and the model option is --llm-model (because
    -m/--model is the whisper model there).
    """
    if run_mode:
        p.add_argument("--llm-model", metavar="NAME",
                       default=os.environ.get("OLLAMA_MODEL", "qwen3.6"),
                       help="Ollama model tag for summaries "
                            "(default: qwen3.6; env OLLAMA_MODEL)")
        p.add_argument("--summary-lang", default=None,
                       help="force summary language, e.g. en, de (default: "
                            "the transcript's own language)")
    else:
        p.add_argument("--model", metavar="NAME",
                       default=os.environ.get("OLLAMA_MODEL", "qwen3.6"),
                       help="Ollama model tag (default: qwen3.6; env OLLAMA_MODEL)")
        p.add_argument("--lang", default=None,
                       help="force summary language, e.g. en, de (default: "
                            "the transcript's own language)")
    p.add_argument("--host", default=os.environ.get("OLLAMA_HOST",
                                                    "http://127.0.0.1:11434"),
                   help="Ollama base URL (default: http://127.0.0.1:11434; "
                        "env OLLAMA_HOST)")
    p.add_argument("--style", metavar="NAME", default=None,
                   help="summary style: the shape the note takes, chosen "
                        "per run (built-in: actionable, short, study-guide; "
                        "any other prompt variant name works too; default: "
                        "the standard actionable-or-overview summary). A "
                        "style may carry its own word cap; see README "
                        "'Summary styles'")
    p.add_argument("--max-words", type=int, default=None,
                   help="hard word cap for the summary (default: 500, or "
                        "the selected style's own cap)")
    p.add_argument("--chunk-words", type=int, default=1500,
                   help="words per map-reduce chunk for long transcripts "
                        "(default: 1500; 0 = never chunk)")
    p.add_argument("--notes-words", type=int, default=220,
                   help="word cap for per-chunk notes (default: 220)")
    p.add_argument("--think", action="store_true",
                   help="let the model use its thinking mode (slower)")
    p.add_argument("--timeout", type=float, default=900.0,
                   help="per-request timeout in seconds (default: 900)")
    p.add_argument("--max-num-ctx", type=int, default=32768,
                   help="ceiling for the Ollama context window (default: 32768)")
    p.add_argument("--force", action="store_true",
                   help="re-summarize even if a summary already exists")
    p.add_argument("--stdout", action="store_true",
                   help="print the markdown note instead of writing a file")
    p.add_argument("--no-autostart", action="store_true",
                   help="never start 'ollama serve' ourselves")
    add_prompts_dir_opt(p)


def build_parser():
    ap = argparse.ArgumentParser(
        prog=PROG,
        description="Local YouTube pipeline: download audio (yt-dlp), "
                    "transcribe (whisper.cpp), summarize (Ollama). "
                    "One executable, no cloud APIs.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version",
                    version=f"{PROG} {VERSION}")
    sub = ap.add_subparsers(dest="command", metavar="<command>")

    p_run = sub.add_parser(
        "run", help="full pipeline: URL -> mp3 -> transcript -> summary",
        description="End to end: download the audio, transcribe it with "
                    "whisper.cpp, summarize the transcript with the local "
                    "LLM and write an Obsidian-ready note. Accepts several "
                    "videos (comma-separated, several arguments and/or "
                    "--batch FILE), processed with a small worker pool.",
        epilog="examples:\n"
               "  vidsummarizer run url1,url2,url3 -j 2\n"
               "  vidsummarizer run --batch videos.txt --timestamps\n\n"
               "Calling the executable with just a URL or video id "
               "(no subcommand) does the same thing.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_transcribe_opts(
        p_run, out_flag=("-o", "--transcript-out"),
        out_help="output transcript file (default: "
                 "<workdir>/transcripts/<Title>.txt)")
    add_summarize_opts(p_run, run_mode=True)
    p_run.add_argument("--out", metavar="DIR", default=None,
                       help="output directory for notes "
                            "(default: <workdir>/summaries)")
    add_workdir(p_run)
    p_fetch = sub.add_parser(
        "fetch", help="download the audio (mp3) only",
        description="Download the video's audio track as mp3; no "
                    "transcription, no summary. Accepts several videos.")
    add_fetch_opts(p_fetch)
    add_workdir(p_fetch)

    p_tr = sub.add_parser(
        "transcribe", help="URL -> transcript only (old yt-transcribe)",
        description="Download the audio (or reuse the cached mp3), transcribe "
                    "it locally with whisper.cpp and write a plain-text "
                    "transcript. Accepts several videos.")
    add_transcribe_opts(p_tr)
    add_workdir(p_tr)

    p_sum = sub.add_parser(
        "summarize", help="transcript -> markdown summary only "
                          "(old yt-summarize)",
        description="Summarize transcript(s) with the local LLM (Ollama) "
                    "into Obsidian-ready markdown notes.",
        epilog="examples:\n"
               "  vidsummarizer summarize iCvmsMzlF7o\n"
               "  vidsummarizer summarize --all\n"
               "  vidsummarizer summarize iCvmsMzlF7o --style study-guide\n"
               "  vidsummarizer summarize ~/notes/transcript.txt "
               "--out ~/vault/Video\\ notes\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_sum.add_argument("targets", nargs="*",
                       help="transcript .txt file, 11-char video id, or "
                            "YouTube URL")
    p_sum.add_argument("--all", action="store_true",
                       help="summarize every transcript in the transcripts dir "
                            "that has no summary yet")
    p_sum.add_argument("--transcripts", metavar="DIR", default=None,
                       help="transcripts directory (default: "
                            "$YT_TRANSCRIPTS_DIR, else <workdir>/transcripts)")
    p_sum.add_argument("--out", metavar="DIR", default=None,
                       help="output directory for notes (default: "
                            "<workdir>/summaries)")
    add_summarize_opts(p_sum)
    add_workdir(p_sum)

    p_bm = sub.add_parser(
        "benchmark", help="score prompt variants on the benchmark set",
        description="Run the summarizer over the fixed benchmark set of "
                    "transcripts with one or more prompt variants, score "
                    "every summary (automatic checks + a local LLM judge "
                    "rating faithfulness, actionability and clarity) and "
                    "write a comparison report. The winning variant is "
                    "only *suggested*; make it the default with "
                    "'vidsummarizer prompts promote'.",
        epilog="examples:\n"
               "  vidsummarizer benchmark\n"
               "  vidsummarizer benchmark --prompts v1-baseline,tighter\n"
               "  vidsummarizer benchmark --judge-model llama3.2\n"
               "  vidsummarizer benchmark --min-overall 3.5   # quality gate\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_bm.add_argument("--prompts", metavar="A,B", default=None,
                      help="prompt variants to compare, comma-separated "
                           "(default: every variant in the prompts dir)")
    p_bm.add_argument("--set", metavar="FILE", default=None,
                      help="benchmark set.json (or its directory; default: "
                           "<workdir>/benchmarks/set.json, else the "
                           "bundled copy)")
    p_bm.add_argument("--model", metavar="NAME",
                      default=os.environ.get("OLLAMA_MODEL", "qwen3.6"),
                      help="Ollama model that writes the summaries "
                           "(default: qwen3.6; env OLLAMA_MODEL)")
    p_bm.add_argument("--judge-model", metavar="NAME",
                      default=os.environ.get("OLLAMA_JUDGE_MODEL", None),
                      help="Ollama model that rates the summaries against "
                           "the rubric (default: same as --model; env "
                           "OLLAMA_JUDGE_MODEL)")
    p_bm.add_argument("--out", metavar="DIR", default=None,
                      help="report directory (default: "
                           "<workdir>/benchmarks/results/<timestamp>)")
    p_bm.add_argument("--min-overall", type=float, default=None,
                      metavar="SCORE",
                      help="quality gate: exit non-zero when the ACTIVE "
                           "variant scores below SCORE or fails an "
                           "automatic check (used by 'make bench-real')")
    p_bm.add_argument("--host", default=os.environ.get("OLLAMA_HOST",
                                                        "http://127.0.0.1:11434"),
                      help="Ollama base URL (default: "
                           "http://127.0.0.1:11434; env OLLAMA_HOST)")
    p_bm.add_argument("--max-words", type=int, default=None,
                      help="hard word cap for the summaries (default: 500, "
                           "or the benchmarked variant's own style cap)")
    p_bm.add_argument("--chunk-words", type=int, default=1500,
                      help="words per map-reduce chunk (default: 1500; "
                           "0 = never chunk)")
    p_bm.add_argument("--notes-words", type=int, default=220,
                      help="word cap for per-chunk notes (default: 220)")
    p_bm.add_argument("--think", action="store_true",
                      help="let the models use thinking mode (slower)")
    p_bm.add_argument("--timeout", type=float, default=900.0,
                      help="per-request timeout in seconds (default: 900)")
    p_bm.add_argument("--max-num-ctx", type=int, default=32768,
                      help="ceiling for the Ollama context window "
                           "(default: 32768)")
    p_bm.add_argument("--no-autostart", action="store_true",
                      help="never start 'ollama serve' ourselves")
    add_prompts_dir_opt(p_bm)
    add_workdir(p_bm)

    p_pr = sub.add_parser(
        "prompts", help="manage the summarizer's prompt variants",
        description="The summarizer's prompts live as versioned JSON "
                    "files (one per variant) in the prompts directory. "
                    "This command lists, shows, creates and promotes "
                    "them; 'benchmark' compares them.",
        epilog="examples:\n"
               "  vidsummarizer prompts list\n"
               "  vidsummarizer prompts show v1-baseline\n"
               "  vidsummarizer prompts new tighter --from v1-baseline\n"
               "  vidsummarizer prompts promote tighter\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_pr.add_argument("action", nargs="?", default="list",
                      choices=("list", "show", "new", "promote"),
                      help="list (default), show, new or promote")
    p_pr.add_argument("name", nargs="?", default=None,
                      help="variant name (show/new/promote)")
    p_pr.add_argument("--from", dest="from_variant", default=None,
                      metavar="VARIANT",
                      help="variant to copy for 'new' (default: the active "
                           "one)")
    p_pr.add_argument("--yes", action="store_true",
                      help="promote without asking for confirmation")
    add_prompts_dir_opt(p_pr)
    add_workdir(p_pr)

    p_doc = sub.add_parser(
        "doctor", help="check this machine for everything the pipeline needs",
        description="Check Python, the external tools (yt-dlp, ffmpeg, "
                    "whisper-cli, ollama), the Ollama server and model, and "
                    "the data directory. Reports clearly what is missing and "
                    "how to install it.")
    p_doc.add_argument("--host", default=os.environ.get(
        "OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama base URL (default: http://127.0.0.1:11434)")
    p_doc.add_argument("--llm-model", metavar="NAME",
                       default=os.environ.get("OLLAMA_MODEL", "qwen3.6"),
                       help="Ollama model tag to check (default: qwen3.6)")
    add_workdir(p_doc)

    return ap


# --------------------------------------------------------------------------- #
# Command implementations that live in their own modules
# --------------------------------------------------------------------------- #

def cmd_summarize(args):
    from .summarize import (ensure_video_id, output_path, parse_transcript,
                            resolve_transcripts, summarized_video_ids,
                            summarize_targets)
    from .common import die, log, transcripts_by_id
    workdir = resolve_workdir(args.workdir)
    args.workdir = str(workdir)

    if args.transcripts:
        dirs = [args.transcripts]
    elif os.environ.get("YT_TRANSCRIPTS_DIR"):
        dirs = [os.environ["YT_TRANSCRIPTS_DIR"]]
    else:
        dirs = [str(workdir / "transcripts")]
    transcripts_dirs = []
    for d in (os.path.abspath(x) for x in dirs):
        if d not in transcripts_dirs:
            transcripts_dirs.append(d)
    args.transcripts_dirs = transcripts_dirs
    args.out = args.out or str(workdir / "summaries")

    if args.all:
        import os as _os
        src = transcripts_dirs[0]
        if not _os.path.isdir(src):
            die(f"transcripts directory not found: {src}")
        files = sorted(Path(src) / f for f in _os.listdir(src)
                       if f.endswith(".txt"))
        if not files:
            die(f"no .txt transcripts in {src}")
        done = summarized_video_ids(args.out)
        # a video may have several transcript files on disk (a legacy
        # <id>.txt next to its newer title-named file): keep one per video
        keep = transcripts_by_id(src)
        targets, seen, skipped_done, skipped_dupes = [], set(), 0, 0
        for f in files:
            meta, _ = parse_transcript(open(f, encoding="utf-8",
                                            errors="replace").read())
            # a headerless pre-2.3 <id>.txt is identified by its filename,
            # so it deduplicates and counts as done like any other file
            vid = ensure_video_id(meta, f)
            if vid:
                if vid in done and not args.force:
                    skipped_done += 1
                    continue
                if vid in seen or keep.get(vid) not in (None, f):
                    skipped_dupes += 1  # another file of this video won
                    continue
                seen.add(vid)
            elif os.path.exists(
                    output_path(args.out, meta,
                                os.path.splitext(f.name)[0])) \
                    and not args.force:
                skipped_done += 1
                continue
            targets.append(f)
        # report the two skip reasons separately: a deduped extra copy of
        # an already-listed video is not "already summarized"
        bits = []
        if skipped_done:
            bits.append(f"{skipped_done} already summarized")
        if skipped_dupes:
            word = "copy" if skipped_dupes == 1 else "copies"
            bits.append(f"{skipped_dupes} duplicate {word}")
        log(f"found {len(files)} transcripts, "
            f"{', '.join(bits) if bits else '0 already summarized'}, "
            f"{len(targets)} to do"
            + (" (--force to redo)" if skipped_done else ""))
        if not targets:
            log("nothing to do.")
            return 0
    else:
        if not args.targets:
            build_parser().error(
                "summarize: give a transcript file / video id / URL, or use --all")
        targets = resolve_transcripts(args, args.targets)

    return summarize_targets(targets, args)


def cmd_benchmark(args):
    from .bench import run_benchmark
    if not args.judge_model:
        args.judge_model = args.model
    args.workdir = str(resolve_workdir(args.workdir))
    return run_benchmark(args)


def cmd_prompts(args):
    from . import prompts as P
    workdir = str(resolve_workdir(args.workdir))
    read_dir = P.resolve_prompts_dir(args.prompts_dir, workdir)

    if args.action == "list":
        active = P.active_name(read_dir)
        variants = P.list_variants(read_dir)
        print(f"prompt variants in {read_dir}  (active: {active})")
        for v in variants:
            mark = "  [active]" if v.name == active else ""
            print(f"  {v.name:<24} v{v.version:<3} {v.description}{mark}")
        print("benchmark them with: vidsummarizer benchmark")
        return 0

    if not args.name:
        build_parser().error(f"prompts {args.action}: a variant name is "
                             "required")

    if args.action == "show":
        v = P.load_variant(read_dir, args.name)
        import json as _json
        print(_json.dumps(v.as_dict(), indent=2, ensure_ascii=False))
        return 0

    write_dir = P.writable_prompts_dir(read_dir, workdir,
                                       explicit=args.prompts_dir)

    if args.action == "new":
        src = args.from_variant or P.active_name(read_dir)
        path = P.new_variant(read_dir, write_dir, args.name, src)
        from .common import log
        log(f"created {path} (copy of {src})")
        print(f"created {path}\nedit it, then: vidsummarizer benchmark "
              f"--prompts {args.name}")
        return 0

    # promote
    v = P.load_variant(read_dir, args.name)
    current = P.active_name(read_dir)
    if v.name == current and read_dir == write_dir:
        print(f"'{v.name}' is already the active prompt variant")
        return 0
    if not args.yes:
        print(f"Promote '{v.name}' (v{v.version}, {v.description}) to be "
              f"the active summarizer prompt?\nCurrent active: {current}. "
              "Future run/summarize calls will use it.")
        answer = input("promote? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            from .common import log
            log("promote cancelled")
            return 1
    path = P.set_active(write_dir, v.name)
    from .common import log
    log(f"active prompt variant is now '{v.name}' ({path})")
    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    # 'vidsummarizer <url>' means 'vidsummarizer run <url>'; the same shorthand
    # works when the batch options lead: 'vidsummarizer --batch f.txt -j 3'.
    if argv and argv[0] not in COMMANDS:
        first = argv[0].split("=", 1)[0]
        if not argv[0].startswith("-") or first in ("-b", "--batch", "-j", "--jobs"):
            argv = ["run"] + list(argv)

    args = build_parser().parse_args(argv)
    if args.command is None:
        build_parser().print_help()
        raise SystemExit(2)

    from .pipeline import cmd_doctor, cmd_fetch, cmd_run, cmd_transcribe
    handlers = {
        "run": cmd_run,
        "fetch": cmd_fetch,
        "transcribe": cmd_transcribe,
        "summarize": cmd_summarize,
        "benchmark": cmd_benchmark,
        "prompts": cmd_prompts,
        "doctor": cmd_doctor,
    }
    rc = handlers[args.command](args)
    raise SystemExit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
