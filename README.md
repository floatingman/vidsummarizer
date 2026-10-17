# yt-pipeline

One portable executable for the whole local YouTube pipeline:

```
YouTube URL -> yt-dlp (mp3) -> whisper.cpp transcript -> local-LLM summary (Obsidian note)
```

It merges the two former tools, `yt-transcribe` 1.0.0 (download + transcribe)
and `yt-summarize` 1.0.0 (local-LLM summary) into a single self-contained
file. No cloud APIs, no API keys; audio download, transcription and
summarization all run on your machine. The whole tool is Python standard
library only, so it ships as one executable file that can be copied to other
computers.

```
yt-pipeline <url(s)>            full pipeline in one command (same as 'run')
yt-pipeline run <url(s)>        ...explicit form, with all stage options
yt-pipeline fetch <url(s)>      only download the mp3
yt-pipeline transcribe <url(s)> only the transcript (the old yt-transcribe)
yt-pipeline summarize <id|url|file> | --all    only the note (the old yt-summarize)
yt-pipeline doctor               check this machine for everything the pipeline needs
```

All current behavior of the two original tools is preserved: LLM-readable
plain-text transcripts, and markdown summaries of at most 500 words
(actionable content preferred, otherwise an overview) that import into an
Obsidian vault as regular notes. See "Output formats" below.

Every video-taking command (`<url(s)>` above) accepts a whole list of
videos in one run: several arguments, comma-separated values, a `--batch`
FILE with one video per line, or any mix; batches are processed with a
small worker pool so transcription of later videos overlaps with the
summarization of earlier ones ("Batch processing" below).

## The one-file executable

`./build.sh` produces `dist/yt-pipeline`, a single self-contained file
(~28 KB) built with the Python standard library's `zipapp` module: the whole
`ytpipeline` package is packed into one executable archive with a
`#!/usr/bin/env python3` shebang. It does not need this checkout, a git
clone, or any pip packages at runtime; it only needs a Python 3.9+
interpreter on the machine where it runs.

```bash
./build.sh                 # builds and smoke-tests dist/yt-pipeline
./dist/yt-pipeline --version
```

(Rebuild after changing any file under `ytpipeline/`. The build stage copies
only the package into the archive; README, .git and data never get packed.)

### Transferring to another computer

```bash
scp dist/yt-pipeline user@host:~/yt-pipeline/
ssh user@host 'cd ~/yt-pipeline && ./yt-pipeline doctor'   # see what's missing
ssh user@host 'cd ~/yt-pipeline && ./yt-pipeline <youtube-url>'
```

The file is plain Python bytecode in a zip container, so it also runs on a
different Linux distribution or CPU architecture as long as that machine has
`python3` (3.9 or newer). To run it from a directory not on `PATH`
invocations, mark it executable once: `chmod +x yt-pipeline`.

Data (see "Data layout" below) accumulates next to the executable by
default, exactly like the old project checkouts. Optionally copy the old
`models/`, `audio/`, `transcripts/` folders alongside it to reuse caches;
the whisper model is auto-downloaded on first use anyway.

## What the target computer needs

Whatever cannot reasonably live inside the executable is detected at
runtime, with a clear message about what is missing and how to install it.
`yt-pipeline doctor` prints the full report; each command also checks its
own needs before starting and exits with an install hint instead of a
confusing stack trace.

| Requirement | Needed for | Install |
|---|---|---|
| Python 3.9+ | running the executable | preinstalled on virtually all Linux distros (`python3 --version`) |
| `yt-dlp` on PATH | every command that touches YouTube (run, fetch, transcribe) | `pipx install yt-dlp` or the standalone binary from [github.com/yt-dlp/yt-dlp](https://github.com/yt-dlp/yt-dlp) |
| `ffmpeg` on PATH | audio download (mp3 encode), wav fallback | system package manager, e.g. `sudo apt install ffmpeg` |
| `whisper-cli` on PATH | transcription (run, transcribe) | build [whisper.cpp](https://github.com/ggml-org/whisper.cpp): `cmake -B build && cmake --build build -j --target whisper-cli`, then put `whisper-cli` on PATH |
| `ollama` on the machine | summaries (run, summarize) | install from [ollama.com](https://ollama.com); the tool auto-starts `ollama serve` if the server is not running |
| an Ollama model | summaries | `ollama pull qwen3.6` (default model, ~23 GB); any other tag via `--model` / `--llm-model` |
| whisper GGML model | transcription | nothing to do: downloaded automatically on first use (~466 MiB for `ggml-small.bin`) |

Not bundled on purpose (too big / machine-specific): the whisper model
files, the Ollama installation and its models, and the external CLIs. The
whisper model downloads itself on first use; everything else is reported by
`doctor` with the install commands above.

Example `doctor` output on a machine missing whisper.cpp and Ollama:

```
$ ./yt-pipeline doctor
yt-pipeline 2.1.0 doctor: checking this machine for the local pipeline

Core:
  [ OK ] Python 3.12.3 (/usr/bin/python3)

Audio + transcription (commands: run, fetch, transcribe):
  [ OK ] yt-dlp 2026.10.10  (/home/user/.local/bin/yt-dlp)
  [ OK ] ffmpeg 6.1.1  (/usr/bin/ffmpeg)
  [MISS] whisper-cli (transcribes the audio locally (whisper.cpp))
         install: build whisper.cpp from https://github.com/ggml-org/whisper.cpp ...
  [AUTO] whisper model ggml-small.bin at /home/user/yt-pipeline/models/ggml-small.bin
         (downloaded automatically on first use, ~466 MiB ...)

Summaries (commands: run, summarize):
  [MISS] ollama (serves the local LLM that writes the summaries)
         install: install from https://ollama.com, then pull a model with: ollama pull qwen3.6

Data:
  [ OK ] workdir /home/user/yt-pipeline (writable)

Result: missing whisper-cli, ollama.
Without them only these commands can run: fetch.
Fix the [MISS] items (install hints above) and re-run 'yt-pipeline doctor'.
```

`doctor` exits 0 when the full pipeline can run and 1 when something is
missing, so it can gate automated setups.

## Usage

### Full pipeline

```bash
./yt-pipeline https://www.youtube.com/watch?v=iCvmsMzlF7o    # one command, end to end
./yt-pipeline run iCvmsMzlF7o --timestamps                   # explicit form
./yt-pipeline run jNQXAC9IVRw --no-keep-audio --stdout       # print the note, delete the mp3
./yt-pipeline url1,url2,iCvmsMzlF7o                          # a whole list in one run
./yt-pipeline run --batch watchlist.txt -j 3                 # ...or from a file, 3 at a time
```

Downloads the audio, writes `transcripts/<video_id>.txt`, then the summary
note into `summaries/<Title>.md`, and prints what it did to stderr. On the
reference machine (Ryzen 7 5850U, CPU-only) a 21-minute video takes about
7 minutes end to end (2 min whisper.cpp + 5 min qwen3.6 summarization).

### Batch processing (many videos at once)

`run`, `fetch` and `transcribe` take any number of videos. Three input
forms, freely combinable:

```bash
./yt-pipeline run https://youtu.be/a,https://youtu.be/b,jNQXAC9IVRw  # comma-separated
./yt-pipeline run url1 url2 url3                                     # several arguments
./yt-pipeline run --batch watchlist.txt                              # one video per line
```

A batch file has one video URL or id per line (anything after a `#` is a
comment, blank lines are ignored, commas also separate); `--batch -` reads
the list from stdin, and `--batch` may be given more than once:

```
# watchlist.txt
https://www.youtube.com/watch?v=iCvmsMzlF7o    # the merge explainer
jNQXAC9IVRw
```

Videos are validated and de-duplicated before anything is downloaded (a
typo in the list costs nothing), and every line of progress output is
prefixed with the video id so concurrent videos stay readable:

```
* batch: processing 2 video(s), 2 at a time, 4 whisper thread(s) each
[jNQXAC9IVRw] * Me at the zoo [00:19] -> downloading audio
[dQw4w9WgXcQ] * Rick Astley - Never Gonna Give You Up ... [03:33] -> downloading audio
[jNQXAC9IVRw] * transcribing jNQXAC9IVRw.mp3 with ggml-small.bin (lang=auto)
[jNQXAC9IVRw] 35 words -> single pass (qwen3.6)
[dQw4w9WgXcQ] * transcribing dQw4w9WgXcQ.mp3 with ggml-small.bin (lang=auto)
[jNQXAC9IVRw] wrote .../summaries/Me at the zoo.md (44 words, 30s)
[dQw4w9WgXcQ] * wrote .../transcripts/dQw4w9WgXcQ.txt (494 words, 3 paragraphs)
...
* batch finished in 01:45: 2/2 video(s) ok
```

A video that fails (unavailable, geo-blocked, empty transcript, ...) does
not stop the batch: it is reported, the rest carry on, and the exit code is
1 if anything failed. `--stdout` with several videos forces `--jobs 1` so
the notes print in order.

**Parallelism.** `-j/--jobs N` (default 2, env `YT_JOBS`) sets how many
videos are in flight at once. Summarization (local LLM) is slower than
transcription (whisper.cpp), so even 2 jobs already overlap the stages:
while an earlier video is still being summarized, the next one is already
downloading/transcribing, as in the log above. Higher values overlap more,
bounded by the machine:

- each job runs one `whisper-cli` (CPU) and one Ollama request at a time;
  the Ollama server queues concurrent generation requests itself, so the
  LLM stays a single hot process no matter how many jobs you set;
- unless `-t` is given explicitly, whisper's default 8 CPU threads are
  split across the jobs that can actually run at once
  (`max(2, 8/min(--jobs, videos))` each), so a batch's total CPU use stays
  near that of a single-video run; pass `-t` to override;
- `--jobs 1` reproduces the exact one-at-a-time behavior (useful on small
  machines or when comparing runs).

### Individual steps

```bash
./yt-pipeline fetch jNQXAC9IVRw                              # mp3 only
./yt-pipeline fetch --batch list.txt -j 4                    # ...several at once
./yt-pipeline transcribe jNQXAC9IVRw                         # mp3 -> transcript
./yt-pipeline transcribe a,b,c --timestamps                  # several, with [MM:SS] markers
./yt-pipeline transcribe DP-VTlJkzOY --lang de               # force audio language
./yt-pipeline transcribe -m models/ggml-base.bin <id>        # different (auto-downloaded) model
./yt-pipeline transcribe --ytdlp '--cookies-from-browser firefox' <id>
./yt-pipeline summarize iCvmsMzlF7o                          # transcript -> note (by id)
./yt-pipeline summarize https://youtu.be/iCvmsMzlF7o         # by URL
./yt-pipeline summarize ~/any/transcript.txt                 # by file
./yt-pipeline summarize --all                                # every transcript without a note
./yt-pipeline summarize --all --force                        # redo everything
./yt-pipeline summarize --all --out ~/vault/Video\ notes     # write straight into a vault
./yt-pipeline summarize jNQXAC9IVRw --lang en                # force note language
```

Progress goes to stderr; transcripts and notes are files; `--stdout` on
run/summarize prints the note to stdout instead.

### Options

Batch input and parallelism (`run`, `fetch`, `transcribe`):

| Option | Default | Description |
|---|---|---|
| `-b, --batch FILE` | none | file with one video URL/id per line (`#` comments and blank lines ignored); `-` reads stdin; repeatable, combinable with positional URLs/ids |
| `-j, --jobs N` | `2` (env `YT_JOBS`) | videos processed concurrently; `1` = strictly sequential |

Transcription stage (`run`, `transcribe`):

| Option | Default | Description |
|---|---|---|
| `-o, --out FILE` / `-o, --transcript-out FILE` | `<workdir>/transcripts/<id>.txt` | output transcript path (single video only) |
| `-m, --model PATH` | `<workdir>/models/ggml-small.bin` | whisper GGML model; auto-downloaded if missing |
| `--model-url URL` | HuggingFace `ggerganov/whisper.cpp` | override download URL for the model |
| `-l, --lang LANG` | `auto` | spoken language (`en`, `de`, ...); `auto` lets whisper detect |
| `-t, --threads N` | `8` | whisper CPU threads per job; split across jobs actually running (`max(2, 8/min(--jobs, videos))` each) when batching without an explicit `-t` |
| `--timestamps` | off | prefix each paragraph with a `[MM:SS]` start marker |
| `--audio-dir DIR` | `<workdir>/audio` | where mp3 files are stored |
| `--keep-audio` / `--no-keep-audio` | keep | keep or delete the mp3 afterwards |
| `--redownload` | off | re-download audio even if the mp3 exists |
| `--ytdlp 'ARGS'` | none | extra arguments for yt-dlp |

Summarization stage (`run`, `summarize`):

| Option | Default | Description |
|---|---|---|
| `--model NAME` (summarize) / `--llm-model NAME` (run) | `qwen3.6` (env `OLLAMA_MODEL`) | Ollama model tag |
| `--out DIR` | `<workdir>/summaries` | output directory for notes |
| `--host URL` | `http://127.0.0.1:11434` (env `OLLAMA_HOST`) | Ollama base URL |
| `--max-words N` | 500 | hard word cap for the summary |
| `--lang CODE` (summarize) / `--summary-lang CODE` (run) | transcript's language | force the summary language |
| `--chunk-words N` | 1500 | words per map-reduce chunk for long transcripts (`0` = never chunk) |
| `--notes-words N` | 220 | word cap for per-chunk notes |
| `--think` | off | let the model use thinking mode (5-10x slower) |
| `--force` | off | re-summarize even if a note exists (`--all` mode) |
| `--stdout` | off | print the note instead of writing a file |
| `--no-autostart` | off | never start `ollama serve` ourselves |
| `--timeout S` | 900 | per-request timeout |
| `--max-num-ctx N` | 32768 | ceiling for the Ollama context window |

`summarize` only: `--all`, `--transcripts DIR` (default `$YT_TRANSCRIPTS_DIR`
else `<workdir>/transcripts`). All commands: `--workdir DIR` (default
`$YT_WORKDIR`, else the executable's own directory).

Accepted URL forms: `youtube.com/watch?v=ID`, `youtu.be/ID`,
`youtube.com/shorts/ID`, `/live/ID`, `/embed/ID`, or a bare 11-character
video id.

## Data layout

Everything lives in one work directory, resolved in this order:

1. `--workdir DIR`
2. `$YT_WORKDIR`
3. the directory containing the executable (like the old project checkouts)
4. if that is read-only (e.g. the binary was installed to `/usr/local/bin`):
   `~/.local/share/yt-pipeline` (with a notice on stderr)

```
<workdir>/
  audio/          downloaded mp3s (cache; delete freely)
  transcripts/    plain-text transcripts
  summaries/      Obsidian-ready markdown notes
  models/         whisper GGML models (auto-downloaded on first use)
```

Old data can be adopted by copying the old projects' `models/`,
`audio/`, `transcripts/` and `summaries/` folders into the workdir; file
formats are unchanged, and the summarizer still accepts transcripts written
by the old `yt-transcribe` (and vice versa).

## Output formats (unchanged from the original tools)

**Transcript** (`transcripts/<video_id>.txt`): UTF-8 plain text, a short
`key: value` header (`title`, `channel`, `upload_date`, `duration`,
`video_id`, `url`, `language`, `model`, `tool`), one blank line, then the
transcript as paragraphs split at silence gaps of >= 1.5 s (soft cap ~280
words at sentence ends, hard cap 500 words). Designed for LLM consumption.

**Summary note** (`summaries/<Title>.md`): Obsidian-ready markdown with YAML
frontmatter properties (`title`, `channel`, `date`, `url`, `video_id`,
`duration`, `language`, `transcribed_with`, `summarized_with`,
`summary_words`, `summarizer`, `created`, `tags`), a `# title` heading, a
source line with the YouTube link, then the summary: `## Actionable
summary` with numbered `### Actionable steps` when the video has actionable
content, otherwise a `## Overview` of 2-4 paragraphs. The summary body is
capped at 500 words (`--max-words`); if the model runs over it is asked
once for a tighter rewrite, then trimmed at a sentence boundary. Long
transcripts are summarized map-reduce style (per-chunk notes, then a final
pass). The note's filename strips characters Obsidian forbids; on a name
clash between different videos the video id is appended.

## Differences from the original two tools

Only what the merge itself forces; formats and pipeline behavior are
unchanged:

- The `tool:` transcript line and `summarizer:` frontmatter now say
  `yt-pipeline 2.1.0` instead of the old tool names.
- One work directory (see above) instead of two sibling checkouts;
  `yt-summarize`'s machine-specific default transcript path is gone
  (`--transcripts` and `$YT_TRANSCRIPTS_DIR` still work).
- On the `run` subcommand, two flags are renamed to avoid clashes:
  `--model` (whisper model) vs `--llm-model` (Ollama model), and
  `--lang` (audio language) vs `--summary-lang` (note language).
  The `transcribe` and `summarize` subcommands keep the original flags.
- Missing external tools now abort with per-tool install hints (and
  `doctor` for the full picture) instead of a bare "missing required
  tool(s)" line.

## Development

```
yt-pipeline/
  ytpipeline/            the tool (Python stdlib only)
    cli.py               argument parsing and subcommand dispatch
    common.py            shared helpers, tool detection, workdir resolution,
                         video-list parsing, thread-safe tagged logging
    transcribe.py        yt-dlp + whisper.cpp stage (from yt-transcribe)
    summarize.py         Ollama stage (from yt-summarize)
    batch.py             worker pool: list of videos, overlapped, failures isolated
    pipeline.py          run/fetch/transcribe/doctor commands
  build.sh               builds dist/yt-pipeline (zipapp)
  README.md
```

Run from the checkout without building: `python3 -m ytpipeline <command> ...`
(same behavior; the workdir then defaults to the repo root).

## Troubleshooting

- **"[MISS] ..." at startup or in doctor**: install the listed tool on the
  target machine; hints are printed next to every missing item.
- **HTTP 403 / throttling from YouTube**: transient; downloads retry 3x.
  Persistent failures usually mean yt-dlp needs an update (`yt-dlp -U`).
- **"model 'qwen3.6' is not installed"**: `ollama pull qwen3.6`, or pick
  another with `--model`; `ollama list` shows what is installed.
- **Ollama not running**: `run`/`summarize` start `ollama serve`
  automatically (log: `/tmp/ollama-serve-yt-pipeline.log`); with
  `--no-autostart`, start it yourself (`systemctl start ollama`).
- **First transcription on a new machine is slow to start**: it downloads
  the ~466 MiB whisper model once into `<workdir>/models/`.
- **Disliked summary**: re-run with `--force`; different models give
  different flavors (`--model`).
- **One bad video in a batch**: expected to happen; the run continues with
  the other videos and lists the failed ids at the end (exit code 1).
- **Batch too slow / machine overloaded**: lower `--jobs` (default 2) or
  pin whisper threads with `-t`; see "Batch processing" for the CPU budget
  model. `OLLAMA_NUM_PARALLEL` on the Ollama server side controls whether
  summaries themselves generate in parallel.
