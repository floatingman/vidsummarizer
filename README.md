# vidsummarizer

**Fully local YouTube transcription and summarization. No cloud, no API keys.**

[![tests](https://github.com/floatingman/vidsummarizer/actions/workflows/ci.yml/badge.svg)](https://github.com/floatingman/vidsummarizer/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/floatingman/vidsummarizer)](https://github.com/floatingman/vidsummarizer/releases/latest)
[![license](https://img.shields.io/github/license/floatingman/vidsummarizer)](LICENSE)

One portable executable for the whole pipeline:

```
YouTube URL -> yt-dlp (mp3) -> whisper.cpp transcript -> local-LLM summary (Obsidian note)
```

<p align="center">
  <a href="docs/demo.cast" title="asciinema recording of the run shown below"><img src="docs/demo.gif" alt="A real run of vidsummarizer: one command downloads a YouTube video's audio, transcribes it locally with whisper.cpp, summarizes it with a local Ollama model, and prints the resulting Obsidian-ready markdown note."></a>
</p>

<p align="center"><em>A real run, nothing staged: one command, three local stages, zero cloud.</em></p>

Audio download, transcription and summarization all run on your machine.
The whole tool is Python standard library only, so it ships as one small
executable file that can be copied to any computer with Python 3.9+.

## Quickstart

```bash
curl -LO https://github.com/floatingman/vidsummarizer/releases/latest/download/vidsummarizer
chmod +x vidsummarizer
./vidsummarizer https://www.youtube.com/watch?v=iCvmsMzlF7o
```

That is the whole pipeline: mp3, transcript, summary note. On a fresh
machine run `./vidsummarizer doctor` first: it checks for `yt-dlp`,
`ffmpeg`, `whisper-cli` and `ollama`, and prints exactly how to install
whatever is missing (see ["What the target computer needs"](#what-the-target-computer-needs)
below).

## Features

- **Fully local and private** - yt-dlp + whisper.cpp + Ollama on your machine. No cloud APIs, no API keys, no accounts; audio and transcripts never leave the computer.
- **One portable file** - a single ~50 KB executable (a Python zipapp, stdlib only) that runs anywhere Python 3.9+ does; `scp` it to another box and go.
- **Batch lists** - many videos in one run: several arguments, comma-separated lists, `--batch` files, or any mix; a failing video never stops the batch.
- **Parallel pipeline** - a worker pool overlaps transcription of later videos with summarization of earlier ones (`-j N`), so batches finish sooner than one-at-a-time.
- **Summary styles** - `--style actionable | short | study-guide`, or add your own style as a versioned prompt-variant file.
- **Prompt benchmarking** - `benchmark` scores the prompt variants with deterministic checks plus a local LLM-as-judge, so you promote a measured winner instead of a guess.
- **Obsidian-ready output** - LLM-readable plain-text transcripts; markdown notes with YAML frontmatter that drop into a vault as regular notes.
- **Doctor** - `doctor` inventories the machine with per-tool install hints; every command checks its own needs before starting instead of dying mid-run.

## Local vs cloud/API summarizers

| | vidsummarizer | cloud / API-key tools |
|---|---|---|
| API key, account | none, ever | required up front |
| Where audio and transcripts go | stay on your machine | uploaded to a service |
| Cost after setup | free (your CPU/GPU) | per-video or per-minute fees |
| Works offline | summarize: yes (saved transcripts, local Ollama + installed model); fetch/transcribe: no, they need YouTube | no |
| Move to another machine | copy one file | reinstall and re-authenticate |
| Batch a whole watchlist | `--batch list.txt -j 3` | usually one video at a time |

## Commands

```
vidsummarizer <url(s)>            full pipeline in one command (same as 'run')
vidsummarizer run <url(s)>        ...explicit form, with all stage options
vidsummarizer fetch <url(s)>      only download the mp3
vidsummarizer transcribe <url(s)> only the transcript (the old yt-transcribe)
vidsummarizer summarize <id|url|file> | --all    only the note (the old yt-summarize)
vidsummarizer benchmark           score & compare the prompt variants
vidsummarizer prompts <list|show|new|promote>   manage the prompt variants
vidsummarizer doctor               check this machine for everything the pipeline needs
```

It merges the two former tools, `yt-transcribe` 1.0.0 (download +
transcribe) and `yt-summarize` 1.0.0 (local-LLM summary) into a single
self-contained file (released under the name `yt-pipeline` until 3.0.0;
formats and behavior are unchanged, and the old repository URL redirects
here). All current behavior of the two original tools is preserved:
LLM-readable plain-text transcripts, and markdown summaries of at most
500 words (actionable content preferred, otherwise an overview) that
import into an Obsidian vault as regular notes. The summary *style* is
selectable per run (`--style`: actionable, short, study guide, ...); see
"Summary styles" below. See "Output formats" below.

Every video-taking command (`<url(s)>` above) accepts a whole list of
videos in one run: several arguments, comma-separated values, a `--batch`
FILE with one video per line, or any mix; batches are processed with a
small worker pool so transcription of later videos overlaps with the
summarization of earlier ones ("Batch processing" below).

## The one-file executable

`./build.sh` produces `dist/vidsummarizer`, a single self-contained file
(~50 KB) built with the Python standard library's `zipapp` module: the whole
`vidsummarizer` package is packed into one executable archive with a
`#!/usr/bin/env python3` shebang. It does not need this checkout, a git
clone, or any pip packages at runtime; it only needs a Python 3.9+
interpreter on the machine where it runs.

```bash
./build.sh                 # builds and smoke-tests dist/vidsummarizer
./dist/vidsummarizer --version
```

(Rebuild after changing any file under `vidsummarizer/`. The build stage copies
only the package into the archive; README, .git and data never get packed.)

### Transferring to another computer

```bash
scp dist/vidsummarizer user@host:~/vidsummarizer/
ssh user@host 'cd ~/vidsummarizer && ./vidsummarizer doctor'   # see what's missing
ssh user@host 'cd ~/vidsummarizer && ./vidsummarizer <youtube-url>'
```

The file is plain Python bytecode in a zip container, so it also runs on a
different Linux distribution or CPU architecture as long as that machine has
`python3` (3.9 or newer). To run it from a directory not on `PATH`
invocations, mark it executable once: `chmod +x vidsummarizer`.

Data (see "Data layout" below) accumulates next to the executable by
default, exactly like the old project checkouts. Optionally copy the old
`models/`, `audio/`, `transcripts/` folders alongside it to reuse caches;
the whisper model is auto-downloaded on first use anyway.

## What the target computer needs

Whatever cannot reasonably live inside the executable is detected at
runtime, with a clear message about what is missing and how to install it.
`vidsummarizer doctor` prints the full report; each command also checks its
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
$ ./vidsummarizer doctor
vidsummarizer 3.0.0 doctor: checking this machine for the local pipeline

Core:
  [ OK ] Python 3.12.3 (/usr/bin/python3)

Audio + transcription (commands: run, fetch, transcribe):
  [ OK ] yt-dlp 2026.10.10  (/home/user/.local/bin/yt-dlp)
  [ OK ] ffmpeg 6.1.1  (/usr/bin/ffmpeg)
  [MISS] whisper-cli (transcribes the audio locally (whisper.cpp))
         install: build whisper.cpp from https://github.com/ggml-org/whisper.cpp ...
  [AUTO] whisper model ggml-small.bin at /home/user/vidsummarizer/models/ggml-small.bin
         (downloaded automatically on first use, ~466 MiB ...)

Summaries (commands: run, summarize):
  [MISS] ollama (serves the local LLM that writes the summaries)
         install: install from https://ollama.com, then pull a model with: ollama pull qwen3.6

Data:
  [ OK ] workdir /home/user/vidsummarizer (writable)

Result: missing whisper-cli, ollama.
Without them only these commands can run: fetch.
Fix the [MISS] items (install hints above) and re-run 'vidsummarizer doctor'.
```

`doctor` exits 0 when the full pipeline can run and 1 when something is
missing, so it can gate automated setups.

## Usage

### Full pipeline

```bash
./vidsummarizer https://www.youtube.com/watch?v=iCvmsMzlF7o    # one command, end to end
./vidsummarizer run iCvmsMzlF7o --timestamps                   # explicit form
./vidsummarizer run jNQXAC9IVRw --no-keep-audio --stdout       # print the note, delete the mp3
./vidsummarizer url1,url2,iCvmsMzlF7o                          # a whole list in one run
./vidsummarizer run --batch watchlist.txt -j 3                 # ...or from a file, 3 at a time
```

Downloads the audio, writes the transcript to `transcripts/<Title>.txt`,
then the summary note into `summaries/<Title>.md` (both files are named
after the video's title; see "Output formats"), and prints what it did to
stderr. On the reference machine (Ryzen 7 5850U, CPU-only) a 21-minute
video takes about 7 minutes end to end (2 min whisper.cpp + 5 min qwen3.6
summarization).

### Batch processing (many videos at once)

`run`, `fetch` and `transcribe` take any number of videos. Three input
forms, freely combinable:

```bash
./vidsummarizer run https://youtu.be/a,https://youtu.be/b,jNQXAC9IVRw  # comma-separated
./vidsummarizer run url1 url2 url3                                     # several arguments
./vidsummarizer run --batch watchlist.txt                              # one video per line
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
[dQw4w9WgXcQ] * wrote .../transcripts/Rick Astley - Never Gonna Give You Up.txt (494 words, 3 paragraphs)
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
./vidsummarizer fetch jNQXAC9IVRw                              # mp3 only
./vidsummarizer fetch --batch list.txt -j 4                    # ...several at once
./vidsummarizer transcribe jNQXAC9IVRw                         # mp3 -> transcript
./vidsummarizer transcribe a,b,c --timestamps                  # several, with [MM:SS] markers
./vidsummarizer transcribe DP-VTlJkzOY --lang de               # force audio language
./vidsummarizer transcribe -m models/ggml-base.bin <id>        # different (auto-downloaded) model
./vidsummarizer transcribe --ytdlp '--cookies-from-browser firefox' <id>
./vidsummarizer summarize iCvmsMzlF7o                          # transcript -> note (by id)
./vidsummarizer summarize https://youtu.be/iCvmsMzlF7o         # by URL
./vidsummarizer summarize ~/any/transcript.txt                 # by file
./vidsummarizer summarize --all                                # every transcript without a note
./vidsummarizer summarize --all --force                        # redo everything
./vidsummarizer summarize --all --out ~/vault/Video\ notes     # write straight into a vault
./vidsummarizer summarize jNQXAC9IVRw --lang en                # force note language
./vidsummarizer summarize jNQXAC9IVRw --style short            # pick a summary style
```

Progress goes to stderr; transcripts and notes are files; `--stdout` on
run/summarize prints the note to stdout instead.

### Summary styles (`--style`)

The summarize step is selectable per run: the same transcript can be
turned into different kinds of output, chosen with `--style NAME` on
`summarize` or `run`:

```bash
./vidsummarizer summarize iCvmsMzlF7o --style short        # brief overview
./vidsummarizer summarize iCvmsMzlF7o --style study-guide  # study companion
./vidsummarizer summarize --all --style short --out ~/vault/Video\ notes
./vidsummarizer run url1,url2 --style actionable           # full pipeline, action list
```

Built-in styles:

| Style | What you get | Word cap |
|---|---|---|
| `actionable` | the concrete steps, decisions and recommendations as a numbered action list, always (even when the default would fall back to an overview); if the video has nothing actionable the note says so instead of inventing steps | 500 |
| `short` | a brief overview: one short paragraph on what the video covers and its single most important takeaway | 150 |
| `study-guide` | key concepts explained in your own words, a definitions list, and questions to test understanding; a study companion for reviewing the material | 700 |
| *(no `--style`)* | the standard summary, unchanged: actionable steps when the video has them, otherwise an overview | 500 |

Word caps: `short` uses 150 and `study-guide` 700 because their output
shape clearly needs a different bound than the usual 500 (one paragraph
stays far below it; concepts + definitions + test questions need a little
room above it). An explicit `--max-words N` always wins over a style's own
cap. Everything else is identical for every style: summaries are written
by the local model only (Ollama), the output is one Obsidian-ready
markdown note per video, long transcripts go through the same map-reduce
notes pass, and the over-limit rewrite/trim pass enforces the cap. The
note's frontmatter records which prompt variant produced it (`prompt:`),
so notes from different styles are distinguishable in the vault.

A style is just a prompt variant file (see "Adding a new style" below),
and `--style` accepts any variant name, so a benchmark candidate can be
tried per run too: `--style v2-tighter`. An unknown name fails at startup
with the list of available styles.

`summarize --all` treats a video as done once *any* note exists, whatever
style produced it; re-summarize in another style with `--force`, or write
to a different `--out` directory to keep both versions of the note.

### Adding a new style

Styles are prompt variant files with a small `style` block, so adding one
is a new file, not a code change:

1. Copy a starting point:
   `vidsummarizer prompts new deep-dive --from study-guide`
   (creates `<workdir>/prompts/deep-dive.json`; in a source checkout it
   lands in the repo's `prompts/`, where you can also just copy a `.json`
   file by hand).
2. Edit the JSON:
   - `prompts.system/final/notes/reduce/condense`: the five prompt
     templates, plain text with placeholders (`{transcript}`,
     `{max_words}`, `{lang_instr}`, `{notes}`, `{chunk}`, ...). The output
     expectations (which headings, sections and lists the note uses) live
     in this wording, so be as specific as the built-ins are.
   - `style`: what makes the variant a style; every key is optional:
     `label` (name shown in messages), `summary` (one-line description),
     `max_words` (the style's own word cap, applied when `--max-words` is
     not given), `fallback_heading` (heading prepended if the model output
     has no `##` heading). A variant without a `style` block is a plain
     variant: still selectable with `--style NAME`, just not advertised as
     a style.
   - set `description` (shown by `vidsummarizer prompts list`) and bump
     `version` when you later change a file that exists somewhere.
3. Use and compare it:
   `vidsummarizer summarize <id> --style deep-dive`, and
   `vidsummarizer benchmark --prompts deep-dive,study-guide` to score it
   against another style on the benchmark set.

Names are lowercase letters, digits and dashes, matching the file name
(`prompts/<name>.json`). Malformed files (unknown placeholders, bad
`style` values, wrong `name`) are rejected at load time with a precise
message, so a typo fails at startup instead of mid-summary. Which
directory the files come from follows the usual prompts resolution
(`--prompts-dir`, `$YT_PROMPTS_DIR`, `<workdir>/prompts`, the copy bundled
into the executable); `vidsummarizer prompts list` shows the directory in
use and everything it found.

### Prompt benchmarking (`benchmark`)

Summary quality is measurable, so prompt changes can be compared instead
of guessed at. `benchmark` runs the summarizer over a fixed benchmark set
of transcripts with one or more prompt variants and scores every summary:

```bash
vidsummarizer benchmark                              # every variant in the prompts dir
vidsummarizer benchmark --prompts v1-baseline,short  # just these, compared
vidsummarizer benchmark --judge-model llama3.2       # a different local judge
vidsummarizer benchmark --min-overall 3.5            # quality gate (make bench-real)
```

**The benchmark set** is `benchmarks/set.json` plus
`benchmarks/transcripts/` (three transcripts: the summarize-stage test
fixture, an actionable how-to video and a conceptual talk, so both summary
kinds are exercised). Each case declares the expected `kind`
(`actionable`/`overview`) and `must_contain` facts that must survive
summarization. `--set FILE` points at another set; a deployed executable
reads the copy bundled into it.

**Scoring** has two halves:

- *Automatic checks* (deterministic, offline): word cap, markdown/Obsidian
  structure (top heading, no code fences, no rules, no H1), the right
  summary kind for the case, and the must-contain facts. A style variant
  is checked against its own top heading and word cap.
- *LLM-as-judge*: a local model (Ollama; `--judge-model`, default the
  summarizer's own model) rates each summary 1-5 against a rubric:
  faithfulness to the transcript, actionability, clarity. Judging never
  leaves the machine, like the rest of the pipeline.

A case's `Overall` = mean of the three judge dimensions minus 0.25 per
failed automatic check; variants are ranked by their average. The report
is printed and written to `<workdir>/benchmarks/results/<timestamp>/`
(`report.md`, machine-readable `results.json`, and the generated
`summaries/<variant>/<case>.md` for eyeballing). Judged scores vary a
little from run to run; compare variants within one run, and re-run to
confirm a close call.

**Promoting a winner is manual, by design**: the report informs, the
human decides. `vidsummarizer prompts promote NAME` asks for confirmation
(`--yes` skips it), then points `active.txt` in the prompts directory at
the winner, and every later `run`/`summarize` uses it. In a checkout that
pointer is a versioned file in `prompts/` (commit it); with a deployed
executable the variants are first materialized into `<workdir>/prompts`,
then the pointer is written there.

**Quality gate**: `make bench-real` (= `benchmark --min-overall 3.5`)
exits non-zero when the *active* variant scores below 3.5 overall or
fails an automatic check, so a regression in summary quality is caught
like any other test failure. The offline suite (`make test`) already
causes the deterministic half on every run: structure checks, judge-answer
parsing, scoring/report plumbing, and that the shipped v1-baseline is
byte-identical to the historical prompts.

### Options

Batch input and parallelism (`run`, `fetch`, `transcribe`):

| Option | Default | Description |
|---|---|---|
| `-b, --batch FILE` | none | file with one video URL/id per line (`#` comments and blank lines ignored); `-` reads stdin; repeatable, combinable with positional URLs/ids |
| `-j, --jobs N` | `2` (env `YT_JOBS`) | videos processed concurrently; `1` = strictly sequential |

Transcription stage (`run`, `transcribe`):

| Option | Default | Description |
|---|---|---|
| `-o, --out FILE` / `-o, --transcript-out FILE` | `<workdir>/transcripts/<Title>.txt` | output transcript path (single video only) |
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
| `--style NAME` | none | summary style: the shape the note takes (built-in: `actionable`, `short`, `study-guide`; any other prompt variant name works too). Default: the standard actionable-or-overview summary. See "Summary styles" |
| `--out DIR` | `<workdir>/summaries` | output directory for notes |
| `--host URL` | `http://127.0.0.1:11434` (env `OLLAMA_HOST`) | Ollama base URL |
| `--max-words N` | 500, or the selected style's own cap | hard word cap for the summary (explicit values beat the style's cap) |
| `--lang CODE` (summarize) / `--summary-lang CODE` (run) | transcript's language | force the summary language |
| `--chunk-words N` | 1500 | words per map-reduce chunk for long transcripts (`0` = never chunk) |
| `--notes-words N` | 220 | word cap for per-chunk notes |
| `--think` | off | let the model use thinking mode (5-10x slower) |
| `--force` | off | re-summarize even if a note exists (`--all` mode) |
| `--stdout` | off | print the note instead of writing a file |
| `--no-autostart` | off | never start `ollama serve` ourselves |
| `--prompts-dir DIR` | see "Adding a new style" | directory holding the prompt variant `*.json` files (default: `$YT_PROMPTS_DIR`, else `<workdir>/prompts` if it has any, else the bundled copy) |
| `--timeout S` | 900 | per-request timeout |
| `--max-num-ctx N` | 32768 | ceiling for the Ollama context window |

`summarize` only: `--all`, `--transcripts DIR` (default `$YT_TRANSCRIPTS_DIR`
else `<workdir>/transcripts`). `benchmark` and `prompts` have their own
options (sets, judge model, quality gate, confirmation); see
`vidsummarizer benchmark --help` / `vidsummarizer prompts --help`. All
commands: `--workdir DIR` (default `$YT_WORKDIR`, else the executable's
own directory).

Accepted URL forms: `youtube.com/watch?v=ID`, `youtu.be/ID`,
`youtube.com/shorts/ID`, `/live/ID`, `/embed/ID`, or a bare 11-character
video id.

## Data layout

Everything lives in one work directory, resolved in this order:

1. `--workdir DIR`
2. `$YT_WORKDIR`
3. the directory containing the executable (like the old project checkouts)
4. if that is read-only (e.g. the binary was installed to `/usr/local/bin`):
   `~/.local/share/vidsummarizer` (with a notice on stderr)

```
<workdir>/
  audio/          downloaded mp3s (cache; delete freely)
  transcripts/    plain-text transcripts
  summaries/      Obsidian-ready markdown notes
  models/         whisper GGML models (auto-downloaded on first use)
  prompts/        prompt variants + active.txt (created on first write)
  benchmarks/     the fixed benchmark set; results/ holds the reports
```

Old data can be adopted by copying the old projects' `models/`,
`audio/`, `transcripts/` and `summaries/` folders into the workdir; file
formats are unchanged, and the summarizer still accepts transcripts written
by the old `yt-transcribe` (and vice versa). Old id-named transcripts
(`transcripts/<video_id>.txt`) keep working: they are found by video id
like the new title-named ones, and re-transcribing a video replaces its
id-named file with the title-named one.

## Output formats

**Transcript** (`transcripts/<Title>.txt`): UTF-8 plain text, a short
`key: value` header (`title`, `channel`, `upload_date`, `duration`,
`video_id`, `url`, `language`, `model`, `tool`), one blank line, then the
transcript as paragraphs split at silence gaps of >= 1.5 s (soft cap ~280
words at sentence ends, hard cap 500 words). Designed for LLM consumption.

**Summary note** (`summaries/<Title>.md`): Obsidian-ready markdown with YAML
frontmatter properties (`title`, `channel`, `date`, `url`, `video_id`,
`duration`, `language`, `transcribed_with`, `summarized_with`, `prompt`,
`summary_words`, `summarizer`, `created`, `tags`), a `# title` heading, a
source line with the YouTube link, then the summary: `## Actionable
summary` with numbered `### Actionable steps` when the video has actionable
content, otherwise a `## Overview` of 2-4 paragraphs. With `--style NAME`
the body instead takes that style's shape (see "Summary styles"), and the
`prompt:` frontmatter property records which prompt variant (style) wrote
the note. The summary body is capped at 500 words (`--max-words`, or the
selected style's own cap); if the model runs over it is asked once for a
tighter rewrite, then trimmed at a sentence boundary. Long transcripts are
summarized map-reduce style (per-chunk notes, then a final pass).

**File naming** (both outputs): files are named after the video's title,
made filesystem-safe: characters hostile to filenames (path separators,
`\\/:*?"<>|#^[]`, control characters, and what Obsidian forbids in note
names) are dropped, whitespace is collapsed, surrounding dots are trimmed,
and the stem is capped at 90 characters / 200 UTF-8 bytes. The video id
stays discoverable through the `video_id` in the header and frontmatter,
so lookups by id keep working (`summarize <id>`, `summarize --all`); the
audio cache (`audio/<video_id>.mp3`) keeps its id-based name, which never
collides. Two different videos whose sanitized titles collide never
overwrite each other: the second one gets ` (<video_id>)` appended to its
filename, and re-running a video always updates its own single file
(name resolution and the write are serialized, both across the workers of
one batch and across concurrent yt-pipeline processes sharing the work
directory; the write itself is an atomic rename, so a concurrent reader
never sees a half-written file). Each output directory also gets a hidden,
empty `.yt-pipeline.lock` file, the mutex behind that serialization.

## Differences from the original two tools

Only what the merge itself forces; formats and pipeline behavior are
unchanged:

- The `tool:` transcript line and `summarizer:` frontmatter now say
  `vidsummarizer 3.0.0` instead of the old tool names.
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
vidsummarizer/
  vidsummarizer/            the tool (Python stdlib only)
    cli.py               argument parsing and subcommand dispatch
    common.py            shared helpers, tool detection, workdir resolution,
                         video-list parsing, thread-safe tagged logging
    transcribe.py        yt-dlp + whisper.cpp stage (from yt-transcribe)
    summarize.py         Ollama stage (from yt-summarize)
    prompts.py           versioned prompt variants: load/validate/promote
    bench.py             benchmark command: scoring, local LLM judge, reports
    batch.py             worker pool: list of videos, overlapped, failures isolated
    pipeline.py          run/fetch/transcribe/doctor commands
  tests/                 unit + E2E test suite: `make test` (see Testing)
  prompts/               the summarizer's prompt variants (versioned JSON)
  benchmarks/            fixed benchmark set (set.json + transcripts/)
  build.sh               builds dist/vidsummarizer (zipapp)
  Makefile               test / build entry points
  .github/               CI workflows: test gate + automated releases (see below)
  README.md
```

Run from the checkout without building: `python3 -m vidsummarizer <command> ...`
(same behavior; the workdir then defaults to the repo root).

## Testing

The test suite needs nothing but Python itself (stdlib `unittest`, like the
tool). The default run is fast, offline and deterministic:

```
make test        # unit + E2E, a few seconds; no network, no models
make test-unit   # unit tests only
make test-e2e    # end-to-end tests only
make test-real   # opt-in: REAL whisper-cli + local model on the bundled
                 # speech fixture (no network)
make test-live   # opt-in: a real 19-second YouTube video through the real
                 # pipeline (network + model; ~10-30s)
make bench-real  # opt-in: benchmark the prompt variants with the real
                 # local LLM; gates on the active variant's score
```

- **Unit tests** (`tests/unit/`) cover the modules in isolation with every
  external process stubbed: video-id and batch-list parsing, tagged
  logging, the subprocess wrapper and tool detection, workdir resolution
  (including the read-only fallback), paragraphing rules and the transcript
  format, transcript parsing, chunking, markdown cleanup and note building,
  prompt variant loading/validation and the style metadata (`--style`),
  the prompt-variant registry end to end (active pointer, promotion,
  materialization, the byte-exact v1-baseline guard), the benchmark
  harness (set loading, automatic checks, judge-answer parsing, scoring,
  reports),
  the Ollama HTTP client (against an in-process fake server), the worker
  pool's failure containment, and the CLI parser/dispatch.
- **E2E tests** (`tests/e2e/`) run the real CLI (`python3 -m vidsummarizer`)
  as a subprocess through the whole flow. `tests/fakes/` is put first on
  `PATH`, so yt-dlp, whisper-cli, ffmpeg and ollama are stubs serving the
  fixtures, and an in-process fake Ollama HTTP server answers the summarize
  stage. Each run gets its own temporary workdir (with a dummy whisper model
  seeded, so nothing downloads); covered: fetch caching, transcribe
  (timestamps, wav fallback, audio retention), batch lists/files, dedup,
  failure isolation and exit codes, summarize (by id/url/file, `--all`
  skip logic, over-limit rewrite, language override, `--style` selection
  including its own word cap) and the full `run` pipeline, plus the
  benchmark and prompts commands (report writing, the `--min-overall`
  gate, promotion with and without confirmation) against the fake server
  doubling as the judge. If
  `dist/vidsummarizer` exists, extra checks run the built executable too
  (including one `--style` run, proving the styles are packed into it).

### Fixtures (`tests/fixtures/`)

| File | What it is |
|---|---|
| `audio/sample.mp3` | 7 seconds of real synthesized speech ("Hello. This is a short test recording for the yt pipeline test suite. The quick brown fox..."); every fake "download" serves this file, and `make test-real` transcribes it for real |
| `meta/sample.json` | canned `yt-dlp -J` metadata returned by the fake yt-dlp |
| `whisper/sample.json` | canned whisper-cli JSON output (two paragraphs, separated by a 1.7s silence gap) |
| `transcript/sample.txt` | a finished transcript (metadata header + body) for the summarize stage |

The benchmark set lives next door in `benchmarks/` (`set.json` +
`transcripts/`): `sample-short.txt` is a copy of the `transcript/sample.txt`
fixture, and `howto-tls.txt` / `overview-history.txt` extend it with an
actionable and an overview video. A unit test keeps the copy in sync with
the fixture, so the set and the test suite cannot drift apart.

### Adding a regression test

When a bug is found:

1. **Reproduce it in the smallest failing test.** Pure logic bugs usually
   belong in `tests/unit/` against a single function; bugs in CLI behavior,
   exit codes, file layout or how stages interact belong in `tests/e2e/`,
   driven through the real executable with the fakes.
2. **Name it after the symptom** (e.g. `test_transcribe_keeps_audio`), keep
   arrange/act/assert tight, and add or reuse fixtures under
   `tests/fixtures/` instead of reaching for the network. If the fakes must
   misbehave to trigger the bug, extend the env-var controls documented in
   `tests/fakes/yt-dlp` / `whisper-cli` (e.g. `YT_FAKE_YTDLP_FAIL_IDS`).
3. **Fix the code, then `make test`.** The suite must be green before and
   after; the test doubles as the documentation of the bug.

### Automated releases (CI)

GitHub Actions (`.github/workflows/`) gates every change and publishes
releases automatically:

- **Every pull request** (and every push to `master`) runs the gate:
  `make build` plus `make test` on Python 3.9 and 3.14, a version check on
  the built executable, and a `doctor` sanity check on the bare runner
  (it must detect the missing tools and exit 1). The suite is offline and
  deterministic, so a red gate always means a real regression.
- **Every push to `master`** then publishes: if the `VERSION` in
  `vidsummarizer/__init__.py` has no `v<VERSION>` release yet, the workflow
  tags that commit, builds `dist/vidsummarizer` and attaches it
  (uncompressed, as documented above) to a GitHub Release with notes from
  `.github/release-notes-template.md` plus the commits since the previous
  release, each with its full message
  (`.github/format-changelog.py`), so every change explains itself.
  A push without a version bump runs the tests and does nothing else.

So the normal release procedure is: bump `VERSION`, update docs if needed,
`make test`, commit, push. If the gate is red, nothing is released. The
`release` workflow can also be dispatched manually to re-run the check.

## Troubleshooting

- **"[MISS] ..." at startup or in doctor**: install the listed tool on the
  target machine; hints are printed next to every missing item.
- **HTTP 403 / throttling from YouTube**: transient; downloads retry 3x.
  Persistent failures usually mean yt-dlp needs an update (`yt-dlp -U`).
- **"model 'qwen3.6' is not installed"**: `ollama pull qwen3.6`, or pick
  another with `--model`; `ollama list` shows what is installed.
- **Ollama not running**: `run`/`summarize` start `ollama serve`
  automatically (log: `/tmp/ollama-serve-vidsummarizer.log`); with
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
