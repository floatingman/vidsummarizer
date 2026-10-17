## What this is

**yt-pipeline {{VERSION}}** is one portable executable for the whole local YouTube pipeline:

```
YouTube URL -> yt-dlp (mp3) -> whisper.cpp transcript -> local-LLM summary (Obsidian-ready note)
```

Audio download, transcription and summarization all run on your machine: no cloud APIs, no API keys. The tool is Python standard library only, so it ships as one small file (a Python zipapp) that can be copied to other computers.

```bash
yt-pipeline <url>[,<url>...]   full pipeline in one command (same as 'run')
yt-pipeline run --batch list.txt -j 2     ...several videos, overlapped jobs
yt-pipeline fetch <url>        only download the mp3
yt-pipeline transcribe <url>   only the transcript
yt-pipeline summarize --all    only the notes (also: by id, url or transcript file)
yt-pipeline summarize --style short   ...or in another summary style (actionable, study guide, ...)
yt-pipeline benchmark          score every prompt variant on the benchmark set (LLM judge)
yt-pipeline prompts            list / show / new / promote the prompt variants
yt-pipeline doctor             check this machine for everything the pipeline needs
```

Transcripts are LLM-readable plain text; summaries are markdown notes (at most 500 words by default: actionable content preferred, otherwise an overview, or any installed style via `--style`) that import into an Obsidian vault as regular notes. The summarizer's prompts are versioned JSON variants you can list, compare with `benchmark` and promote. Full documentation is in the repo's [README](https://github.com/floatingman/yt-pipeline/blob/master/README.md).

## What the target computer needs

The asset below is a single file that needs a **Python 3.9+** interpreter to run (preinstalled on virtually all Linux distros). Everything else is detected at runtime with a clear install hint instead of a stack trace; `yt-pipeline doctor` prints the full report and exits non-zero when something is missing.

| Requirement | Needed for | Install |
|---|---|---|
| `yt-dlp` on PATH | every command that touches YouTube | `pipx install yt-dlp` or the standalone binary from [yt-dlp/yt-dlp](https://github.com/yt-dlp/yt-dlp) |
| `ffmpeg` on PATH | audio download (mp3 encode) | system package manager, e.g. `sudo apt install ffmpeg` |
| `whisper-cli` on PATH | transcription | build [whisper.cpp](https://github.com/ggml-org/whisper.cpp): `cmake -B build && cmake --build build -j --target whisper-cli`, then put `whisper-cli` on PATH |
| `ollama` on the machine | summaries | install from [ollama.com](https://ollama.com); the tool auto-starts `ollama serve` if the server is not running |
| an Ollama model | summaries | `ollama pull qwen3.6` (default model, ~23 GB); any other tag via `--model` / `--llm-model` |
| whisper GGML model | transcription | nothing to do: downloaded automatically on first use (~466 MiB for `ggml-small.bin`) |

Not bundled on purpose (too big / machine-specific): the whisper model files, the Ollama installation and its models, and the external CLIs above.

## Download and run it on another computer

The asset is the executable itself, uncompressed (a single small file; no archive to unpack). Since this repository is private, you must be signed in to GitHub with access to it to download.

```bash
# on the target machine (or download the 'yt-pipeline' asset in the browser):
gh release download {{VERSION}} --repo floatingman/yt-pipeline     # needs gh, signed in
# or from the machine that has it: scp yt-pipeline user@host:~/yt-pipeline/

chmod +x yt-pipeline          # once; it runs from any directory
./yt-pipeline doctor          # see what is missing on this machine
./yt-pipeline https://www.youtube.com/watch?v=iCvmsMzlF7o     # end to end
```

The file is plain Python in a zip container, so it also runs on a different Linux distribution or CPU architecture than the build machine, as long as that machine has `python3` (3.9 or newer). Data accumulates next to the executable by default (`audio/`, `transcripts/`, `summaries/`, `models/`; override with `--workdir` or `$YT_WORKDIR`).

## Build and provenance

The asset was built by the documented process (stdlib `zipapp` via `build.sh`) from the commit this release tags, after the full offline test suite passed in CI. SHA-256 of the asset: `{{SHA256}}`.
