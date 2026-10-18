"""Summarization stage, ported from yt-summarize 1.0.0.

Reads the plain-text transcripts written by the transcription stage,
summarizes them with a local model served by Ollama, and writes an
Obsidian-ready markdown note per video:

    transcripts/<Title>.txt  ->  Ollama (local LLM)  ->  summaries/<Title>.md

Both files are named after the (sanitized) video title; the video id stays
discoverable through the header/frontmatter metadata, so 'summarize <id>'
and 'summarize --all' keep working, including for pre-2.3 id-named
transcripts already on disk.

External dependencies (checked at runtime): the `ollama` binary plus the
model given by --model (default qwen3.6), both served on the local Ollama
HTTP API. The server is auto-started if `ollama serve` is not running.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from . import VERSION
from .common import (die, find_transcript, log, naming_lock, safe_name,
                     titled_path, warn)  # safe_name is re-exported here

TOOL = f"yt-pipeline {VERSION} (local LLM via Ollama)"

DEFAULT_MODEL = "qwen3.6"

# Hard word ceiling for summary text. --max-words overrides it; a style
# variant may carry its own cap (prompts/<style>.json "style.max_words")
# which applies when --max-words was not given.
DEFAULT_MAX_WORDS = 500

# Language codes -> instruction phrase (fallback: "the same language as the
# transcript").  Anything not listed falls back to that phrase, which works.
LANG_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "sv": "Swedish",
    "no": "Norwegian", "da": "Danish", "fi": "Finnish", "pl": "Polish",
    "ru": "Russian", "uk": "Ukrainian", "tr": "Turkish", "ar": "Arabic",
    "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "vi": "Vietnamese",
    "th": "Thai", "id": "Indonesian", "cs": "Czech", "el": "Greek",
    "he": "Hebrew", "hu": "Hungarian", "ro": "Romanian",
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def count_words(text):
    """Words = whitespace-separated tokens (markdown syntax counts as words;
    it is part of the visible note, so the cap should include it)."""
    return len(text.split())


def sentences(text):
    """Split into sentences; falls back to hard splitting for scripts that do
    not use sentence-final punctuation (CJK)."""
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [p for p in parts if p.strip()]


def chunk_text(text, chunk_words):
    """Split text into sentence-aligned chunks of ~chunk_words words."""
    sents = sentences(text)
    if not sents:
        return [text] if text.strip() else []
    chunks, cur, cur_words = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and cur_words + w > chunk_words:
            chunks.append(" ".join(cur))
            cur, cur_words = [], 0
        cur.append(s)
        cur_words += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def yaml_str(value):
    """JSON string encoding is valid YAML double-quoted scalar syntax."""
    return json.dumps(str(value), ensure_ascii=False)


def normalize_host(host):
    host = host or "http://127.0.0.1:11434"
    if "://" not in host:
        host = "http://" + host
    return host.rstrip("/")


# --------------------------------------------------------------------------- #
# Transcript parsing
# --------------------------------------------------------------------------- #

HEADER_LINE = re.compile(r"^([a-z_]+):\s?(.*)$")


def parse_transcript(text):
    """Split a transcript file into (metadata dict, body text).

    The transcription stage writes a `key: value` header (title, channel,
    upload_date, duration, video_id, url, language, model, tool), one blank
    line, then the transcript as plain paragraphs.  Plain text files without
    a header are accepted too; metadata comes back empty.
    """
    meta, lines, i = {}, text.splitlines(), 0
    for line in lines:
        m = HEADER_LINE.match(line)
        if not m:
            break
        meta[m.group(1)] = m.group(2).strip()
        i += 1
    # header must be followed by a blank line to count as a header
    if i and i < len(lines) and lines[i].strip() != "":
        return {}, text.strip()
    body = "\n".join(lines[i:]).strip()
    return meta, body


def transcript_language_instruction(meta, body):
    lang = (meta.get("language") or "").strip().lower()
    name = LANG_NAMES.get(lang)
    if name:
        return f"Write the summary in {name}."
    if lang:
        return f"Write the summary in the transcript's language (language code '{lang}')."
    return "Write the summary in the same language as the transcript."


# --------------------------------------------------------------------------- #
# Ollama client
# --------------------------------------------------------------------------- #

class Ollama:
    def __init__(self, host, model, timeout, no_autostart=False):
        self.host = normalize_host(host)
        self.model = model
        self.timeout = timeout
        if not self._ping():
            if no_autostart:
                die(f"cannot reach Ollama at {self.host} "
                    f"(start it with: ollama serve, or systemd: systemctl start ollama)")
            self._autostart()

    # -- server lifecycle ---------------------------------------------------

    def _get(self, path, timeout=10):
        req = urllib.request.Request(self.host + path)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def _ping(self):
        try:
            self._get("/api/version", timeout=3)
            return True
        except Exception:
            return False

    def _autostart(self):
        log(f"Ollama not running at {self.host}; trying to start 'ollama serve' ...")
        try:
            logf = open(os.path.join(os.environ.get("TMPDIR", "/tmp"),
                                     "ollama-serve-yt-pipeline.log"), "ab")
            subprocess.Popen(["ollama", "serve"], stdout=logf, stderr=logf,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        except FileNotFoundError:
            die("Ollama is not installed (command 'ollama' not found). "
                "Install it from https://ollama.com and re-run.")
        for _ in range(60):
            time.sleep(1.0)
            if self._ping():
                log("Ollama started.")
                return
        die("started 'ollama serve' but it did not answer at "
            f"{self.host}/api/version within 60s; see the log in /tmp.")

    def check_model(self):
        try:
            tags = self._get("/api/tags")
        except Exception as e:
            die(f"cannot list models from Ollama: {e}")
        names = [m.get("name", "") for m in tags.get("models", [])]
        for n in names:
            if n == self.model or n.split(":")[0] == self.model:
                return
        hint = f"\nInstalled models: {', '.join(names)}" if names else \
            "\nNo models are installed yet."
        die(f"model '{self.model}' is not installed in Ollama.{hint}\n"
            f"Pull it first with:  ollama pull {self.model}")

    # -- chat ---------------------------------------------------------------

    def chat(self, system, user, options, think=False, retries=3):
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": options,
        }
        if think:
            body["think"] = True
        else:
            body["think"] = False  # qwen3.6 defaults to thinking on; disable it
        data = json.dumps(body).encode()
        last_err = None
        for attempt in range(1, retries + 1):
            req = urllib.request.Request(
                self.host + "/api/chat", data=data,
                headers={"Content-Type": "application/json"})
            try:
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.load(r)
                content = out.get("message", {}).get("content", "")
                if out.get("done_reason") == "load" or not content.strip():
                    raise RuntimeError("empty response from model")
                stats = {
                    "s": time.time() - t0,
                    "prompt_tok": out.get("prompt_eval_count", 0),
                    "gen_tok": out.get("eval_count", 0),
                }
                if stats["gen_tok"] and stats["s"] > 0.5:
                    stats["tok_s"] = stats["gen_tok"] / max(
                        (out.get("eval_duration", 0) or 1e9) / 1e9, 1e-6)
                else:
                    stats["tok_s"] = 0.0
                return content, stats
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read(400).decode("utf-8", "replace")
                except Exception:
                    pass
                # models without thinking support reject think:true
                if e.code == 400 and think and "think" in detail.lower():
                    body.pop("think", None)
                    data = json.dumps(body).encode()
                    continue
                last_err = f"HTTP {e.code}: {detail or e.reason}"
            except Exception as e:
                last_err = str(e)
            if attempt < retries:
                wait = 5 * attempt
                log(f"  request failed ({last_err}); retrying in {wait}s "
                    f"({attempt}/{retries}) ...")
                time.sleep(wait)
        die(f"Ollama request failed after {retries} attempts: {last_err}")


# --------------------------------------------------------------------------- #
# Prompting
#
# The five prompt templates live as versioned JSON "variant" files in the
# project (prompts/*.json), not as constants here; see ytpipeline/prompts.py.
# v1-baseline.json holds the exact historical wording of this stage.
# --------------------------------------------------------------------------- #

from .prompts import (PromptSet, load_active, load_variant,
                      resolve_prompts_dir, styles_in, variant_names)


def prompts_for(opts):
    """The prompt variant to use for this run, in priority order:

    1. opts.prompts    - pinned by a caller (benchmark, tests)
    2. opts.style      - a summary style (or any variant) chosen per run
                         with --style NAME
    3. the active variant of the resolved prompts directory (the default
       summary behavior, untouched by styles)
    """
    pinned = getattr(opts, "prompts", None)
    if isinstance(pinned, PromptSet):
        return pinned
    d = resolve_prompts_dir(getattr(opts, "prompts_dir", None),
                            getattr(opts, "workdir", None))
    style = (getattr(opts, "style", None) or "").strip()
    if style:
        names = variant_names(d)
        if style not in names:
            styled = ", ".join(f"{s.name} ({s.style_label})"
                               for s in styles_in(d)) or "none"
            die(f"unknown style '{style}' in {d}\n"
                f"styles: {styled}\n"
                f"all prompt variants: {', '.join(names) or 'none'}")
        return load_variant(d, style)
    return load_active(d)


# --------------------------------------------------------------------------- #
# Summarization pipeline
# --------------------------------------------------------------------------- #

def gen_options(user_content_words, gen_word_budget, opts, temperature):
    """Pick num_ctx big enough for prompt + generation."""
    prompt_tok = int(user_content_words * 1.45) + 900  # words->tokens + instructions
    num_ctx = prompt_tok + int(gen_word_budget * 2.0) + 350
    num_ctx = max(2048, min(num_ctx, opts.max_num_ctx))
    return {
        "num_ctx": num_ctx,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,   # model's baked-in default 1.5 hurts fidelity
    }


def clean_markdown(text, fallback="## Overview"):
    """Normalize model output: keep from the first '## ' heading on, collapse
    whitespace, drop code fences if the model wrapped the answer. A summary
    without any '##' heading gets `fallback` prepended (styles use their own
    top heading there)."""
    text = text.strip()
    # strip a wrapping code fence
    fence = re.match(r"^```[a-zA-Z]*\n(.*)\n```$", text, re.S)
    if fence:
        text = fence.group(1).strip()
    idx = None
    for m in re.finditer(r"^##\s+.*$", text, re.M):
        idx = m.start()
        break
    if idx is not None:
        text = text[idx:]
    else:
        text = fallback + "\n\n" + text
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def language_instruction(opts, meta, body):
    if opts.lang:
        name = LANG_NAMES.get(opts.lang.strip().lower(), opts.lang.strip())
        return f"Write the summary in {name}."
    return transcript_language_instruction(meta, body)


def trim_to_word_cap(text, cap):
    """Last-resort hard trim that keeps headings and stops at the word cap;
    used only if the model ignores the limit after a rewrite pass."""
    out, words = [], 0
    for block in re.split(r"\n{2,}", text):
        if re.match(r"^#{1,6}\s", block):  # keep headings
            out.append(block)
            words += len(block.split())
            continue
        for s in sentences(block):
            w = len(s.split())
            if words + w > cap:
                warn(f"  summary still over {cap} words after rewrite; "
                     f"hard-trimmed to {words} words")
                return "\n\n".join(out)
            out.append(s)
            words += w
        # rejoin sentences of one paragraph
    return "\n\n".join(out)


def summarize_transcript(ollama, path, opts):
    """Summarize one transcript file; returns (markdown_note, info dict)."""
    t_start = time.time()
    raw = open(path, encoding="utf-8", errors="replace").read()
    meta, body = parse_transcript(raw)
    if not body:
        die(f"{path}: transcript is empty")
    vid = meta.get("video_id", "")
    tag = vid or os.path.basename(path)
    ps = prompts_for(opts)
    # a style may set its own word cap; an explicit --max-words always wins
    if getattr(opts, "max_words", None) is None:
        opts.max_words = ps.max_words or DEFAULT_MAX_WORDS
    lang_instr = language_instruction(opts, meta, body)
    title = meta.get("title") or os.path.splitext(os.path.basename(path))[0]

    # context line so the model knows what video this is (title/channel only)
    context = f"Video title: {title}"
    if meta.get("channel"):
        context += f"\nChannel: {meta['channel']}"

    if opts.chunk_words <= 0:
        chunks = [body]
    else:
        chunks = chunk_text(body, opts.chunk_words)
    body_words = count_words(body)

    if len(chunks) <= 1:
        log(f"[{tag}] {body_words} words -> single pass ({ollama.model}, "
            f"prompt {ps.name})")
        prompt = ps.final.format(
            lang_instr=lang_instr, max_words=opts.max_words,
            transcript=context + "\n\n" + body)
        options = gen_options(count_words(prompt), opts.max_words, opts, 0.3)
        summary, st = ollama.chat(ps.system, prompt, options, think=opts.think)
        log(f"[{tag}] summary drafted in {st['s']:.0f}s "
            f"({st['gen_tok']} tok, {st['tok_s']:.1f} tok/s)")
    else:
        log(f"[{tag}] {body_words} words -> {len(chunks)} chunks "
            f"of <= {opts.chunk_words} words ({ollama.model})")
        notes = []
        for i, chunk in enumerate(chunks, 1):
            prompt = ps.notes.format(
                i=i, n=len(chunks), lang_instr=lang_instr,
                notes_words=opts.notes_words, chunk=chunk)
            options = gen_options(count_words(prompt), opts.notes_words, opts, 0.3)
            note, st = ollama.chat(ps.system, prompt, options, think=opts.think)
            notes.append(note.strip())
            log(f"[{tag}] notes {i}/{len(chunks)}: {count_words(note)} words "
                f"in {st['s']:.0f}s ({st['tok_s']:.1f} tok/s)")
        combined = "\n\n".join(notes)
        prompt = ps.reduce.format(
            lang_instr=lang_instr, max_words=opts.max_words,
            notes=context + "\n\n" + combined)
        options = gen_options(count_words(prompt), opts.max_words, opts, 0.3)
        summary, st = ollama.chat(ps.system, prompt, options, think=opts.think)
        log(f"[{tag}] final summary from notes in {st['s']:.0f}s "
            f"({st['gen_tok']} tok, {st['tok_s']:.1f} tok/s)")

    summary = clean_markdown(summary, ps.fallback_heading)
    words = count_words(summary)
    if words > opts.max_words:
        log(f"[{tag}] {words} words > {opts.max_words}; asking for a tighter rewrite")
        prompt = ps.condense.format(actual=words, max_words=opts.max_words,
                                    text=summary)
        options = gen_options(count_words(prompt), opts.max_words, opts, 0.2)
        summary, st = ollama.chat(ps.system, prompt, options, think=opts.think)
        summary = clean_markdown(summary, ps.fallback_heading)
        if count_words(summary) > opts.max_words:
            summary = trim_to_word_cap(summary, opts.max_words)
    summary_words = count_words(summary)

    note = build_note(title, meta, summary, summary_words, ollama.model,
                      prompt_name=ps.name)
    info = {"video_id": vid, "words": summary_words,
            "elapsed": time.time() - t_start, "chunks": len(chunks),
            "summary": summary, "prompt": ps.name}
    return note, info


def build_note(title, meta, summary, summary_words, model, prompt_name=""):
    fm = [
        "---",
        f"title: {yaml_str(title)}",
        f"channel: {yaml_str(meta.get('channel', 'unknown'))}",
        f"date: {yaml_str(meta.get('upload_date', ''))}",
        f"url: {yaml_str(meta.get('url', ''))}",
        f"video_id: {yaml_str(meta.get('video_id', ''))}",
        f"duration: {yaml_str(meta.get('duration', ''))}",
        f"language: {yaml_str(meta.get('language', ''))}",
        f"transcribed_with: {yaml_str(meta.get('model', ''))}",
        f"summarized_with: {yaml_str(model)}",
    ]
    if prompt_name:
        fm.append(f"prompt: {yaml_str(prompt_name)}")
    fm.extend([
        f"summary_words: {summary_words}",
        f"summarizer: {yaml_str(TOOL)}",
        f"created: {yaml_str(datetime.now(timezone.utc).isoformat(timespec='seconds'))}",
        "tags: [youtube, video-summary]",
        "---",
    ])
    header_meta = []
    if meta.get("channel"):
        header_meta.append(f"**Channel:** {meta['channel']}")
    if meta.get("upload_date"):
        header_meta.append(f"**Published:** {meta['upload_date']}")
    if meta.get("duration"):
        header_meta.append(f"**Duration:** {meta['duration']}")
    parts = ["\n".join(fm), "", f"# {title}", ""]
    if header_meta:
        parts.append(" · ".join(header_meta) + "\n")
    if meta.get("url"):
        parts.append(f"[Watch on YouTube]({meta['url']})\n")
    parts.append(summary)
    parts.append("\n---\n")
    parts.append(f"*Summary generated locally by yt-pipeline {VERSION} with "
                 f"{model} (Ollama), from a transcript by "
                 f"{meta.get('tool', 'yt-transcribe')}. {summary_words} words.*")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# File selection / output
# --------------------------------------------------------------------------- #

def resolve_transcripts(opts, targets):
    """Turn positional args (paths, video ids, URLs) into transcript paths.

    Ids and URLs locate the transcript through its header's video_id, so
    both the current title-named files and the old id-named ones are found."""
    ids = re.compile(r"^[A-Za-z0-9_-]{11}$")
    url_id = re.compile(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})")
    out = []
    for t in targets:
        if os.path.isfile(t):
            out.append(t)
            continue
        m = url_id.search(t) or (ids.match(t) and t)
        vid = m.group(1) if hasattr(m, "group") else m
        if vid:
            hit = find_transcript(vid, opts.transcripts_dirs)
            if hit:
                out.append(str(hit))
            else:
                die(f"no transcript for video {vid} found in "
                    f"{', '.join(opts.transcripts_dirs)}; run "
                    f"'yt-pipeline transcribe {vid}' first "
                    f"(or pass a .txt path directly)")
        else:
            die(f"not a transcript file, video id or YouTube URL: {t}")
    return out


def summarized_video_ids(out_dir):
    """video_ids already present in existing summary notes."""
    done = set()
    if not os.path.isdir(out_dir):
        return done
    for fn in os.listdir(out_dir):
        if not fn.endswith(".md"):
            continue
        try:
            with open(os.path.join(out_dir, fn), encoding="utf-8",
                      errors="replace") as f:
                head = f.read(4096)
        except OSError:
            continue
        m = re.search(r"^video_id:\s*\"?([A-Za-z0-9_-]{11})\"?", head, re.M)
        if m:
            done.add(m.group(1))
    return done


def output_path(out_dir, meta, stem):
    """summaries/<sanitized title>.md, sharing the transcript naming rules
    (see common.titled_path): re-runs of the same video update its own note,
    and a different video with the same sanitized title gets the id
    appended instead of overwriting it."""
    vid = meta.get("video_id", "")
    base = safe_name(meta.get("title") or stem) or (vid or stem)
    return titled_path(out_dir, base, vid, ".md")


# --------------------------------------------------------------------------- #
# Core entry point (used by the 'summarize' and 'run' commands)
# --------------------------------------------------------------------------- #

def summarize_targets(paths, opts, ollama=None):
    """Summarize the given transcript files and write/print the notes.

    opts must have fields: out, model, host, timeout, no_autostart, think,
    max_words, chunk_words, notes_words, max_num_ctx, force, stdout, lang.
    A pre-built, running Ollama client (batch mode) can be passed in; by
    default one is created (and the model checked) here.
    Returns the process exit code (0 ok, 1 if any file failed).
    """
    if ollama is None:
        ollama = Ollama(opts.host, opts.model, opts.timeout,
                        no_autostart=opts.no_autostart)
        ollama.check_model()

    if not isinstance(getattr(opts, "prompts", None), PromptSet):
        opts.prompts = prompts_for(opts)  # resolve once for the whole run

    failures = 0
    for path in paths:
        try:
            note, info = summarize_transcript(ollama, path, opts)
        except SystemExit:
            raise
        except Exception as e:
            warn(f"{path}: summarization failed: {e}")
            failures += 1
            continue
        tag = info["video_id"] or os.path.basename(path)
        if opts.stdout:
            print(note)
            log(f"[{tag}] {info['words']} words in {info['elapsed']:.0f}s "
                f"(printed to stdout)")
        else:
            meta, _ = parse_transcript(open(path, encoding="utf-8",
                                            errors="replace").read())
            with naming_lock:  # same-titled videos must not race the write
                out = output_path(opts.out, meta,
                                  os.path.splitext(os.path.basename(path))[0])
                with open(out, "w", encoding="utf-8") as f:
                    f.write(note + "\n")
            log(f"[{tag}] wrote {out} ({info['words']} words, "
                f"{info['elapsed']:.0f}s)")
    return 1 if failures else 0
