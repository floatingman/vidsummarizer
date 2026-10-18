"""Unit tests for ytpipeline.summarize (Ollama stage): transcript parsing,
chunking, markdown cleanup, note building, the Ollama HTTP client (against
an in-process fake server), and file selection/output logic."""

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import (FakeOllamaServer, assert_dies, capture, ns,
                           stderr_of)

from ytpipeline import summarize
from ytpipeline.summarize import (Ollama, build_note, chunk_text,
                                  clean_markdown, count_words, gen_options,
                                  output_path, parse_transcript,
                                  resolve_transcripts, safe_name, sentences,
                                  summarize_targets, summarize_transcript,
                                  summarized_video_ids, trim_to_word_cap,
                                  yaml_str)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE = (FIXTURES / "transcript" / "sample.txt").read_text(encoding="utf-8")
REPO_PROMPTS = Path(__file__).resolve().parent.parent.parent / "prompts"

# A duck-typed Ollama client returning canned chat responses.
STUB_NOTE = ("## Overview\n\nThis fixture summary mentions the lazy dog and "
             "the number 42.")


class StubOllama:
    model = "stub-model"

    def __init__(self, responses=(STUB_NOTE,)):
        self.responses = list(responses)
        self.calls = []

    def chat(self, system, user, options, think=False, retries=3):
        self.calls.append({"system": system, "user": user,
                           "options": options, "think": think})
        if len(self.responses) > 1:
            content = self.responses.pop(0)
        else:
            content = self.responses[0]
        return content, {"s": 0.5, "prompt_tok": 10, "gen_tok": 10,
                         "tok_s": 20.0}


def opts(**over):
    base = dict(out=None, model="stub-model", host="http://unused",
                timeout=30, no_autostart=True, think=False, max_words=500,
                lang=None, style=None, chunk_words=1500, notes_words=220,
                max_num_ctx=32768, force=False, stdout=False,
                prompts_dir=None, workdir=None)
    base.update(over)
    return ns(**base)


# A minimal prompt variant that IS a style (used by the style tests):
# distinctive markers so tests can tell its prompts from any other's.
MINI_STYLE = {
    "name": "mini",
    "version": 1,
    "description": "minimal test style",
    "created": "2026-10-16",
    "style": {"label": "Mini", "summary": "mini style for tests",
              "max_words": 42, "fallback_heading": "## Mini summary"},
    "prompts": {
        "system": "MINI SYSTEM.",
        "final": "MINI STYLE FINAL. {lang_instr} Write at most "
                 "{max_words} words.\n\n{transcript}",
        "notes": "MINI NOTES {i}/{n} {lang_instr} at most {notes_words} "
                 "words.\n\n{chunk}",
        "reduce": "MINI REDUCE {lang_instr} at most {max_words} "
                  "words.\n\n{notes}",
        "condense": "MINI CONDENSE {actual} -> {max_words} words.\n\n{text}",
    },
}


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #

class TestTextHelpers(unittest.TestCase):
    def test_count_words(self):
        self.assertEqual(count_words(""), 0)
        self.assertEqual(count_words("one two  three\n"), 3)
        self.assertEqual(count_words("## Heading"), 2)  # markdown counts

    def test_sentences(self):
        self.assertEqual(sentences("One. Two! Three?"),
                         ["One.", "Two!", "Three?"])
        self.assertEqual(sentences("  "), [])
        self.assertEqual(sentences("no terminal punctuation"),
                         ["no terminal punctuation"])

    def test_sentences_cjk_no_whitespace_not_split(self):
        # Known limitation: the sentence splitter needs whitespace after the
        # punctuation, so CJK text without spaces stays one "sentence".
        parts = sentences("第一句。第二句！")
        self.assertEqual(parts, ["第一句。第二句！"])
        parts = sentences("第一句。 第二句！")  # with spaces it works
        self.assertEqual(parts, ["第一句。", "第二句！"])

    def test_chunk_text_empty(self):
        self.assertEqual(chunk_text("   ", 100), [])

    def test_chunk_text_single_chunk(self):
        self.assertEqual(chunk_text("One. Two.", 100), ["One. Two."])

    def test_chunk_text_sentence_aligned(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(20))
        chunks = chunk_text(text, 30)  # ~5 words per sentence -> ~6 sentences/chunk
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c.split()), 30 + 6)  # last sentence may
            # slightly overshoot (a sentence is never split)
            self.assertTrue(c.endswith("."))
        # round-trip preserves all sentences
        joined = " ".join(chunks)
        self.assertEqual(len(sentences(joined)), 20)

    def test_safe_name_strips_forbidden_chars(self):
        self.assertEqual(safe_name('a/b\\c:d*e?f"g<h>i|j#k^l[m]n'),
                         "abcdefghijklmn")

    def test_safe_name_collapses_whitespace_and_strips_dots(self):
        self.assertEqual(safe_name("  A   B .. "), "A B")

    def test_safe_name_max_len(self):
        self.assertEqual(len(safe_name("x" * 200, max_len=90)), 90)

    def test_safe_name_byte_cap(self):
        # 90 CJK chars would be 270 UTF-8 bytes: the byte cap must trim
        # further so the filename stays within OS limits even with the
        # " (<id>)" suffix added
        name = safe_name("\u4e2d" * 200)  # 3-byte chars
        self.assertLessEqual(len(name.encode("utf-8")), 200)
        self.assertLess(len(name), 90)

    def test_yaml_str_is_json_double_quoted(self):
        self.assertEqual(yaml_str('he said "hi"'), '"he said \\"hi\\""')
        self.assertEqual(yaml_str("line\nbreak"), '"line\\nbreak"')
        self.assertEqual(yaml_str("héllo"), '"héllo"')  # not \\u escaped

    def test_normalize_host(self):
        self.assertEqual(summarize.normalize_host(None),
                         "http://127.0.0.1:11434")
        self.assertEqual(summarize.normalize_host("127.0.0.1:11434"),
                         "http://127.0.0.1:11434")
        self.assertEqual(summarize.normalize_host("http://h//"),
                         "http://h")


# --------------------------------------------------------------------------- #
# transcript parsing
# --------------------------------------------------------------------------- #

class TestParseTranscript(unittest.TestCase):
    def test_full_header(self):
        meta, body = parse_transcript(SAMPLE)
        self.assertEqual(meta["title"], "Fixture Talk - Testing Pipelines")
        self.assertEqual(meta["video_id"], "FxTureVid01")
        self.assertEqual(meta["language"], "en")
        self.assertEqual(meta["url"],
                         "https://www.youtube.com/watch?v=FxTureVid01")
        self.assertTrue(body.startswith("Hello."))
        self.assertNotIn("title:", body)

    def test_header_value_may_contain_colons(self):
        meta, _ = parse_transcript("url: https://x/y?a=b\n\nbody")
        self.assertEqual(meta["url"], "https://x/y?a=b")

    def test_plain_text_has_no_meta(self):
        meta, body = parse_transcript("Just words.\n\nMore words.")
        self.assertEqual(meta, {})
        self.assertTrue(body.startswith("Just words."))

    def test_key_colon_lines_need_blank_line_after(self):
        # "title: x" NOT followed by a blank line is body text, not a header
        meta, body = parse_transcript("title: x\nstill the body")
        self.assertEqual(meta, {})
        self.assertEqual(body, "title: x\nstill the body")

    def test_header_only_file_gives_empty_body(self):
        meta, body = parse_transcript("title: x\nvideo_id: abcdefghijk")
        self.assertEqual(meta["title"], "x")
        self.assertEqual(body, "")

    def test_timestamps_prefix_is_part_of_body(self):
        text = "title: x\n\n[00:00] first paragraph.\n\n[00:12] second."
        _, body = parse_transcript(text)
        self.assertIn("[00:12]", body)


class TestLanguageInstruction(unittest.TestCase):
    def test_known_language(self):
        instr = summarize.transcript_language_instruction(
            {"language": "en"}, "body")
        self.assertEqual(instr, "Write the summary in English.")

    def test_case_insensitive(self):
        instr = summarize.transcript_language_instruction(
            {"language": "DE"}, "body")
        self.assertEqual(instr, "Write the summary in German.")

    def test_unknown_code(self):
        instr = summarize.transcript_language_instruction(
            {"language": "xx"}, "body")
        self.assertIn("'xx'", instr)

    def test_no_language(self):
        instr = summarize.transcript_language_instruction({}, "body")
        self.assertEqual(instr,
                         "Write the summary in the same language as the "
                         "transcript.")

    def test_opts_override(self):
        from ytpipeline.summarize import language_instruction
        instr = language_instruction(opts(lang="de"), {}, "body")
        self.assertIn("German", instr)
        instr = language_instruction(opts(lang="Klingon"), {}, "body")
        self.assertIn("Klingon", instr)
        instr = language_instruction(opts(lang=None), {"language": "fr"},
                                     "body")
        self.assertIn("French", instr)


# --------------------------------------------------------------------------- #
# prompting / options / cleanup
# --------------------------------------------------------------------------- #

class TestGenOptions(unittest.TestCase):
    def test_floor_and_ceiling(self):
        o = gen_options(0, 10, opts(), 0.3)
        self.assertEqual(o["num_ctx"], 2048)
        o = gen_options(10 ** 9, 10 ** 9, opts(), 0.3)
        self.assertEqual(o["num_ctx"], 32768)

    def test_scales_with_prompt(self):
        small = gen_options(100, 100, opts(), 0.3)
        big = gen_options(5000, 500, opts(), 0.3)
        self.assertGreater(big["num_ctx"], small["num_ctx"])

    def test_sampling_defaults(self):
        o = gen_options(100, 100, opts(), 0.3)
        self.assertEqual(o["temperature"], 0.3)
        self.assertEqual(o["presence_penalty"], 0.0)


class TestCleanMarkdown(unittest.TestCase):
    def test_keeps_from_first_heading(self):
        out = clean_markdown("preamble\n\n## Overview\n\nbody")
        self.assertTrue(out.startswith("## Overview"))
        self.assertNotIn("preamble", out)

    def test_code_fence_stripped(self):
        out = clean_markdown("```markdown\n## Overview\n\nbody\n```")
        self.assertTrue(out.startswith("## Overview"))
        self.assertNotIn("```", out)

    def test_missing_heading_gets_overview(self):
        self.assertTrue(clean_markdown("just text").startswith("## Overview"))

    def test_whitespace_collapsed(self):
        out = clean_markdown("## Overview\n\n\n\ntext\n")
        self.assertNotIn("\n\n\n", out)

    def test_later_headings_kept(self):
        out = clean_markdown("## Overview\n\na\n\n### Actionable steps\n\n1. x")
        self.assertIn("### Actionable steps", out)


class TestTrimToWordCap(unittest.TestCase):
    def test_under_cap_untouched(self):
        text = "## Overview\n\nOne two three."
        self.assertEqual(trim_to_word_cap(text, 10), text)

    def test_stops_at_cap_keeps_headings(self):
        text = ("## Overview\n\n"
                + " ".join(f"Sentence {i} with several words." for i in range(30)))
        out = trim_to_word_cap(text, 15)
        self.assertTrue(out.startswith("## Overview"))
        self.assertLessEqual(len(out.split()), 15 + 3)  # heading words +
        # at most one overshooting sentence is dropped, not cut mid-way

    def test_warns_when_trimming(self):
        text = "## Overview\n\n" + " ".join(f"Sentence {i} here." for i in range(50))
        err, _ = stderr_of(lambda: trim_to_word_cap(text, 10))
        self.assertIn("hard-trimmed", err)


# --------------------------------------------------------------------------- #
# note building
# --------------------------------------------------------------------------- #

class TestBuildNote(unittest.TestCase):
    META = {"title": "Fixture Talk - Testing Pipelines",
            "channel": "Fixture Channel", "upload_date": "2024-01-02",
            "duration": "01:01", "video_id": "FxTureVid01",
            "url": "https://www.youtube.com/watch?v=FxTureVid01",
            "language": "en", "model": "ggml-small.bin",
            "tool": "yt-pipeline 2.1.0 (local whisper.cpp)"}

    def test_note_structure(self):
        note = build_note(self.META["title"], self.META, STUB_NOTE, 12,
                          "qwen3.6")
        lines = note.splitlines()
        self.assertEqual(lines[0], "---")
        self.assertIn('title: "Fixture Talk - Testing Pipelines"', lines)
        self.assertIn('video_id: "FxTureVid01"', lines)
        self.assertIn("summary_words: 12", lines)
        self.assertIn("tags: [youtube, video-summary]", lines)
        self.assertEqual(lines[lines.index("---", 1) + 1], "")  # blank after FM
        self.assertIn("# Fixture Talk - Testing Pipelines", lines)
        # header meta is one dot-separated line
        meta_line = [ln for ln in lines if ln.startswith("**Channel:**")][0]
        self.assertIn("**Published:** 2024-01-02", meta_line)
        self.assertIn("**Duration:** 01:01", meta_line)
        self.assertIn("[Watch on YouTube](https://www.youtube.com/watch?v=FxTureVid01)",
                      lines)
        self.assertIn("## Overview", lines)
        self.assertIn("generated locally by yt-pipeline", note)
        self.assertIn("qwen3.6", note)

    def test_missing_meta_lines_omitted(self):
        note = build_note("T", {}, STUB_NOTE, 5, "m")
        self.assertNotIn("**Channel:**", note)
        self.assertNotIn("**Published:**", note)
        self.assertNotIn("[Watch on YouTube]", note)


# --------------------------------------------------------------------------- #
# file selection / output
# --------------------------------------------------------------------------- #

class TestResolveTranscripts(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-rt-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.tdir = self.dir / "transcripts"
        self.tdir.mkdir()
        self.o = ns(transcripts_dirs=[str(self.tdir)])

    def seed(self, name, video_id=None, title=None):
        text = ""
        if title:
            text += f"title: {title}\n"
        if video_id:
            text += f"video_id: {video_id}\n"
        if text:
            text += "\n"
        path = self.tdir / name
        path.write_text(text + "body", encoding="utf-8")
        return path

    def test_existing_file_passthrough(self):
        p = self.seed("abcdefghijk.txt", video_id="abcdefghijk")
        self.assertEqual(resolve_transcripts(self.o, [str(p)]), [str(p)])

    def test_bare_id_resolves_title_named_transcript(self):
        p = self.seed("A Great Talk.txt", video_id="abcdefghijk",
                      title="A Great Talk")
        self.assertEqual(resolve_transcripts(self.o, ["abcdefghijk"]),
                         [str(p)])

    def test_bare_id_resolves_id_suffixed_transcript(self):
        p = self.seed("A Great Talk (abcdefghijk).txt", video_id="abcdefghijk")
        self.assertEqual(resolve_transcripts(self.o, ["abcdefghijk"]),
                         [str(p)])

    def test_bare_id_resolves_legacy_id_named_transcript(self):
        # transcripts written by older versions keep working untouched
        p = self.seed("abcdefghijk.txt", video_id="abcdefghijk")
        self.assertEqual(resolve_transcripts(self.o, ["abcdefghijk"]),
                         [str(p)])

    def test_legacy_headerless_transcript_resolved_by_name(self):
        p = self.seed("abcdefghijk.txt")
        self.assertEqual(resolve_transcripts(self.o, ["abcdefghijk"]),
                         [str(p)])

    def test_bare_id_skips_legacy_name_with_foreign_header(self):
        # a file named <id>.txt whose header names another video is that
        # video's title-named output (its title just looks like an id)
        self.seed("abcdefghijk.txt", video_id="zzzzzzzzzzz")
        assert_dies(self, lambda: resolve_transcripts(self.o, ["abcdefghijk"]),
                    pattern="no transcript")


    def test_ensure_video_id_infers_legacy_id_from_the_filename(self):
        meta = {}
        self.assertEqual(summarize.ensure_video_id(meta, "d/abcdefghijk.txt"),
                         "abcdefghijk")
        self.assertEqual(meta["video_id"], "abcdefghijk")  # note + bookkeeping
        meta = {"video_id": "zzzzzzzzzzz"}
        self.assertEqual(summarize.ensure_video_id(meta, "d/abcdefghijk.txt"),
                         "zzzzzzzzzzz")  # the header id always wins
        meta = {}
        self.assertEqual(summarize.ensure_video_id(meta, "d/My Talk.txt"), "")
        self.assertNotIn("video_id", meta)


    def test_title_named_wins_over_legacy_duplicate(self):
        self.seed("abcdefghijk.txt", video_id="abcdefghijk")
        titled = self.seed("A Great Talk.txt", video_id="abcdefghijk")
        self.assertEqual(resolve_transcripts(self.o, ["abcdefghijk"]),
                         [str(titled)])

    def test_url_resolved(self):
        p = self.seed("A Great Talk.txt", video_id="abcdefghijk")
        self.assertEqual(
            resolve_transcripts(self.o, ["https://youtu.be/abcdefghijk"]),
            [str(p)])

    def test_missing_transcript_dies(self):
        assert_dies(self, lambda: resolve_transcripts(self.o, ["zzzzzzzzzzz"]),
                    pattern="no transcript")

    def test_junk_target_dies(self):
        assert_dies(self, lambda: resolve_transcripts(self.o, ["nonsense/x"]),
                    pattern="not a transcript file, video id or YouTube URL")


class TestSummarizedVideoIds(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-svi-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_collects_ids_from_frontmatter(self):
        (self.dir / "Note one.md").write_text(
            '---\ntitle: "One"\nvideo_id: "abcdefghijk"\n---\n\nbody')
        (self.dir / "Note two.md").write_text(
            '---\nvideo_id: lmnopqrstuv\n---\n')
        (self.dir / "not-a-note.txt").write_text("video_id: ignored")
        self.assertEqual(summarized_video_ids(str(self.dir)),
                         {"abcdefghijk", "lmnopqrstuv"})

    def test_missing_dir_is_empty(self):
        self.assertEqual(summarized_video_ids(str(self.dir / "nope")), set())


class TestOutputPath(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-op-")
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name) / "summaries"

    def test_title_sanitized(self):
        p = output_path(str(self.out), {"title": 'a/b: c?'}, "stem")
        self.assertEqual(p, self.out / "ab c.md")  # forbidden chars
        # removed, real whitespace kept

    def test_stem_used_when_no_title(self):
        p = output_path(str(self.out), {}, "the stem")
        self.assertEqual(p, self.out / "the stem.md")

    def test_video_id_used_when_no_title_or_stem(self):
        p = output_path(str(self.out), {"video_id": "abcdefghijk"}, "")
        self.assertEqual(p, self.out / "abcdefghijk.md")

    def test_same_name_different_video_disambiguated(self):
        existing = self.out / "Talk.md"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text('---\nvideo_id: "abcdefghijk"\n---\n')
        p = output_path(str(self.out), {"title": "Talk",
                                        "video_id": "lmnopqrstuv"}, "x")
        self.assertEqual(p, self.out / "Talk (lmnopqrstuv).md")

    def test_same_video_overwrites(self):
        existing = self.out / "Talk.md"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text('---\nvideo_id: "abcdefghijk"\n---\n')
        p = output_path(str(self.out), {"title": "Talk",
                                        "video_id": "abcdefghijk"}, "x")
        self.assertEqual(p, existing)

    def test_runaway_title_capped(self):
        p = output_path(str(self.out), {"title": "L" * 300,
                                        "video_id": "abcdefghijk"}, "x")
        self.assertEqual(len(p.stem), 90)


# --------------------------------------------------------------------------- #
# summarize_transcript (with a stub Ollama)
# --------------------------------------------------------------------------- #

class TestSummarizeTranscript(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-st-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "FxTureVid01.txt"
        self.path.write_text(SAMPLE, encoding="utf-8")

    def test_single_pass(self):
        stub = StubOllama()
        (note, info), _ = capture(
            lambda: summarize_transcript(stub, str(self.path), opts()))
        self.assertEqual(len(stub.calls), 1)
        call = stub.calls[0]
        self.assertIn("transcript", call["user"].lower())
        self.assertIn("The quick brown fox", call["user"])
        self.assertIn("Write the summary in English.", call["user"])
        self.assertIn("Fixture Talk - Testing Pipelines", call["user"])  # context
        self.assertFalse(call["think"])
        self.assertEqual(info["video_id"], "FxTureVid01")
        self.assertEqual(info["chunks"], 1)
        self.assertIn("## Overview", note)
        self.assertIn("lazy dog", note)
        self.assertIn('title: "Fixture Talk - Testing Pipelines"', note)

    def test_chunked_map_reduce(self):
        # body is ~45 words; force chunking with a tiny chunk size
        stub = StubOllama()
        (note, info), _ = capture(
            lambda: summarize_transcript(stub, str(self.path),
                                         opts(chunk_words=10)))
        self.assertGreater(info["chunks"], 1)
        self.assertEqual(len(stub.calls), info["chunks"] + 1)  # notes + reduce
        self.assertIn("part 1 of", stub.calls[0]["user"])
        self.assertIn("notes", stub.calls[-1]["user"].lower())
        self.assertIn("## Overview", note)

    def test_chunk_info_records_chunk_count(self):
        stub = StubOllama()
        (_, info), _ = capture(
            lambda: summarize_transcript(stub, str(self.path),
                                         opts(chunk_words=10)))
        self.assertGreaterEqual(info["chunks"], 2)

    def test_never_chunk_when_disabled(self):
        stub = StubOllama()
        (note, info), _ = capture(
            lambda: summarize_transcript(stub, str(self.path),
                                         opts(chunk_words=0)))
        self.assertEqual(len(stub.calls), 1)
        self.assertEqual(info["chunks"], 1)

    def test_over_limit_triggers_rewrite_then_trim(self):
        huge = "## Overview\n\n" + " ".join(f"word{i}" for i in range(600))
        still_huge = "## Overview\n\n" + " ".join(f"w{i}" for i in range(600))
        stub = StubOllama(responses=[huge, still_huge])
        (note, _), _ = capture(
            lambda: summarize_transcript(stub, str(self.path), opts()))
        self.assertEqual(len(stub.calls), 2)  # draft + condense
        self.assertRegex(stub.calls[1]["user"], r"\d+ words")  # rewrite prompt
        # hard trim keeps it near the cap (headings are kept for free)
        self.assertLessEqual(count_words(note), 500 + 5)

    def test_over_limit_single_rewrite_suffices(self):
        huge = "## Overview\n\n" + " ".join(f"word{i}" for i in range(600))
        stub = StubOllama(responses=[huge, STUB_NOTE])
        (note, _), _ = capture(
            lambda: summarize_transcript(stub, str(self.path), opts()))
        self.assertEqual(len(stub.calls), 2)
        self.assertIn("lazy dog", note)

    def test_empty_transcript_dies(self):
        p = self.dir / "empty.txt"
        p.write_text("title: x\n\n")
        assert_dies(self,
                    lambda: summarize_transcript(StubOllama(), str(p), opts()),
                    pattern="transcript is empty")


class TestSummarizeTargets(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-tgt-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.out = self.dir / "summaries"
        for name in ("FxTureVid01", "SecndVid001"):
            src = SAMPLE
            if name == "SecndVid001":
                src = (SAMPLE.replace("FxTureVid01", name)
                       .replace("Fixture Talk - Testing Pipelines",
                                "Second Fixture Talk"))
            (self.dir / f"{name}.txt").write_text(src, encoding="utf-8")

    def test_writes_all_notes(self):
        stub = StubOllama()
        rc = summarize_targets([str(self.dir / "FxTureVid01.txt"),
                                str(self.dir / "SecndVid001.txt")],
                               opts(out=str(self.out)), ollama=stub)
        self.assertEqual(rc, 0)
        self.assertTrue((self.out / "Fixture Talk - Testing Pipelines.md").exists())
        self.assertTrue((self.out / "Second Fixture Talk.md").exists())

    def test_stdout_prints_instead_of_writing(self):
        import contextlib, io
        stub = StubOllama()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = summarize_targets([str(self.dir / "FxTureVid01.txt")],
                                   opts(stdout=True), ollama=stub)
        self.assertEqual(rc, 0)
        self.assertIn("## Overview", buf.getvalue())
        self.assertFalse(self.out.exists())

    def test_failure_of_one_does_not_kill_the_rest(self):
        stub = StubOllama()
        with mock.patch("ytpipeline.summarize.summarize_transcript",
                        side_effect=[RuntimeError("boom"),
                                     (STUB_NOTE, {"video_id": "SecndVid001",
                                                  "words": 10, "elapsed": 0.1,
                                                  "chunks": 1})]):
            rc = summarize_targets([str(self.dir / "FxTureVid01.txt"),
                                    str(self.dir / "SecndVid001.txt")],
                                   opts(out=str(self.out)), ollama=stub)
        self.assertEqual(rc, 1)
        self.assertTrue((self.out / "Second Fixture Talk.md").exists())


# --------------------------------------------------------------------------- #
# summary styles (--style): per-run prompt-variant selection + the style's
# own output expectations (word cap, fallback heading)
# --------------------------------------------------------------------------- #

class TestStyles(unittest.TestCase):
    def setUp(self):
        import shutil
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-style-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        from ytpipeline.prompts import write_variant
        write_variant(self.dir, MINI_STYLE)
        # a plain variant so the default (active) path also works in this dir
        shutil.copyfile(REPO_PROMPTS / "v1-baseline.json",
                        self.dir / "v1-baseline.json")
        self.path = self.dir / "FxTureVid01.txt"
        self.path.write_text(SAMPLE, encoding="utf-8")

    def test_style_selects_its_prompts(self):
        stub = StubOllama()
        (note, info), _ = capture(lambda: summarize_transcript(
            stub, str(self.path), opts(style="mini", prompts_dir=str(self.dir))))
        self.assertEqual(info["prompt"], "mini")
        self.assertEqual(stub.calls[0]["system"], "MINI SYSTEM.")
        self.assertIn("MINI STYLE FINAL.", stub.calls[0]["user"])
        self.assertIn("The quick brown fox", stub.calls[0]["user"])
        self.assertIn('prompt: "mini"', note)  # recorded in the frontmatter

    def test_style_word_cap_applies_when_max_words_unset(self):
        stub = StubOllama()
        o = opts(style="mini", prompts_dir=str(self.dir), max_words=None)
        _, _ = capture(lambda: summarize_transcript(stub, str(self.path), o))
        self.assertIn("Write at most 42 words.", stub.calls[0]["user"])
        self.assertEqual(o.max_words, 42)  # resolved for the whole call

    def test_explicit_max_words_beats_style_cap(self):
        stub = StubOllama()
        o = opts(style="mini", prompts_dir=str(self.dir), max_words=500)
        _, _ = capture(lambda: summarize_transcript(stub, str(self.path), o))
        self.assertIn("Write at most 500 words.", stub.calls[0]["user"])
        self.assertEqual(o.max_words, 500)

    def test_default_without_style_stays_the_active_variant(self):
        stub = StubOllama()
        (note, info), _ = capture(lambda: summarize_transcript(
            stub, str(self.path), opts(prompts_dir=str(self.dir))))
        self.assertEqual(info["prompt"], "v1-baseline")
        self.assertIn("Decide what kind of video it is",
                      stub.calls[0]["user"])  # baseline final prompt

    def test_style_fallback_heading_when_model_omits_heading(self):
        stub = StubOllama(responses=["Just text, no heading at all."])
        (note, info), _ = capture(lambda: summarize_transcript(
            stub, str(self.path), opts(style="mini",
                                       prompts_dir=str(self.dir))))
        self.assertIn("## Mini summary", note)  # not the default Overview
        self.assertNotIn("## Overview", note)

    def test_style_cap_used_in_chunked_reduce_pass_too(self):
        stub = StubOllama()
        o = opts(style="mini", prompts_dir=str(self.dir), max_words=None,
                 chunk_words=10)
        ((note, info), _) = capture(lambda: summarize_transcript(
            stub, str(self.path), o))
        self.assertGreater(info["chunks"], 1)
        self.assertIn("MINI REDUCE", stub.calls[-1]["user"])
        self.assertIn("at most 42 words", stub.calls[-1]["user"])

    def test_unknown_style_dies_listing_available_styles(self):
        err = assert_dies(
            self,
            lambda: summarize_transcript(
                StubOllama(), str(self.path),
                opts(style="nope", prompts_dir=str(self.dir))),
            pattern="unknown style 'nope'")
        self.assertIn("styles: mini (Mini)", err)
        self.assertIn("v1-baseline", err)  # plain variants listed too

    def test_pinned_prompts_beat_style_selection(self):
        from ytpipeline.prompts import load_variant
        pinned = load_variant(self.dir, "v1-baseline")
        stub = StubOllama()
        ((note, info), _) = capture(lambda: summarize_transcript(
            stub, str(self.path),
            opts(prompts=pinned, style="mini", prompts_dir=str(self.dir))))
        self.assertEqual(info["prompt"], "v1-baseline")
        self.assertIn("Decide what kind of video it is", stub.calls[0]["user"])

    def test_style_accepts_plain_variant_names_too(self):
        # --style takes any variant name, not only ones with style metadata
        stub = StubOllama()
        ((note, info), _) = capture(lambda: summarize_transcript(
            stub, str(self.path),
            opts(style="v1-baseline", prompts_dir=str(self.dir))))
        self.assertEqual(info["prompt"], "v1-baseline")


# --------------------------------------------------------------------------- #
# the real Ollama client against a fake HTTP server
# --------------------------------------------------------------------------- #

class TestOllamaClient(unittest.TestCase):
    def setUp(self):
        self.server = FakeOllamaServer()
        self.addCleanup(self.server.stop)

    def client(self, **kw):
        kw.setdefault("host", self.server.host)
        kw.setdefault("model", "qwen3.6")
        kw.setdefault("timeout", 10)
        return Ollama(**kw)

    def test_ping_ok_on_init(self):
        c = self.client()
        self.assertEqual(c.host, self.server.host)

    def test_unreachable_with_no_autostart_dies(self):
        assert_dies(self,
                    lambda: Ollama("http://127.0.0.1:9", "m", 1,
                                   no_autostart=True),
                    pattern="cannot reach Ollama")

    def test_autostart_spawns_serve_and_waits(self):
        pings = iter([False, False, True])
        with mock.patch.object(Ollama, "_ping", side_effect=lambda: next(pings)), \
             mock.patch("ytpipeline.summarize.subprocess.Popen") as popen, \
             mock.patch("ytpipeline.summarize.time.sleep") as sleep:
            Ollama("http://127.0.0.1:9", "m", 1)
        popen.assert_called_once()
        self.assertEqual(popen.call_args[0][0], ["ollama", "serve"])
        self.assertEqual(sleep.call_count, 2)

    def test_check_model_exact_and_prefix_match(self):
        self.server.models = ["qwen3.6:latest"]
        self.client().check_model()  # prefix match: no exception
        self.server.models = ["qwen3.6"]
        self.client().check_model()

    def test_check_model_missing_dies_with_pull_hint(self):
        self.server.models = ["llama3:latest"]
        assert_dies(self, lambda: self.client(model="qwen3.6").check_model(),
                    pattern=r"ollama pull qwen3.6")

    def test_chat_round_trip(self):
        c = self.client()
        content, stats = c.chat("sys", "user text", {"num_ctx": 2048})
        self.assertEqual(content, self.server.default_content)
        self.assertGreater(stats["gen_tok"], 0)
        self.assertGreaterEqual(stats["tok_s"], 0)  # 0 for sub-0.5s replies
        req = self.server.chat_requests[-1]
        self.assertEqual(req["model"], "qwen3.6")
        self.assertEqual(req["stream"], False)
        self.assertFalse(req["think"])
        self.assertEqual(req["messages"][0]["role"], "system")
        self.assertEqual(req["messages"][1]["content"], "user text")

    def test_chat_think_flag(self):
        c = self.client()
        c.chat("s", "u", {}, think=True)
        self.assertTrue(self.server.chat_requests[-1]["think"])

    def test_chat_think_fallback_when_model_rejects_it(self):
        self.server.fail_queue.append(
            {"status": 400, "body": {"error": "model does not support think"}})
        with mock.patch("ytpipeline.summarize.time.sleep"):
            content, _ = self.client().chat("s", "u", {}, think=True)
        self.assertEqual(len(self.server.chat_requests), 2)
        self.assertTrue(self.server.chat_requests[0]["think"])
        self.assertNotIn("think", self.server.chat_requests[1])  # dropped
        self.assertEqual(content, self.server.default_content)

    def test_chat_retries_transient_errors(self):
        self.server.fail_queue.extend([{"status": 500, "body": {"e": 1}},
                                       {"status": 500, "body": {"e": 2}}])
        with mock.patch("ytpipeline.summarize.time.sleep") as sleep:
            content, _ = self.client().chat("s", "u", {})
        self.assertEqual(len(self.server.chat_requests), 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(content, self.server.default_content)

    def test_chat_empty_content_retried(self):
        self.server.contents = ["", "real answer"]
        with mock.patch("ytpipeline.summarize.time.sleep"):
            content, _ = self.client().chat("s", "u", {})
        self.assertEqual(len(self.server.chat_requests), 2)
        self.assertEqual(content, "real answer")

    def test_chat_gives_up_after_retries(self):
        self.server.contents = [""]  # always empty
        with mock.patch("ytpipeline.summarize.time.sleep"):
            assert_dies(self, lambda: self.client().chat("s", "u", {}),
                        pattern="failed after 3 attempts")


if __name__ == "__main__":
    unittest.main()
