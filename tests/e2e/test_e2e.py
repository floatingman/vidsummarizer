"""End-to-end tests: the real CLI (`python3 -m ytpipeline`) run as a
subprocess, with the external tools replaced by fakes.

The fakes in tests/fakes/ are put first on PATH, so the pipeline's yt-dlp,
whisper-cli and ffmpeg calls hit scripts that serve canned fixture data; the
Ollama HTTP API is served in-process by FakeOllamaServer. No network, no
models, fast and deterministic. The workdir gets a dummy whisper model
seeded so ensure_model() never tries a real download.

Tests that need the real tools/network live in tests/real/ (make test-real).
"""

import json
import unittest
from pathlib import Path

from tests.helpers import (FIXTURES, FAKE_BIN, FakeOllamaServer, fake_env,
                           run_cli, tool_log)

VID = "dQw4w9WgXcQ"
VID2 = "jNQXAC9IVRw"
URL = f"https://www.youtube.com/watch?v={VID}"
FIXTURE_TRANSCRIPT = (FIXTURES / "transcript" / "sample.txt").read_text(
    encoding="utf-8")
EXPECTED_PARAS = [
    "Hello. This is a short test recording. for the yt pipeline test suite.",
    "The quick brown fox jumps over the lazy dog.",
]


class E2ETest(unittest.TestCase):
    """Base: temp workdir, fake-tool env, fake Ollama server."""

    @classmethod
    def setUpClass(cls):
        cls.ollama = FakeOllamaServer()

    @classmethod
    def tearDownClass(cls):
        cls.ollama.stop()

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-e2e-")
        self.addCleanup(self._tmp.cleanup)
        self.workdir = Path(self._tmp.name)
        # dummy model so nothing ever downloads the real 466 MiB file
        self.model = self.workdir / "models" / "ggml-small.bin"
        self.model.parent.mkdir(parents=True)
        self.model.write_bytes(b"fake-ggml-model")
        self.ollama.reset()
        self.env = fake_env(self.workdir)

    # -- helpers ---------------------------------------------------------- #

    def cli(self, *args, timeout=120, env=None):
        env = env or self.env
        return run_cli([str(a) for a in args], env, timeout=timeout)

    def seed_transcript(self, video_id="FxTureVid01", title=None):
        text = FIXTURE_TRANSCRIPT
        if video_id != "FxTureVid01":
            text = text.replace("FxTureVid01", video_id)
        if title:
            text = text.replace("Fixture Talk - Testing Pipelines", title)
        tdir = self.workdir / "transcripts"
        tdir.mkdir(parents=True, exist_ok=True)
        path = tdir / f"{video_id}.txt"
        path.write_text(text, encoding="utf-8")
        return path


class TestBasics(E2ETest):
    def test_version(self):
        r = self.cli("--version")
        self.assertEqual(r.returncode, 0)
        self.assertRegex(r.stdout, r"yt-pipeline \d+\.\d+\.\d+")

    def test_no_args_prints_help(self):
        r = self.cli()
        self.assertEqual(r.returncode, 2)
        self.assertIn("usage:", r.stdout)

    def test_doctor_all_present(self):
        r = self.cli("doctor", "--host", self.ollama.host,
                     "--workdir", self.workdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("[ OK ] yt-dlp", r.stdout)
        self.assertIn("[ OK ] ffmpeg", r.stdout)
        self.assertIn("[ OK ] whisper-cli", r.stdout)
        self.assertIn("[ OK ] ollama", r.stdout)
        self.assertIn("[ OK ] whisper model", r.stdout)  # seeded in workdir
        self.assertIn("[ OK ] Ollama server", r.stdout)
        self.assertIn("everything present", r.stdout)

    def test_doctor_reports_missing_tools(self):
        empty_bin = self.workdir / "empty-bin"
        empty_bin.mkdir()
        env = fake_env(self.workdir, extra_path=empty_bin,
                       no_system_path=True)
        env.pop("YT_FAKE_LOG", None)  # fake tools are not on PATH here
        r = self.cli("doctor", env=env, timeout=60)
        self.assertEqual(r.returncode, 1)
        for tool in ("yt-dlp", "ffmpeg", "whisper-cli", "ollama"):
            self.assertIn(f"[MISS] {tool}", r.stdout)
        self.assertIn("install:", r.stdout)      # install hints
        self.assertIn("missing", r.stdout)

    def test_fetch_requires_tools(self):
        empty_bin = self.workdir / "empty-bin"
        empty_bin.mkdir()
        env = fake_env(self.workdir, extra_path=empty_bin,
                       no_system_path=True)
        env.pop("YT_FAKE_LOG", None)
        r = self.cli("fetch", VID, env=env, timeout=60)
        self.assertEqual(r.returncode, 1)
        self.assertIn("missing required external tool", r.stderr)


class TestFetch(E2ETest):
    def test_fetch_downloads_mp3(self):
        r = self.cli("fetch", VID, "--workdir", self.workdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        mp3 = self.workdir / "audio" / f"{VID}.mp3"
        self.assertTrue(mp3.exists())
        self.assertGreater(mp3.stat().st_size, 0)
        self.assertIn("audio ready", r.stderr)

    def test_audio_cache_is_reused(self):
        self.cli("fetch", VID, "--workdir", self.workdir)
        self.cli("fetch", VID, "--workdir", self.workdir)
        downloads = [c for c in tool_log(self.workdir)
                     if c["tool"] == "yt-dlp" and "-J" not in c["argv"]]
        self.assertEqual(len(downloads), 1)  # second run hit the cache
        self.cli("fetch", VID, "--redownload", "--workdir", self.workdir)
        downloads = [c for c in tool_log(self.workdir)
                     if c["tool"] == "yt-dlp" and "-J" not in c["argv"]]
        self.assertEqual(len(downloads), 2)  # --redownload bypasses cache

    def test_ytdlp_download_failure_is_reported(self):
        env = dict(self.env)
        env["YT_FAKE_YTDLP_FAIL_DOWNLOAD_IDS"] = VID
        r = self.cli("fetch", VID, "-j", "1", "--workdir", self.workdir,
                     env=env)
        self.assertEqual(r.returncode, 1)
        self.assertIn("yt-dlp audio download", r.stderr)


class TestTranscribe(E2ETest):
    def test_transcribe_writes_transcript(self):
        r = self.cli("transcribe", VID, "--workdir", self.workdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = self.workdir / "transcripts" / f"{VID}.txt"
        self.assertTrue(out.exists())
        text = out.read_text(encoding="utf-8")
        self.assertIn("title: Me at the zoo (yt-pipeline fixture)", text)
        self.assertIn("channel: jawed", text)
        self.assertIn("upload_date: 2005-04-23", text)
        self.assertIn(f"video_id: {VID}", text)
        self.assertIn("language: en", text)
        header, body = text.split("\n\n", 1)
        self.assertEqual(body.strip(), "\n\n".join(EXPECTED_PARAS))
        # audio is kept by default after transcription
        self.assertTrue((self.workdir / "audio" / f"{VID}.mp3").exists())

    def test_transcribe_no_keep_audio(self):
        r = self.cli("transcribe", VID, "--no-keep-audio",
                     "--workdir", self.workdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.workdir / "transcripts" / f"{VID}.txt").exists())
        self.assertFalse((self.workdir / "audio" / f"{VID}.mp3").exists())

    def test_transcribe_timestamps(self):
        r = self.cli("transcribe", VID, "--timestamps",
                     "--workdir", self.workdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        text = (self.workdir / "transcripts" / f"{VID}.txt").read_text()
        self.assertIn("[00:00] " + EXPECTED_PARAS[0], text)
        self.assertIn("[00:06] " + EXPECTED_PARAS[1], text)

    def test_transcribe_custom_out(self):
        out = self.workdir / "custom" / "note.txt"
        r = self.cli("transcribe", VID, "-j", "1", "-o", out,
                     "--workdir", self.workdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(out.exists())

    def test_transcribe_wav_fallback(self):
        env = dict(self.env)
        env["YT_FAKE_WHISPER_NEEDS_WAV"] = "1"
        r = self.cli("transcribe", VID, "--workdir", self.workdir, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.workdir / "transcripts" / f"{VID}.txt").exists())
        calls = tool_log(self.workdir)
        kinds = [c["tool"] for c in calls]
        self.assertEqual(kinds.count("ffmpeg"), 1)  # mp3 -> 16k wav
        whisper_inputs = [c["argv"][c["argv"].index("-f") + 1]
                          for c in calls if c["tool"] == "whisper-cli"]
        self.assertEqual(len(whisper_inputs), 2)
        self.assertTrue(whisper_inputs[1].endswith(".16k.wav"))

    def test_transcribe_batch_comma_separated(self):
        r = self.cli("transcribe", f"{VID},{VID2}", "-j", "2",
                     "--workdir", self.workdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        for vid in (VID, VID2):
            self.assertTrue(
                (self.workdir / "transcripts" / f"{vid}.txt").exists())
        self.assertIn("batch finished", r.stderr)

    def test_transcribe_batch_file_with_comments_and_dupes(self):
        batch = self.workdir / "videos.txt"
        batch.write_text(
            "# two videos, one repeated\n"
            f"{VID}\n"
            "\n"
            f"{VID2},{VID}\n", encoding="utf-8")
        r = self.cli("transcribe", "--batch", batch, "--workdir", self.workdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ignoring duplicate video", r.stderr)
        for vid in (VID, VID2):
            self.assertTrue(
                (self.workdir / "transcripts" / f"{vid}.txt").exists())

    def test_transcribe_batch_failure_does_not_kill_the_rest(self):
        env = dict(self.env)
        env["YT_FAKE_YTDLP_FAIL_IDS"] = VID  # first video is "dead"
        r = self.cli("transcribe", f"{VID},{VID2}", "-j", "2",
                     "--workdir", self.workdir, env=env)
        self.assertEqual(r.returncode, 1)
        self.assertIn("1/2 video(s) failed", r.stderr)
        self.assertFalse(
            (self.workdir / "transcripts" / f"{VID}.txt").exists())
        self.assertTrue(
            (self.workdir / "transcripts" / f"{VID2}.txt").exists())

    def test_transcribe_dead_video_fails_cleanly(self):
        env = dict(self.env)
        env["YT_FAKE_YTDLP_FAIL_IDS"] = VID
        r = self.cli("transcribe", VID, "-j", "1",
                     "--workdir", self.workdir, env=env)
        self.assertEqual(r.returncode, 1)
        self.assertIn("yt-dlp metadata lookup", r.stderr)

    def test_bad_video_id_rejected_before_any_work(self):
        r = self.cli("transcribe", "definitely-not-a-video", "--workdir", self.workdir)
        self.assertEqual(r.returncode, 1)
        self.assertIn("11-character YouTube video id", r.stderr)
        self.assertEqual(tool_log(self.workdir), [])  # nothing was invoked

    def test_no_videos_given(self):
        r = self.cli("transcribe", "--workdir", self.workdir)
        self.assertEqual(r.returncode, 1)
        self.assertIn("no videos given", r.stderr)


class TestSummarize(E2ETest):
    def summarize_args(self):
        return ["--host", self.ollama.host, "--no-autostart",
                "--workdir", self.workdir]

    def test_summarize_transcript_by_id(self):
        self.seed_transcript()
        r = self.cli("summarize", "FxTureVid01", *self.summarize_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        note_path = self.workdir / "summaries" / "Fixture Talk - Testing Pipelines.md"
        self.assertTrue(note_path.exists())
        note = note_path.read_text(encoding="utf-8")
        self.assertTrue(note.startswith("---"))
        self.assertIn('title: "Fixture Talk - Testing Pipelines"', note)
        self.assertIn('video_id: "FxTureVid01"', note)
        self.assertIn("tags: [youtube, video-summary]", note)
        self.assertIn("# Fixture Talk - Testing Pipelines", note)
        self.assertIn("[Watch on YouTube](https://www.youtube.com/watch?v=FxTureVid01)",
                      note)
        self.assertIn("## Overview", note)
        # the request carried the transcript and its language
        req = self.ollama.chat_requests[-1]
        self.assertEqual(req["model"], "qwen3.6")
        self.assertIn("The quick brown fox", req["messages"][1]["content"])
        self.assertIn("Write the summary in English.",
                      req["messages"][1]["content"])

    def test_summarize_by_url_and_by_file_path(self):
        self.seed_transcript()
        r = self.cli("summarize", "https://youtu.be/FxTureVid01",
                     *self.summarize_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        r = self.cli("summarize",
                     str(self.workdir / "transcripts" / "FxTureVid01.txt"),
                     "--stdout", *self.summarize_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("## Overview", r.stdout)

    def test_summarize_stdout_writes_no_file(self):
        self.seed_transcript()
        r = self.cli("summarize", "FxTureVid01", "--stdout",
                     *self.summarize_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("## Overview", r.stdout)
        self.assertFalse((self.workdir / "summaries").exists())

    def test_summarize_missing_transcript(self):
        r = self.cli("summarize", "aaaaaaaaaaa", *self.summarize_args())
        self.assertEqual(r.returncode, 1)
        self.assertIn("no transcript", r.stderr)

    def test_summarize_all_skips_already_summarized(self):
        self.seed_transcript("FxTureVid01")
        self.seed_transcript("SecndVid001", title="Second Fixture Talk")
        # pre-existing note for the first video
        out = self.workdir / "summaries"
        out.mkdir()
        (out / "Fixture Talk - Testing Pipelines.md").write_text(
            '---\nvideo_id: "FxTureVid01"\n---\n\n## Overview\n\nold\n',
            encoding="utf-8")
        r = self.cli("summarize", "--all", *self.summarize_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("1 already summarized", r.stderr)
        self.assertEqual(len(self.ollama.chat_requests), 1)  # only the new one
        self.assertTrue((out / "Second Fixture Talk.md").exists())

    def test_summarize_all_redoes_with_force(self):
        self.seed_transcript("FxTureVid01")
        out = self.workdir / "summaries"
        out.mkdir()
        (out / "Fixture Talk - Testing Pipelines.md").write_text(
            '---\nvideo_id: "FxTureVid01"\n---\n\n## Overview\n\nold\n',
            encoding="utf-8")
        r = self.cli("summarize", "--all", "--force", *self.summarize_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(self.ollama.chat_requests), 1)

    def test_summarize_over_limit_rewrite(self):
        self.seed_transcript()
        huge = "## Overview\n\n" + " ".join(f"word{i}" for i in range(800))
        self.ollama.contents = [huge]
        r = self.cli("summarize", "FxTureVid01",
                     "--max-words", "100", *self.summarize_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertGreaterEqual(len(self.ollama.chat_requests), 2)  # rewrite
        note = (self.workdir / "summaries" /
                "Fixture Talk - Testing Pipelines.md").read_text()
        body_words = len(note.split())
        self.assertLessEqual(body_words, 200)  # hard-capped near the limit

    def test_summarize_language_override(self):
        self.seed_transcript()
        r = self.cli("summarize", "FxTureVid01", "--lang", "de",
                     *self.summarize_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Write the summary in German.",
                      self.ollama.chat_requests[-1]["messages"][1]["content"])

    def test_summarize_unreachable_ollama_dies_fast(self):
        self.seed_transcript()
        r = self.cli("summarize", "FxTureVid01",
                     "--host", "http://127.0.0.1:9", "--no-autostart",
                     "--workdir", self.workdir)
        self.assertEqual(r.returncode, 1)
        self.assertIn("cannot reach Ollama", r.stderr)

    def test_summarize_style_selects_prompt_and_its_word_cap(self):
        self.seed_transcript()
        r = self.cli("summarize", "FxTureVid01", "--style", "short",
                     *self.summarize_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        user = self.ollama.chat_requests[-1]["messages"][1]["content"]
        self.assertIn("SHORT OVERVIEW", user)          # the style's prompt
        self.assertIn("at most 150 words", user)       # the style's own cap
        note = (self.workdir / "summaries" /
                "Fixture Talk - Testing Pipelines.md").read_text("utf-8")
        self.assertIn('prompt: "short"', note)         # recorded per note
        self.assertIn("## Overview", note)  # canned reply, Obsidian markdown

    def test_summarize_style_explicit_max_words_wins(self):
        self.seed_transcript()
        r = self.cli("summarize", "FxTureVid01", "--style", "short",
                     "--max-words", "100", *self.summarize_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        user = self.ollama.chat_requests[-1]["messages"][1]["content"]
        self.assertIn("at most 100 words", user)
        self.assertNotIn("at most 150 words", user)

    def test_summarize_unknown_style_lists_the_styles(self):
        self.seed_transcript()
        r = self.cli("summarize", "FxTureVid01", "--style", "nope",
                     *self.summarize_args())
        self.assertEqual(r.returncode, 1)
        self.assertIn("unknown style 'nope'", r.stderr)
        for name in ("actionable", "short", "study-guide"):
            self.assertIn(name, r.stderr)


class TestRunPipeline(E2ETest):
    def run_args(self):
        return ["--host", self.ollama.host, "--no-autostart",
                "--workdir", self.workdir]

    def test_run_full_pipeline(self):
        r = self.cli("run", URL, *self.run_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        transcript = self.workdir / "transcripts" / f"{VID}.txt"
        note = (self.workdir / "summaries" /
                "Me at the zoo (yt-pipeline fixture).md")
        self.assertTrue(transcript.exists())
        self.assertTrue(note.exists())
        self.assertIn(f"video_id: {VID}", transcript.read_text())
        self.assertIn('video_id: "dQw4w9WgXcQ"', note.read_text())
        self.assertEqual(len(self.ollama.chat_requests), 1)

    def test_bare_url_shorthand_runs_full_pipeline(self):
        r = self.cli(URL, *self.run_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.workdir / "transcripts" / f"{VID}.txt").exists())
        self.assertTrue((self.workdir / "summaries").is_dir())

    def test_run_batch_two_videos(self):
        r = self.cli("run", f"{VID},{VID2}", "-j", "2", *self.run_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        for vid in (VID, VID2):
            self.assertTrue(
                (self.workdir / "transcripts" / f"{vid}.txt").exists())
        self.assertEqual(len(self.ollama.chat_requests), 2)
        self.assertIn("batch finished", r.stderr)

    def test_run_summarize_failure_fails_the_run(self):
        # a model tag the (fake) Ollama server does not serve -> check_model
        # dies before any summary is written; the transcript must survive
        self.seed_transcript()
        r = self.cli("run", URL, "-j", "1", "--llm-model", "ghost-model",
                     *self.run_args())
        self.assertEqual(r.returncode, 1)
        self.assertIn("ghost-model", r.stderr)
        self.assertIn("is not installed", r.stderr)
        self.assertTrue((self.workdir / "transcripts" / f"{VID}.txt").exists())
        self.assertFalse((self.workdir / "summaries" /
                          "Me at the zoo (yt-pipeline fixture).md").exists())

    def test_run_with_style(self):
        r = self.cli("run", URL, "--style", "study-guide", *self.run_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        note = (self.workdir / "summaries" /
                "Me at the zoo (yt-pipeline fixture).md")
        self.assertTrue(note.exists())
        self.assertIn('prompt: "study-guide"', note.read_text("utf-8"))
        user = self.ollama.chat_requests[-1]["messages"][1]["content"]
        self.assertIn("STUDY GUIDE", user)         # the style's prompt ...
        self.assertIn("at most 700 words", user)   # ... and its own cap


class TestZipapp(unittest.TestCase):
    """Optional checks of the built single-file executable when present."""

    EXE = FIXTURES.parent.parent / "dist" / "yt-pipeline"

    def test_built_executable_reports_version(self):
        if not self.EXE.exists():
            self.skipTest("dist/yt-pipeline not built (run ./build.sh)")
        import subprocess, sys
        r = subprocess.run([sys.executable, str(self.EXE), "--version"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0)
        self.assertIn("yt-pipeline", r.stdout)

    def test_built_executable_carries_the_styles(self):
        # the styles are prompt variant files packed into the executable:
        # --style must work with no checkout and no prompts dir anywhere
        if not self.EXE.exists():
            self.skipTest("dist/yt-pipeline not built (run ./build.sh)")
        import subprocess
        import sys
        import tempfile
        from tests.helpers import FakeOllamaServer, fake_env
        server = FakeOllamaServer()
        self.addCleanup(server.stop)
        with tempfile.TemporaryDirectory(prefix="ytp-zip-") as tmp:
            tdir = Path(tmp) / "transcripts"
            tdir.mkdir()
            (tdir / "FxTureVid01.txt").write_text(FIXTURE_TRANSCRIPT,
                                                  encoding="utf-8")
            r = subprocess.run(
                [sys.executable, str(self.EXE), "summarize", "FxTureVid01",
                 "--style", "short", "--stdout", "--no-autostart",
                 "--host", server.host, "--workdir", tmp],
                env=fake_env(Path(tmp)), capture_output=True, text=True,
                errors="replace", timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        user = server.chat_requests[-1]["messages"][1]["content"]
        self.assertIn("SHORT OVERVIEW", user)
        self.assertIn("at most 150 words", user)


if __name__ == "__main__":
    unittest.main()
