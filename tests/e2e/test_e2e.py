"""End-to-end tests: the real CLI (`python3 -m vidsummarizer`) run as a
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
                           run_cli, tool_log, REPO)
from vidsummarizer.common import safe_name, video_id_in_file

VID = "dQw4w9WgXcQ"
VID2 = "jNQXAC9IVRw"
URL = f"https://www.youtube.com/watch?v={VID}"
# the fake yt-dlp serves this title for every video unless overridden
FIXTURE_TITLE = "Me at the zoo (vidsummarizer fixture)"
TRANSCRIPT_NAME = f"{safe_name(FIXTURE_TITLE)}.txt"
# distinct title handed to VID2 when a test needs two non-colliding videos
VID2_TITLE = "Second fixture talk"
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

    def titles_env(self, **titles):
        """Env giving specific titles to specific video ids (the fake
        yt-dlp serves the same fixture title to every video otherwise)."""
        env = dict(self.env)
        env["YT_FAKE_YTDLP_TITLES"] = json.dumps(titles)
        return env

    def transcript_for(self, video_id):
        """The transcript file whose header carries video_id (independent of
        the filename, which is title-based now)."""
        tdir = self.workdir / "transcripts"
        for p in sorted(tdir.glob("*.txt")):
            if f"video_id: {video_id}\n" in p.read_text(encoding="utf-8"):
                return p
        return None

    def seed_transcript(self, video_id="FxTureVid01", title=None, legacy=False):
        """Write a transcript into the workdir: title-named by default (the
        pipeline's own naming), or under the old <video_id>.txt name."""
        text = FIXTURE_TRANSCRIPT
        if video_id != "FxTureVid01":
            text = text.replace("FxTureVid01", video_id)
        if title:
            text = text.replace("Fixture Talk - Testing Pipelines", title)
        tdir = self.workdir / "transcripts"
        tdir.mkdir(parents=True, exist_ok=True)
        stem = safe_name(title or "Fixture Talk - Testing Pipelines") \
            if not legacy else video_id
        path = tdir / f"{stem}.txt"
        path.write_text(text, encoding="utf-8")
        return path


class TestBasics(E2ETest):
    def test_version(self):
        r = self.cli("--version")
        self.assertEqual(r.returncode, 0)
        self.assertRegex(r.stdout, r"vidsummarizer \d+\.\d+\.\d+")

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
    def test_transcribe_writes_title_named_transcript(self):
        r = self.cli("transcribe", VID, "--workdir", self.workdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = self.workdir / "transcripts" / TRANSCRIPT_NAME
        self.assertTrue(out.exists())
        text = out.read_text(encoding="utf-8")
        self.assertIn("title: Me at the zoo (vidsummarizer fixture)", text)
        self.assertIn("channel: jawed", text)
        self.assertIn("upload_date: 2005-04-23", text)
        self.assertIn(f"video_id: {VID}", text)  # id stays discoverable
        self.assertIn("language: en", text)
        header, body = text.split("\n\n", 1)
        self.assertEqual(body.strip(), "\n\n".join(EXPECTED_PARAS))
        # audio is kept by default after transcription
        self.assertTrue((self.workdir / "audio" / f"{VID}.mp3").exists())

    def test_transcribe_sanitizes_hostile_title(self):
        hostile = 'What? A "great" video: part/1 *best*'
        env = self.titles_env(**{VID: hostile})
        r = self.cli("transcribe", VID, "--workdir", self.workdir, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = self.workdir / "transcripts" / "What A great video part1 best.txt"
        self.assertTrue(out.exists(), sorted(
            (self.workdir / "transcripts").glob("*.txt")))
        self.assertIn(f"video_id: {VID}", out.read_text(encoding="utf-8"))

    def test_transcribe_caps_runaway_title(self):
        env = self.titles_env(**{VID: "L" * 300})
        r = self.cli("transcribe", VID, "--workdir", self.workdir, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        files = list((self.workdir / "transcripts").glob("*.txt"))
        self.assertEqual(len(files), 1)
        self.assertLessEqual(len(files[0].stem), 90)

    def test_transcribe_title_collision_gets_id_suffix(self):
        # both videos share the fixture title; -j 1 makes the order fixed
        r = self.cli("transcribe", f"{VID},{VID2}", "-j", "1",
                     "--workdir", self.workdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        tdir = self.workdir / "transcripts"
        plain = tdir / TRANSCRIPT_NAME
        suffixed = tdir / f"{safe_name(FIXTURE_TITLE)} ({VID2}).txt"
        self.assertTrue(plain.exists())
        self.assertTrue(suffixed.exists())
        self.assertEqual(len(list(tdir.glob("*.txt"))), 2)  # nothing merged
        self.assertIn(f"video_id: {VID}\n", plain.read_text(encoding="utf-8"))
        self.assertIn(f"video_id: {VID2}\n",
                      suffixed.read_text(encoding="utf-8"))
        # re-running either video updates its own file, never the other's
        r = self.cli("transcribe", VID2, "-j", "1", "--workdir", self.workdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(list(tdir.glob("*.txt"))), 2)
        self.assertIn(f"video_id: {VID}\n", plain.read_text(encoding="utf-8"))
        self.assertTrue(suffixed.exists())

    def test_transcribe_replaces_legacy_id_named_file(self):
        legacy = self.workdir / "transcripts" / f"{VID}.txt"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text(f"video_id: {VID}\n\nstale\n", encoding="utf-8")
        r = self.cli("transcribe", VID, "--workdir", self.workdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.workdir / "transcripts" / TRANSCRIPT_NAME).exists())
        self.assertFalse(legacy.exists())  # superseded, not left stale
        self.assertIn("replaced", r.stderr)

    def test_transcribe_no_keep_audio(self):
        r = self.cli("transcribe", VID, "--no-keep-audio",
                     "--workdir", self.workdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.workdir / "transcripts" / TRANSCRIPT_NAME).exists())
        self.assertFalse((self.workdir / "audio" / f"{VID}.mp3").exists())

    def test_transcribe_timestamps(self):
        r = self.cli("transcribe", VID, "--timestamps",
                     "--workdir", self.workdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        text = (self.workdir / "transcripts" / TRANSCRIPT_NAME).read_text()
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
        self.assertTrue((self.workdir / "transcripts" / TRANSCRIPT_NAME).exists())
        calls = tool_log(self.workdir)
        kinds = [c["tool"] for c in calls]
        self.assertEqual(kinds.count("ffmpeg"), 1)  # mp3 -> 16k wav
        whisper_inputs = [c["argv"][c["argv"].index("-f") + 1]
                          for c in calls if c["tool"] == "whisper-cli"]
        self.assertEqual(len(whisper_inputs), 2)
        self.assertTrue(whisper_inputs[1].endswith(".16k.wav"))

    def test_transcribe_batch_comma_separated(self):
        env = self.titles_env(**{VID2: VID2_TITLE})
        r = self.cli("transcribe", f"{VID},{VID2}", "-j", "2",
                     "--workdir", self.workdir, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.workdir / "transcripts" / TRANSCRIPT_NAME).exists())
        self.assertTrue(
            (self.workdir / "transcripts" / f"{VID2_TITLE}.txt").exists())
        self.assertIn("batch finished", r.stderr)

    def test_transcribe_batch_file_with_comments_and_dupes(self):
        batch = self.workdir / "videos.txt"
        batch.write_text(
            "# two videos, one repeated\n"
            f"{VID}\n"
            "\n"
            f"{VID2},{VID}\n", encoding="utf-8")
        env = self.titles_env(**{VID2: VID2_TITLE})
        r = self.cli("transcribe", "--batch", batch, "--workdir", self.workdir,
                     env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ignoring duplicate video", r.stderr)
        self.assertTrue((self.workdir / "transcripts" / TRANSCRIPT_NAME).exists())
        self.assertTrue(
            (self.workdir / "transcripts" / f"{VID2_TITLE}.txt").exists())

    def test_transcribe_batch_failure_does_not_kill_the_rest(self):
        env = self.titles_env(**{VID2: VID2_TITLE})
        env["YT_FAKE_YTDLP_FAIL_IDS"] = VID  # first video is "dead"
        r = self.cli("transcribe", f"{VID},{VID2}", "-j", "2",
                     "--workdir", self.workdir, env=env)
        self.assertEqual(r.returncode, 1)
        self.assertIn("1/2 video(s) failed", r.stderr)
        self.assertIsNone(self.transcript_for(VID))
        self.assertIsNotNone(self.transcript_for(VID2))

    def test_transcribe_two_processes_same_title_no_loss(self):
        # regression: two vidsummarizer PROCESSES transcribing different videos
        # with the same sanitized title must not silently overwrite one
        # transcript (the exists-check/write pair is guarded by a flock)
        import subprocess
        import sys
        cmd = [sys.executable, "-m", "vidsummarizer", "transcribe"]
        procs = [subprocess.Popen(
            [*cmd, vid, "-j", "1", "--workdir", str(self.workdir)],
            cwd=str(REPO),
            env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace") for vid in (VID, VID2)]
        for p in procs:
            _, err = p.communicate(timeout=120)
            self.assertEqual(p.returncode, 0, err)
        tdir = self.workdir / "transcripts"
        files = sorted(tdir.glob("*.txt"))
        self.assertEqual(len(files), 2, [f.name for f in files])
        owners = {video_id_in_file(f) for f in files}
        self.assertEqual(owners, {VID, VID2})  # both videos survived
        names = {f.name for f in files}
        self.assertIn(TRANSCRIPT_NAME, names)  # one plain title name
        self.assertEqual(
            len([n for n in names if n.startswith(
                safe_name(FIXTURE_TITLE) + " (")]), 1)  # one id-suffixed

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
        self.seed_transcript()  # title-named: "Fixture Talk - Testing Pipelines"
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

    def test_summarize_by_id_finds_legacy_id_named_transcript(self):
        # transcripts written by pre-2.3 versions keep working untouched
        self.seed_transcript(legacy=True)
        r = self.cli("summarize", "FxTureVid01", *self.summarize_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(
            (self.workdir / "summaries" / "Fixture Talk - Testing Pipelines.md").exists())

    def test_summarize_by_url_and_by_file_path(self):
        self.seed_transcript()
        r = self.cli("summarize", "https://youtu.be/FxTureVid01",
                     *self.summarize_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        r = self.cli("summarize",
                     str(self.workdir / "transcripts" /
                         "Fixture Talk - Testing Pipelines.txt"),
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

    def test_summarize_all_dedupes_legacy_and_title_named_copy(self):
        # a video with both an old id-named and a new title-named transcript
        # is summarized once, from the title-named file
        legacy = self.seed_transcript(legacy=True)
        legacy.write_text(
            legacy.read_text(encoding="utf-8").replace(
                "lazy dog", "STALE LEGACY BODY"), encoding="utf-8")
        self.seed_transcript()
        r = self.cli("summarize", "--all", *self.summarize_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(self.ollama.chat_requests), 1)  # once, not twice
        user = self.ollama.chat_requests[-1]["messages"][1]["content"]
        self.assertIn("lazy dog", user)          # the title-named copy won
        self.assertNotIn("STALE LEGACY BODY", user)
        self.assertTrue(
            (self.workdir / "summaries" / "Fixture Talk - Testing Pipelines.md").exists())

    def test_summarize_all_dedupes_headerless_legacy_copy(self):
        # a pre-2.3 <id>.txt without a header is still recognized as the
        # video's file by its name and deduplicated against the
        # title-named copy
        legacy = self.seed_transcript(legacy=True)
        legacy.write_text("STALE HEADERLESS BODY\n", encoding="utf-8")
        self.seed_transcript()
        r = self.cli("summarize", "--all", *self.summarize_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("1 duplicate copy", r.stderr)
        self.assertEqual(len(self.ollama.chat_requests), 1)
        user = self.ollama.chat_requests[-1]["messages"][1]["content"]
        self.assertNotIn("STALE HEADERLESS BODY", user)  # title-named won

    def test_summarize_all_marks_headerless_legacy_done_by_id(self):
        # the note for a headerless legacy transcript carries the id
        # inferred from its filename, so a re-run does not summarize it again
        legacy = self.seed_transcript(legacy=True)
        legacy.write_text("plain body, no header\n", encoding="utf-8")
        r = self.cli("summarize", "--all", *self.summarize_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        note = self.workdir / "summaries" / "FxTureVid01.md"
        self.assertTrue(note.exists())
        self.assertIn('video_id: "FxTureVid01"', note.read_text(encoding="utf-8"))
        r = self.cli("summarize", "--all", *self.summarize_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("1 already summarized", r.stderr)
        self.assertEqual(len(self.ollama.chat_requests), 1)  # only run one


    def test_summarize_all_counts_duplicates_separately_from_done(self):
        # 4 transcripts covering 3 videos, nothing summarized yet: the
        # deduped copy must not be reported as "already summarized"
        self.seed_transcript("FxTureVid01")
        self.seed_transcript("FxTureVid01", legacy=True)
        self.seed_transcript("SecndVid001", title="Second Fixture Talk")
        self.seed_transcript("ThirdVid001", title="Third Fixture Talk")
        r = self.cli("summarize", "--all", *self.summarize_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("found 4 transcripts, 1 duplicate copy, 3 to do", r.stderr)
        self.assertNotIn("already summarized", r.stderr)
        self.assertEqual(len(self.ollama.chat_requests), 3)

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
        transcript = self.workdir / "transcripts" / TRANSCRIPT_NAME
        note = (self.workdir / "summaries" /
                "Me at the zoo (vidsummarizer fixture).md")
        self.assertTrue(transcript.exists())
        self.assertTrue(note.exists())
        self.assertIn(f"video_id: {VID}", transcript.read_text())
        self.assertIn('video_id: "dQw4w9WgXcQ"', note.read_text())
        self.assertEqual(len(self.ollama.chat_requests), 1)

    def test_bare_url_shorthand_runs_full_pipeline(self):
        r = self.cli(URL, *self.run_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.workdir / "transcripts" / TRANSCRIPT_NAME).exists())
        self.assertTrue((self.workdir / "summaries").is_dir())

    def test_run_batch_two_videos(self):
        env = self.titles_env(**{VID2: VID2_TITLE})
        r = self.cli("run", f"{VID},{VID2}", "-j", "2", *self.run_args(),
                     env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.workdir / "transcripts" / TRANSCRIPT_NAME).exists())
        self.assertTrue(
            (self.workdir / "transcripts" / f"{VID2_TITLE}.txt").exists())
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
        self.assertTrue((self.workdir / "transcripts" / TRANSCRIPT_NAME).exists())
        self.assertFalse((self.workdir / "summaries" /
                          "Me at the zoo (vidsummarizer fixture).md").exists())

    def test_run_with_style(self):
        r = self.cli("run", URL, "--style", "study-guide", *self.run_args())
        self.assertEqual(r.returncode, 0, r.stderr)
        note = (self.workdir / "summaries" /
                "Me at the zoo (vidsummarizer fixture).md")
        self.assertTrue(note.exists())
        self.assertIn('prompt: "study-guide"', note.read_text("utf-8"))
        user = self.ollama.chat_requests[-1]["messages"][1]["content"]
        self.assertIn("STUDY GUIDE", user)         # the style's prompt ...
        self.assertIn("at most 700 words", user)   # ... and its own cap


class TestZipapp(unittest.TestCase):
    """Optional checks of the built single-file executable when present."""

    EXE = FIXTURES.parent.parent / "dist" / "vidsummarizer"

    def test_built_executable_reports_version(self):
        if not self.EXE.exists():
            self.skipTest("dist/vidsummarizer not built (run ./build.sh)")
        import subprocess, sys
        r = subprocess.run([sys.executable, str(self.EXE), "--version"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0)
        self.assertIn("vidsummarizer", r.stdout)

    def test_built_executable_carries_the_styles(self):
        # the styles are prompt variant files packed into the executable:
        # --style must work with no checkout and no prompts dir anywhere
        if not self.EXE.exists():
            self.skipTest("dist/vidsummarizer not built (run ./build.sh)")
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
