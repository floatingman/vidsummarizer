"""Unit tests for vidsummarizer.transcribe (yt-dlp + whisper.cpp stage).

External processes (yt-dlp, whisper-cli, ffmpeg, model download) are stubbed
with mocks; the parsing/formatting logic is tested against fixture-shaped
data.
"""

import json
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import assert_dies, ns, stderr_of

from vidsummarizer import transcribe
from vidsummarizer.transcribe import (build_output, build_paragraphs,
                                   normalize_segments)

VID = "dQw4w9WgXcQ"
URL = f"https://www.youtube.com/watch?v={VID}"


def seg(start, end, text):
    return {"start": start, "end": end, "text": text}


class TestYtdlpBaseArgs(unittest.TestCase):
    def test_deno_present(self):
        with mock.patch("shutil.which", lambda n, path=None:
                        "/usr/bin/deno" if n == "deno" else None):
            self.assertEqual(transcribe.ytdlp_base_args(), [])

    def test_node_fallback(self):
        with mock.patch("shutil.which", lambda n, path=None:
                        "/usr/bin/node" if n == "node" else None):
            self.assertEqual(transcribe.ytdlp_base_args(),
                             ["--js-runtimes", "node"])

    def test_neither(self):
        with mock.patch("shutil.which", lambda n, path=None: None):
            self.assertEqual(transcribe.ytdlp_base_args(), [])


class TestFetchMetadata(unittest.TestCase):
    def test_parses_json_and_passes_url(self):
        proc = ns(returncode=0, stdout=json.dumps({"id": VID, "title": "T"}),
                  stderr="")
        with mock.patch("vidsummarizer.transcribe.run", return_value=proc) as m:
            meta = transcribe.fetch_metadata(VID, [])
        self.assertEqual(meta["id"], VID)
        cmd = m.call_args[0][0]
        self.assertEqual(cmd[0], "yt-dlp")
        self.assertIn("-J", cmd)
        self.assertEqual(cmd[-1], URL)

    def test_extra_args_included(self):
        proc = ns(returncode=0, stdout="{}", stderr="")
        with mock.patch("vidsummarizer.transcribe.run", return_value=proc) as m:
            transcribe.fetch_metadata(VID, ["--cookies", "c.txt"])
        cmd = m.call_args[0][0]
        self.assertIn("--cookies", cmd)

    def test_bad_json_dies(self):
        proc = ns(returncode=0, stdout="not json", stderr="")
        with mock.patch("vidsummarizer.transcribe.run", return_value=proc):
            assert_dies(self, lambda: transcribe.fetch_metadata(VID, []),
                        pattern="could not be parsed as JSON")


class TestDownloadAudio(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-dl-")
        self.addCleanup(self._tmp.cleanup)
        self.audio_dir = Path(self._tmp.name) / "audio"
        self.audio_dir.mkdir()
        self.mp3 = self.audio_dir / f"{VID}.mp3"

    def _ok_run(self, cmd, *a, **kw):
        # fake yt-dlp: write the mp3 where the -o template points
        tmpl = cmd[cmd.index("-o") + 1]
        Path(tmpl).parent.mkdir(parents=True, exist_ok=True)
        out = Path(tmpl).parent / f"{VID}.mp3"
        out.write_bytes(b"fake mp3")
        return ns(returncode=0, stdout="", stderr="")

    def test_existing_mp3_is_reused_without_ytdlp(self):
        self.mp3.write_bytes(b"cached")
        with mock.patch("vidsummarizer.transcribe.run") as m:
            path = transcribe.download_audio(VID, self.audio_dir, [], False)
        self.assertEqual(path, self.mp3)
        m.assert_not_called()

    def test_downloads_when_missing(self):
        with mock.patch("vidsummarizer.transcribe.run",
                        side_effect=self._ok_run) as m:
            path = transcribe.download_audio(VID, self.audio_dir, [], False)
        self.assertEqual(path, self.mp3)
        self.assertTrue(self.mp3.exists())
        cmd = m.call_args[0][0]
        self.assertEqual(cmd[0], "yt-dlp")
        self.assertIn("--audio-format", cmd)
        self.assertIn(URL, cmd)

    def test_redownload_ignores_cache(self):
        self.mp3.write_bytes(b"cached")
        with mock.patch("vidsummarizer.transcribe.run",
                        side_effect=self._ok_run) as m:
            transcribe.download_audio(VID, self.audio_dir, [], True)
        m.assert_called_once()

    def test_transient_failures_are_retried(self):
        attempts = []

        def flaky(cmd, *a, **kw):
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("transient 403")
            return self._ok_run(cmd)

        sleeps = []
        with mock.patch("vidsummarizer.transcribe.run", flaky), \
             mock.patch("vidsummarizer.transcribe.time.sleep", sleeps.append):
            path = transcribe.download_audio(VID, self.audio_dir, [], False)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleeps, [8, 8])
        self.assertEqual(path, self.mp3)

    def test_persistent_failures_abort(self):
        with mock.patch("vidsummarizer.transcribe.run",
                        side_effect=RuntimeError("down for good")), \
             mock.patch("vidsummarizer.transcribe.time.sleep"):
            assert_dies(self,
                        lambda: transcribe.download_audio(VID, self.audio_dir,
                                                          [], False),
                        code=1)

    def test_missing_output_file_dies(self):
        # fake yt-dlp "succeeds" but produces nothing
        with mock.patch("vidsummarizer.transcribe.run",
                        return_value=ns(returncode=0, stdout="", stderr="")):
            assert_dies(self,
                        lambda: transcribe.download_audio(VID, self.audio_dir,
                                                          [], False),
                        pattern="was not created")


class TestEnsureModel(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-model-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.model = self.dir / "models" / "ggml-small.bin"

    def test_existing_model_untouched(self):
        self.model.parent.mkdir(parents=True)
        self.model.write_bytes(b"model-bytes")
        with mock.patch("urllib.request.urlretrieve") as m:
            path = transcribe.ensure_model(self.model)
        self.assertEqual(path, self.model)
        m.assert_not_called()

    def test_missing_model_downloaded_atomically(self):
        def fake_urlretrieve(url, dest, reporthook=None):
            self.assertIn("huggingface.co", url)
            self.assertTrue(str(dest).endswith(".bin.part"))
            Path(dest).write_bytes(b"downloaded-model")

        with mock.patch("urllib.request.urlretrieve", fake_urlretrieve):
            path = transcribe.ensure_model(self.model)
        self.assertEqual(path, self.model)
        self.assertEqual(self.model.read_bytes(), b"downloaded-model")
        self.assertFalse(self.model.with_suffix(".bin.part").exists())

    def test_custom_model_url(self):
        seen = []

        def fake_urlretrieve(url, dest, reporthook=None):
            seen.append(url)
            Path(dest).write_bytes(b"x")

        with mock.patch("urllib.request.urlretrieve", fake_urlretrieve):
            transcribe.ensure_model(self.model, "http://mirror.example/m.bin")
        self.assertEqual(seen, ["http://mirror.example/m.bin"])

    def test_download_failure_cleans_up_and_dies(self):
        def boom(url, dest, reporthook=None):
            Path(dest).write_bytes(b"partial")
            raise OSError("network down")

        with mock.patch("urllib.request.urlretrieve", boom):
            assert_dies(self, lambda: transcribe.ensure_model(self.model),
                        pattern="model download failed")
        self.assertFalse(self.model.with_suffix(".bin.part").exists())


class TestRunWhisper(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-wh-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.model = self.dir / "ggml-small.bin"
        self.model.write_bytes(b"m")
        self.mp3 = self.dir / "video.mp3"
        self.mp3.write_bytes(b"audio")
        self.calls = []
        self.fail_first_mp3 = False

    def _run(self, cmd, what, tail=15, fatal=True):
        self.calls.append(cmd)
        if cmd[0] == "whisper-cli":
            media = cmd[cmd.index("-f") + 1]
            if self.fail_first_mp3 and not media.endswith(".wav"):
                # emulate a whisper-cli that cannot decode mp3 directly
                raise RuntimeError(
                    "whisper-cli did not produce expected output")
            outbase = cmd[cmd.index("-of") + 1]
            Path(outbase).with_suffix(".json").write_text(
                json.dumps({"result": {"language": "en",
                                       "segments": [seg(0, 1, " hi")]}}))
            return ns(returncode=0, stdout="", stderr="")
        if cmd[0] == "ffmpeg":
            wav = self.dir / "video.16k.wav"
            wav.write_bytes(b"wav")
            return ns(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command {cmd}")

    def test_mp3_direct_happy_path(self):
        self.fail_first_mp3 = False
        with mock.patch("vidsummarizer.transcribe.run", side_effect=self._run):
            result = transcribe.run_whisper(self.model, self.mp3, self.dir,
                                            "auto", 4)
        self.assertEqual(len(self.calls), 1)  # no ffmpeg needed
        cmd = self.calls[0]
        self.assertEqual(cmd[0], "whisper-cli")
        self.assertIn("-oj", cmd)
        self.assertEqual(result["result"]["language"], "en")
        # the intermediate whisper json is removed
        self.assertFalse((self.dir / "video.json").exists())

    def test_falls_back_to_wav_conversion(self):
        self.fail_first_mp3 = True
        with mock.patch("vidsummarizer.transcribe.run", side_effect=self._run):
            result = transcribe.run_whisper(self.model, self.mp3, self.dir,
                                            "auto", 4)
        self.assertEqual([c[0] for c in self.calls],
                         ["whisper-cli", "ffmpeg", "whisper-cli"])
        ffmpeg_cmd = self.calls[1]
        self.assertIn("16000", ffmpeg_cmd)      # sample rate
        self.assertIn("pcm_s16le", ffmpeg_cmd)  # 16-bit mono PCM
        self.assertTrue((self.dir / "video.16k.wav").exists())
        self.assertEqual(result["result"]["segments"][0]["text"], " hi")


class TestNormalizeSegments(unittest.TestCase):
    def test_legacy_schema(self):
        result = {"result": {"segments": [seg(0.0, 1.0, "a"), seg(1.0, 2.0, "b")]}}
        out = normalize_segments(result)
        self.assertEqual([s["text"] for s in out], ["a", "b"])
        self.assertEqual(out[0]["start"], 0.0)

    def test_new_unified_schema_offsets_in_ms(self):
        result = {"transcription": [
            {"offsets": {"from": 0, "to": 1500}, "text": " one."},
            {"offsets": {"from": 1500, "to": 3000}, "text": " two."},
        ]}
        out = normalize_segments(result)
        self.assertEqual([s["text"] for s in out], [" one.", " two."])
        self.assertEqual(out[0]["end"], 1.5)
        self.assertEqual(out[1]["start"], 1.5)

    def test_new_schema_wins_over_legacy(self):
        result = {"transcription": [{"offsets": {"from": 0, "to": 10},
                                     "text": "new"}],
                  "result": {"language": "en"}}
        out = normalize_segments(result)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "new")

    def test_empty_result(self):
        self.assertEqual(normalize_segments({}), [])
        self.assertEqual(normalize_segments({"result": {}}), [])
        self.assertEqual(normalize_segments({"result": {"segments": []}}), [])


class TestBuildParagraphs(unittest.TestCase):
    def test_gap_starts_new_paragraph(self):
        paras = build_paragraphs([seg(0, 2, "first"), seg(6, 8, "second")], False)
        self.assertEqual(paras, ["first", "second"])

    def test_short_gap_joins(self):
        paras = build_paragraphs([seg(0, 2, "first"), seg(2.2, 4, "second")],
                                 False)
        self.assertEqual(paras, ["first second"])

    def test_gap_threshold_is_inclusive(self):
        # gap of exactly PARAGRAPH_GAP_SECONDS (1.5) splits
        paras = build_paragraphs([seg(0, 2, "first"), seg(3.5, 5, "second")],
                                 False)
        self.assertEqual(paras, ["first", "second"])

    def test_whitespace_collapsed(self):
        paras = build_paragraphs([seg(0, 1, "  hello   world \n")], False)
        self.assertEqual(paras, ["hello world"])

    def test_empty_segments_skipped(self):
        paras = build_paragraphs([seg(0, 1, ""), seg(1, 2, "   "),
                                  seg(2, 3, "kept")], False)
        self.assertEqual(paras, ["kept"])

    def test_timestamps_prefix(self):
        paras = build_paragraphs([seg(0, 2, "first"), seg(61, 63, "second")],
                                 True)
        self.assertEqual(paras, ["[00:00] first", "[01:01] second"])

    def _words(self, n, end="."):
        words = [f"w{i}" for i in range(n - 1)] + [f"end{end}"]
        return " ".join(words)

    def test_soft_cap_splits_after_sentence(self):
        # 3 x 100 words, last one ends a sentence: paragraph is 300 words
        # (>= 280), so the next segment starts a new paragraph.
        segs = [seg(0, 10, self._words(100)),
                seg(10.2, 20, self._words(100)),
                seg(20.4, 30, self._words(100)),
                seg(30.6, 40, "next paragraph")]
        paras = build_paragraphs(segs, False)
        self.assertEqual(len(paras), 2)
        self.assertEqual(len(paras[0].split()), 300)
        self.assertEqual(paras[1], "next paragraph")

    def test_soft_cap_waits_for_sentence_end(self):
        # same word count but the paragraph never ends a sentence -> no split
        # at the soft cap; the hard cap (500) splits instead.
        segs = [seg(i * 10.2, i * 10.2 + 10, self._words(100, end=""))
                for i in range(6)]
        paras = build_paragraphs(segs, False)
        self.assertEqual(len(paras), 2)
        self.assertEqual(len(paras[0].split()), 500)  # hard cap
        self.assertEqual(len(paras[1].split()), 100)


class TestBuildOutput(unittest.TestCase):
    def test_full_header_and_paragraphs(self):
        meta = {"title": "  A Talk  ", "channel": "Chan", "uploader": "Up",
                "upload_date": "20240102", "duration": 61}
        model = Path("/models/ggml-small.bin")
        out = build_output(meta, VID, "en", model, ["para one", "para two"])
        lines = out.splitlines()
        self.assertEqual(lines[0], "title: A Talk")
        self.assertIn("channel: Chan", lines)
        self.assertIn("upload_date: 2024-01-02", lines)  # reformatted
        self.assertIn("duration: 01:01", lines)
        self.assertIn(f"video_id: {VID}", lines)
        self.assertIn(f"url: {URL}", lines)
        self.assertIn("language: en", lines)
        self.assertIn("model: ggml-small.bin", lines)
        self.assertTrue(any(ln.startswith("tool: vidsummarizer") for ln in lines))
        # blank line between header and body, paragraphs separated by blank
        header, body = out.split("\n\n", 1)
        self.assertFalse(header.startswith("\n"))
        self.assertTrue(body.startswith("para one"))
        self.assertTrue(out.endswith("\n"))

    def test_empty_fields_are_dropped(self):
        meta = {"title": "T", "duration": None}
        out = build_output(meta, VID, "", Path("/m/ggml-small.bin"), ["p"])
        self.assertIn("title: T", out)
        self.assertNotIn("channel:", out)
        self.assertNotIn("upload_date:", out)
        self.assertNotIn("duration:", out)
        self.assertNotIn("language:", out)

    def test_channel_falls_back_to_uploader(self):
        meta = {"title": "T", "uploader": "Up"}
        out = build_output(meta, VID, "en", Path("/m/ggml-small.bin"), ["p"])
        self.assertIn("channel: Up", out)

    def test_upload_date_passed_through_when_not_8_digits(self):
        meta = {"title": "T", "upload_date": "2024-01-02"}
        out = build_output(meta, VID, "en", Path("/m/ggml-small.bin"), ["p"])
        self.assertIn("upload_date: 2024-01-02", out)


class TestTranscribeVideo(unittest.TestCase):
    """transcribe_video() with every external interaction stubbed."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-tv-")
        self.addCleanup(self._tmp.cleanup)
        self.workdir = Path(self._tmp.name)
        self.audio = self.workdir / "audio" / f"{VID}.mp3"
        self.audio.parent.mkdir(parents=True)
        self.audio.write_bytes(b"mp3")
        self.model = self.workdir / "models" / "ggml-small.bin"
        self.model.parent.mkdir(parents=True)
        self.model.write_bytes(b"model")
        self.whisper_json = json.loads(
            (Path(__file__).resolve().parent.parent / "fixtures" / "whisper"
             / "sample.json").read_text())

    def opts(self, **over):
        base = dict(video=URL, out=None, audio_dir=None, keep_audio=True,
                    model=None, model_url=None, lang="auto", threads=4,
                    timestamps=False, redownload=False, ytdlp="")
        base.update(over)
        return ns(**base)

    def _stub(self, **over):
        rt = over.pop("run_whisper", self.whisper_json)
        patches = [
            mock.patch("vidsummarizer.transcribe.require_tools"),
            mock.patch("vidsummarizer.transcribe.fetch_metadata",
                       return_value={"title": "T", "channel": "C",
                                     "upload_date": "20240102",
                                     "duration": 61}),
            mock.patch("vidsummarizer.transcribe.download_audio",
                       return_value=self.audio),
            mock.patch("vidsummarizer.transcribe.ensure_model",
                       return_value=self.model),
            mock.patch("vidsummarizer.transcribe.run_whisper", return_value=rt),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_writes_transcript_and_keeps_audio_by_default(self):
        self._stub()
        out = transcribe.transcribe_video(self.opts(), self.workdir)
        self.assertEqual(out, self.workdir / "transcripts" / f"{VID}.txt")
        text = out.read_text()
        self.assertIn("title: T", text)
        self.assertIn(f"video_id: {VID}", text)
        self.assertIn("language: en", text)  # detected by whisper
        self.assertIn("The quick brown fox", text)
        self.assertTrue(self.audio.exists())  # kept

    def test_no_keep_audio_deletes_mp3(self):
        self._stub()
        transcribe.transcribe_video(self.opts(keep_audio=False), self.workdir)
        self.assertFalse(self.audio.exists())

    def test_explicit_out_path(self):
        self._stub()
        out = transcribe.transcribe_video(
            self.opts(out=str(self.workdir / "x.txt")), self.workdir)
        self.assertEqual(out, self.workdir / "x.txt")
        self.assertTrue(out.exists())

    def test_undetected_language_omitted(self):
        self._stub(run_whisper={"result": {"language": "",
                                           "segments": [seg(0, 1, "hi")]}})
        out = transcribe.transcribe_video(self.opts(), self.workdir)
        self.assertNotIn("language:", out.read_text())

    def test_explicit_lang_used_when_not_detected(self):
        self._stub(run_whisper={"result": {"language": "",
                                           "segments": [seg(0, 1, "hi")]}})
        out = transcribe.transcribe_video(self.opts(lang="de"), self.workdir)
        self.assertIn("language: de", out.read_text())

    def test_no_segments_dies(self):
        self._stub(run_whisper={"result": {"language": "en", "segments": []}})
        assert_dies(self,
                    lambda: transcribe.transcribe_video(self.opts(),
                                                        self.workdir),
                    pattern="no segments")


if __name__ == "__main__":
    unittest.main()
