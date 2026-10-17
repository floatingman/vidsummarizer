"""Unit tests for ytpipeline.pipeline: thread resolution, batch option
plumbing and the command implementations with their collaborators stubbed."""

import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import assert_dies, ns

from ytpipeline import cli, pipeline

VID = "dQw4w9WgXcQ"
VID2 = "jNQXAC9IVRw"


def parse(argv):
    return cli.build_parser().parse_args(argv)


class CleanEnvMixin(unittest.TestCase):
    """Drop env vars that change pipeline behavior (tests stay deterministic
    regardless of the developer's shell)."""

    def setUp(self):
        import os
        env = {k: v for k, v in os.environ.items()
               if k not in ("YT_WORKDIR", "YT_TRANSCRIPTS_DIR", "YT_JOBS",
                            "OLLAMA_HOST", "OLLAMA_MODEL")}
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestResolveThreads(unittest.TestCase):
    def test_single_job_default(self):
        self.assertEqual(pipeline.resolve_threads(ns(threads=None), 1), 8)

    def test_split_across_jobs(self):
        self.assertEqual(pipeline.resolve_threads(ns(threads=None), 2), 4)
        self.assertEqual(pipeline.resolve_threads(ns(threads=None), 3), 2)
        self.assertEqual(pipeline.resolve_threads(ns(threads=None), 8), 2)

    def test_explicit_threads_win(self):
        self.assertEqual(pipeline.resolve_threads(ns(threads=5), 3), 5)


class TestBatchIntro(unittest.TestCase):
    def test_mentions_count_and_jobs(self):
        msg = pipeline._batch_intro("processing", ["a", "b"], 2, threads=4)
        self.assertIn("2 video(s)", msg)
        self.assertIn("2 at a time", msg)
        self.assertIn("4 whisper thread(s)", msg)


class TestCmdFetch(CleanEnvMixin):
    def setUp(self):
        super().setUp()
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-cf-")
        self.addCleanup(self._tmp.cleanup)
        self.workdir = Path(self._tmp.name)
        self.mp3 = self.workdir / "audio" / f"{VID}.mp3"
        self.mp3.parent.mkdir(parents=True)
        self.mp3.write_bytes(b"x" * 1024)

    def test_single_video(self):
        args = parse(["fetch", VID, "-j", "1",
                      "--workdir", str(self.workdir)])
        with mock.patch("ytpipeline.pipeline.fetch_video_audio",
                        return_value=({"title": "T"}, self.mp3)) as fva:
            rc = pipeline.cmd_fetch(args)
        self.assertEqual(rc, 0)
        fva.assert_called_once()
        self.assertEqual(fva.call_args[0][0].video, VID)

    def test_batch_uses_pool(self):
        args = parse(["fetch", f"{VID},{VID2}", "-j", "2",
                      "--workdir", str(self.workdir)])
        with mock.patch("ytpipeline.pipeline.require_tools") as rt, \
             mock.patch("ytpipeline.pipeline.run_batch",
                        return_value=({}, [])) as rb:
            rc = pipeline.cmd_fetch(args)
        self.assertEqual(rc, 0)
        rt.assert_called_once()
        rb.assert_called_once()
        videos = rb.call_args[0][0]
        self.assertEqual(videos, [VID, VID2])


class TestCmdTranscribe(CleanEnvMixin):
    def setUp(self):
        super().setUp()
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-ct-")
        self.addCleanup(self._tmp.cleanup)
        self.workdir = Path(self._tmp.name)
        self.transcript = self.workdir / "transcripts" / f"{VID}.txt"
        self.transcript.parent.mkdir(parents=True)
        self.transcript.write_text("title: T\n\nbody")

    def test_single_video(self):
        args = parse(["transcribe", VID, "-j", "1",
                      "--workdir", str(self.workdir)])
        with mock.patch("ytpipeline.pipeline.transcribe_video",
                        return_value=self.transcript) as tv:
            rc = pipeline.cmd_transcribe(args)
        self.assertEqual(rc, 0)
        t = tv.call_args[0][0]
        self.assertEqual(t.video, VID)
        self.assertEqual(t.threads, 8)          # single job: full budget
        self.assertTrue(t.keep_audio)

    def test_single_video_with_out(self):
        args = parse(["transcribe", VID, "-o", "custom.txt", "-j", "1",
                      "--workdir", str(self.workdir)])
        with mock.patch("ytpipeline.pipeline.transcribe_video",
                        return_value=Path("custom.txt")) as tv:
            pipeline.cmd_transcribe(args)
        self.assertEqual(tv.call_args[0][0].out, "custom.txt")

    def test_out_with_several_videos_dies(self):
        args = parse(["transcribe", f"{VID},{VID2}", "-o", "x.txt",
                      "--workdir", str(self.workdir)])
        with mock.patch("ytpipeline.pipeline.require_tools"), \
                mock.patch("ytpipeline.pipeline.ensure_model"):
            assert_dies(self, lambda: pipeline.cmd_transcribe(args),
                        pattern="drop it for batch runs")

    def test_batch_splits_threads_and_preflights_model(self):
        args = parse(["transcribe", f"{VID},{VID2}", "-j", "2",
                      "--workdir", str(self.workdir)])
        model = self.workdir / "models" / "ggml-small.bin"
        with mock.patch("ytpipeline.pipeline.require_tools") as rt, \
             mock.patch("ytpipeline.pipeline.ensure_model") as em, \
             mock.patch("ytpipeline.pipeline.run_batch",
                        return_value=({}, [])) as rb:
            rc = pipeline.cmd_transcribe(args)
        self.assertEqual(rc, 0)
        rt.assert_called_once_with(pipeline.TRANSCRIBE_TOOLS)
        em.assert_called_once()
        fn = rb.call_args[0][1]
        # running one worker through the stubbed transcribe shows thread split
        with mock.patch("ytpipeline.pipeline.transcribe_video",
                        return_value=self.transcript) as tv:
            fn(VID)
        self.assertEqual(tv.call_args[0][0].threads, 4)  # 8 // 2 jobs

    def test_batch_failure_gives_nonzero_exit(self):
        args = parse(["transcribe", f"{VID},{VID2}", "-j", "2",
                      "--workdir", str(self.workdir)])
        with mock.patch("ytpipeline.pipeline.require_tools"), \
             mock.patch("ytpipeline.pipeline.ensure_model"), \
             mock.patch("ytpipeline.pipeline.run_batch",
                        return_value=({}, [(VID, RuntimeError("x"))])):
            rc = pipeline.cmd_transcribe(args)
        self.assertEqual(rc, 1)


class TestCmdRun(CleanEnvMixin):
    def setUp(self):
        super().setUp()
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-cr-")
        self.addCleanup(self._tmp.cleanup)
        self.workdir = Path(self._tmp.name)
        self.transcript = self.workdir / "transcripts" / f"{VID}.txt"
        self.transcript.parent.mkdir(parents=True)
        self.transcript.write_text("title: T\n\nbody", encoding="utf-8")

    def test_single_video_chains_transcribe_then_summarize(self):
        args = parse(["run", VID, "-j", "1", "--workdir", str(self.workdir)])
        with mock.patch("ytpipeline.pipeline.transcribe_video",
                        return_value=self.transcript) as tv, \
             mock.patch("ytpipeline.pipeline.summarize_targets",
                        return_value=0) as st:
            rc = pipeline.cmd_run(args)
        self.assertEqual(rc, 0)
        tv.assert_called_once()
        st.assert_called_once()
        self.assertEqual(st.call_args[0][0], [str(self.transcript)])
        summarize_opts = st.call_args[0][1]
        self.assertEqual(summarize_opts.model, "qwen3.6")
        self.assertEqual(summarize_opts.out,
                         str(self.workdir / "summaries"))

    def test_single_video_summarize_failure_propagates(self):
        args = parse(["run", VID, "-j", "1", "--workdir", str(self.workdir)])
        with mock.patch("ytpipeline.pipeline.transcribe_video",
                        return_value=self.transcript), \
             mock.patch("ytpipeline.pipeline.summarize_targets",
                        return_value=1):
            rc = pipeline.cmd_run(args)
        self.assertEqual(rc, 1)

    def test_batch_preflights_ollama_once(self):
        args = parse(["run", f"{VID},{VID2}", "-j", "2",
                      "--workdir", str(self.workdir)])
        with mock.patch("ytpipeline.pipeline.require_tools"), \
             mock.patch("ytpipeline.pipeline.ensure_model"), \
             mock.patch("ytpipeline.pipeline.Ollama") as ollama_cls, \
             mock.patch("ytpipeline.pipeline.run_batch",
                        return_value=({}, [])) as rb:
            rc = pipeline.cmd_run(args)
        self.assertEqual(rc, 0)
        ollama_cls.assert_called_once()
        ollama_cls.return_value.check_model.assert_called_once()
        rb.assert_called_once()

    def test_batch_soft_failures_count(self):
        args = parse(["run", f"{VID},{VID2}", "-j", "2",
                      "--workdir", str(self.workdir)])
        with mock.patch("ytpipeline.pipeline.require_tools"), \
             mock.patch("ytpipeline.pipeline.ensure_model"), \
             mock.patch("ytpipeline.pipeline.Ollama"), \
             mock.patch("ytpipeline.pipeline.run_batch",
                        return_value=({VID: 1, VID2: 0}, [])):
            rc = pipeline.cmd_run(args)
        self.assertEqual(rc, 1)

    def test_stdout_with_several_videos_forces_serial(self):
        args = parse(["run", f"{VID},{VID2}", "--stdout",
                      "--workdir", str(self.workdir)])
        with mock.patch("ytpipeline.pipeline.require_tools"), \
             mock.patch("ytpipeline.pipeline.ensure_model"), \
             mock.patch("ytpipeline.pipeline.Ollama"), \
             mock.patch("ytpipeline.pipeline.run_batch",
                        return_value=({}, [])) as rb:
            pipeline.cmd_run(args)
        self.assertEqual(rb.call_args[0][2], 1)  # jobs=1

    def test_transcript_out_with_several_videos_dies(self):
        args = parse(["run", f"{VID},{VID2}", "--transcript-out", "x.txt",
                      "--workdir", str(self.workdir)])
        with mock.patch("ytpipeline.pipeline.require_tools"), \
                mock.patch("ytpipeline.pipeline.ensure_model"):
            assert_dies(self, lambda: pipeline.cmd_run(args),
                        pattern="drop it for batch runs")


if __name__ == "__main__":
    unittest.main()
