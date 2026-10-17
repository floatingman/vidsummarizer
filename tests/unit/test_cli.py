"""Unit tests for ytpipeline.cli: parser defaults, environment handling,
subcommand dispatch and the bare-URL shorthand."""

import os
import unittest
from unittest import mock

from tests.helpers import stderr_of

from ytpipeline import VERSION, cli


def parse(argv):
    return cli.build_parser().parse_args(argv)


class TestParserDefaults(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for var in ("YT_JOBS", "OLLAMA_MODEL", "OLLAMA_HOST"):
            os.environ.pop(var, None)

    def test_run_defaults(self):
        a = parse(["run", "dQw4w9WgXcQ"])
        self.assertEqual(a.command, "run")
        self.assertEqual(a.videos, ["dQw4w9WgXcQ"])
        self.assertEqual(a.jobs, 2)
        self.assertEqual(a.lang, "auto")
        self.assertTrue(a.keep_audio)          # default --keep-audio
        self.assertFalse(parse(["run", "x", "--no-keep-audio"]).keep_audio)
        self.assertIsNone(a.threads)
        self.assertEqual(a.llm_model, "qwen3.6")
        self.assertEqual(a.host, "http://127.0.0.1:11434")
        self.assertIsNone(a.max_words)  # resolved later: 500 or the style's
        self.assertEqual(a.chunk_words, 1500)
        self.assertIsNone(a.summary_lang)
        self.assertIsNone(a.style)

    def test_transcribe_defaults_and_out_flag(self):
        a = parse(["transcribe", "dQw4w9WgXcQ"])
        self.assertEqual(a.command, "transcribe")
        self.assertEqual(a.out, None)
        a = parse(["transcribe", "dQw4w9WgXcQ", "-o", "/tmp/t.txt"])
        self.assertEqual(a.out, "/tmp/t.txt")

    def test_run_uses_transcript_out_flag(self):
        a = parse(["run", "dQw4w9WgXcQ", "--transcript-out", "/tmp/t.txt"])
        self.assertEqual(a.transcript_out, "/tmp/t.txt")

    def test_summarize_defaults(self):
        a = parse(["summarize", "dQw4w9WgXcQ"])
        self.assertEqual(a.model, "qwen3.6")
        self.assertFalse(a.all)
        self.assertIsNone(a.max_words)  # resolved later: 500 or the style's
        self.assertFalse(a.force)
        self.assertIsNone(a.style)

    def test_style_option(self):
        a = parse(["summarize", "x", "--style", "short"])
        self.assertEqual(a.style, "short")
        self.assertIsNone(a.max_words)
        a = parse(["summarize", "x", "--style", "study-guide",
                   "--max-words", "300"])
        self.assertEqual(a.style, "study-guide")
        self.assertEqual(a.max_words, 300)  # explicit cap beats the style's
        a = parse(["run", "x", "--style", "actionable"])
        self.assertEqual(a.style, "actionable")

    def test_ollama_model_env(self):
        with mock.patch.dict(os.environ, {"OLLAMA_MODEL": "llama3"}):
            self.assertEqual(parse(["summarize", "x"]).model, "llama3")
            self.assertEqual(parse(["run", "x"]).llm_model, "llama3")

    def test_jobs_env(self):
        with mock.patch.dict(os.environ, {"YT_JOBS": "5"}):
            self.assertEqual(parse(["run", "x"]).jobs, 5)

    def test_fetch_has_no_transcribe_options(self):
        a = parse(["fetch", "dQw4w9WgXcQ"])
        self.assertFalse(hasattr(a, "lang"))
        self.assertFalse(hasattr(a, "timestamps"))

    def test_batch_option_repeatable(self):
        a = parse(["run", "--batch", "a.txt", "--batch", "b.txt", "x"])
        self.assertEqual(a.batch, ["a.txt", "b.txt"])


class TestJobsType(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(cli._jobs_type("1"), 1)
        self.assertEqual(cli._jobs_type("12"), 12)

    def test_rejects_non_int(self):
        with self.assertRaises(SystemExit):  # argparse turns it into exit 2
            parse(["run", "-j", "two", "x"])

    def test_rejects_below_one(self):
        with self.assertRaises(SystemExit):
            parse(["run", "-j", "0", "x"])

    def test_env_jobs_invalid_dies(self):
        with mock.patch.dict(os.environ, {"YT_JOBS": "nope"}):
            err, exc = stderr_of(cli.env_jobs)
        self.assertIsInstance(exc, SystemExit)
        self.assertIn("$YT_JOBS", err)

    def test_env_jobs_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("YT_JOBS", None)
            self.assertEqual(cli.env_jobs(), 2)

    def test_env_jobs_valid(self):
        with mock.patch.dict(os.environ, {"YT_JOBS": "3"}):
            self.assertEqual(cli.env_jobs(), 3)


class TestMainDispatch(unittest.TestCase):
    """main() with the command handlers patched out (argument plumbing only)."""

    def run_main(self, argv):
        import contextlib, io
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("ytpipeline.pipeline.cmd_run",
                        return_value=0) as cmd_run, \
             mock.patch("ytpipeline.pipeline.cmd_doctor",
                        return_value=0) as cmd_doctor:
            with contextlib.redirect_stdout(out), \
                 contextlib.redirect_stderr(err):
                try:
                    cli.main(list(argv))
                    code = 0
                except SystemExit as e:
                    code = e.code
        return code, out.getvalue(), err.getvalue(), cmd_run, cmd_doctor

    def test_run_subcommand(self):
        code, _, _, cmd_run, _ = self.run_main(["run", "dQw4w9WgXcQ"])
        self.assertEqual(code, 0)
        cmd_run.assert_called_once()
        self.assertEqual(cmd_run.call_args[0][0].videos, ["dQw4w9WgXcQ"])

    def test_bare_url_means_run(self):
        code, _, _, cmd_run, _ = self.run_main(["dQw4w9WgXcQ"])
        self.assertEqual(code, 0)
        cmd_run.assert_called_once()

    def test_bare_url_with_batch_options(self):
        for argv in (["--batch", "f.txt"], ["-j", "2", "x"],
                     ["--jobs=3", "x"], ["-b", "f.txt"]):
            with self.subTest(argv=argv):
                code, _, _, cmd_run, _ = self.run_main(argv)
                self.assertEqual(code, 0)
                cmd_run.assert_called_once()

    def test_doctor_subcommand(self):
        code, _, _, _, cmd_doctor = self.run_main(["doctor"])
        self.assertEqual(code, 0)
        cmd_doctor.assert_called_once()

    def test_unknown_flag_is_argparse_error(self):
        code, _, err, cmd_run, _ = self.run_main(["--bogus", "x"])
        self.assertEqual(code, 2)
        cmd_run.assert_not_called()
        self.assertIn("error:", err)  # argparse rejects it (never a 'run')

    def test_no_args_prints_help(self):
        code, out, _, cmd_run, _ = self.run_main([])
        self.assertEqual(code, 2)
        cmd_run.assert_not_called()
        self.assertIn("usage:", out)

    def test_version(self):
        code, out, _, _, _ = self.run_main(["--version"])
        self.assertEqual(code, 0)
        self.assertIn(f"yt-pipeline {VERSION}", out)

    def test_subcommand_help_shows_examples(self):
        code, out, _, cmd_run, _ = self.run_main(["run", "--help"])
        self.assertEqual(code, 0)
        cmd_run.assert_not_called()
        self.assertIn("examples:", out)


if __name__ == "__main__":
    unittest.main()
