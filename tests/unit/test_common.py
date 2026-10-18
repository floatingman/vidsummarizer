"""Unit tests for vidsummarizer.common (parsing, logging, subprocess wrapper,
tool detection, workdir resolution)."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import assert_dies, capture as _capture, ns, stderr_of

from vidsummarizer import common


class TestVideoIdParsing(unittest.TestCase):
    ID = "dQw4w9WgXcQ"

    def test_bare_id(self):
        self.assertEqual(common.parse_video_id(self.ID), self.ID)

    def test_bare_id_with_dash_and_underscore(self):
        vid = "abc-DEF_gh9"
        self.assertEqual(common.parse_video_id(vid), vid)

    def test_whitespace_is_stripped(self):
        self.assertEqual(common.parse_video_id(f"  {self.ID}\n"), self.ID)

    def test_garbage_dies_with_message(self):
        assert_dies(self, lambda: common.parse_video_id("hello"),
                    pattern=r"11-character YouTube video id")

    def test_wrong_length_dies(self):
        for bad in ("", "short", "waytoolongidentifier11"):
            assert_dies(self, lambda bad=bad: common.parse_video_id(bad),
                        pattern="could not extract")

    def test_plain_domain_dies(self):
        assert_dies(self, lambda: common.parse_video_id("youtube.com"),
                    pattern="could not extract")

    def test_watch_url_is_built_from_id(self):
        self.assertEqual(common.watch_url("abc12345678"),
                         "https://www.youtube.com/watch?v=abc12345678")


class TestVideoTokens(unittest.TestCase):
    def test_none_gives_empty(self):
        self.assertEqual(common.split_video_tokens(None), [])

    def test_single_token(self):
        self.assertEqual(common.split_video_tokens(["abc"]), ["abc"])

    def test_comma_separated(self):
        self.assertEqual(common.split_video_tokens(["a,b", "c"]), ["a", "b", "c"])

    def test_spaces_and_empty_parts_dropped(self):
        self.assertEqual(
            common.split_video_tokens([" a , ,b ", "", "c,"]), ["a", "b", "c"])

    def test_non_string_is_coerced(self):
        self.assertEqual(common.split_video_tokens([123]), ["123"])


class TestBatchFile(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-batch-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_one_per_line_with_comments_and_blanks(self):
        f = self.dir / "videos.txt"
        f.write_text(
            "# watchlist\n"
            "dQw4w9WgXcQ\n"
            "\n"
            "   # indented comment\n"
            "jNQXAC9IVRw\n"
            "https://youtu.be/aaaaaaaaaaa  # trailing comment\n",
            encoding="utf-8")
        self.assertEqual(
            common.read_batch_file(str(f)),
            ["dQw4w9WgXcQ", "jNQXAC9IVRw", "https://youtu.be/aaaaaaaaaaa"])

    def test_commas_also_separate(self):
        f = self.dir / "videos.txt"
        f.write_text("a,b\nc\n", encoding="utf-8")
        self.assertEqual(common.read_batch_file(str(f)), ["a", "b", "c"])

    def test_dash_reads_stdin(self):
        with mock.patch("sys.stdin", new=io_string("a\nb\n")):
            self.assertEqual(common.read_batch_file("-"), ["a", "b"])

    def test_missing_file_dies(self):
        assert_dies(self,
                    lambda: common.read_batch_file(str(self.dir / "nope.txt")),
                    pattern="cannot read batch file")


def io_string(text):
    import io
    return io.StringIO(text)


class TestCollectVideos(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-collect-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_positional_only(self):
        self.assertEqual(common.collect_videos(["dQw4w9WgXcQ"], None),
                         ["dQw4w9WgXcQ"])

    def test_comma_separated_and_urls(self):
        vids = common.collect_videos(
            ["https://youtu.be/dQw4w9WgXcQ,jNQXAC9IVRw",
             "https://www.youtube.com/watch?v=abcdefghijk"], None)
        self.assertEqual(vids, ["dQw4w9WgXcQ", "jNQXAC9IVRw", "abcdefghijk"])

    def test_batch_file_merged_in_order(self):
        f = self.dir / "b.txt"
        f.write_text("jNQXAC9IVRw\n", encoding="utf-8")
        self.assertEqual(
            common.collect_videos(["dQw4w9WgXcQ"], [str(f)]),
            ["dQw4w9WgXcQ", "jNQXAC9IVRw"])

    def test_duplicates_dropped_with_warning(self):
        err, _ = stderr_of(
            lambda: self.assertEqual(
                common.collect_videos(["aaaaaaaaaaa,bbbbbbbbbbb", "aaaaaaaaaaa"],
                                      None),
                ["aaaaaaaaaaa", "bbbbbbbbbbb"]))
        self.assertIn("ignoring duplicate video(s)", err)

    def test_no_videos_dies(self):
        assert_dies(self, lambda: common.collect_videos([], None),
                    pattern="no videos given")

    def test_invalid_token_dies(self):
        assert_dies(self, lambda: common.collect_videos(["short"], None),
                    pattern="could not extract")


class TestTimeFormat(unittest.TestCase):
    def test_fmt_ts_under_a_minute(self):
        self.assertEqual(common.fmt_ts(0), "00:00")
        self.assertEqual(common.fmt_ts(61), "01:01")

    def test_fmt_ts_rounds(self):
        self.assertEqual(common.fmt_ts(59.6), "01:00")
        self.assertEqual(common.fmt_ts(0.4), "00:00")

    def test_fmt_ts_hours(self):
        self.assertEqual(common.fmt_ts(3600), "1:00:00")
        self.assertEqual(common.fmt_ts(3661), "1:01:01")

    def test_fmt_duration_none_is_empty(self):
        self.assertEqual(common.fmt_duration(None), "")

    def test_fmt_duration_matches_fmt_ts_for_numbers(self):
        for s in (0, 61, 3600, 3661):
            self.assertEqual(common.fmt_duration(s), common.fmt_ts(s))


class TestLogging(unittest.TestCase):
    def test_log_warn_prefix(self):
        err, _ = stderr_of(lambda: common.log("hello"))
        self.assertIn("hello", err)

        err, _ = stderr_of(lambda: common.warn("careful"))
        self.assertIn("careful", err)
        self.assertIn("warning", err)

    def test_die_message_and_exit_code(self):
        assert_dies(self, lambda: common.die("bad thing"), pattern="bad thing")

    def test_die_custom_code(self):
        assert_dies(self, lambda: common.die("x", code=3), pattern="x", code=3)

    def test_tag_appears_on_log_lines(self):
        def run():
            common.set_log_tag("vid42")
            try:
                common.log("step one")
                common.warn("step two")
            finally:
                common.set_log_tag(None)
        err, _ = stderr_of(run)
        self.assertIn("[vid42] step one", err)
        self.assertIn("warning: [vid42] step two", err)

    def test_tag_not_doubled_on_bracketed_lines(self):
        common.set_log_tag("vid42")
        try:
            err, _ = stderr_of(lambda: common.log("[already] tagged"))
        finally:
            common.set_log_tag(None)
        self.assertIn("[already] tagged", err)
        self.assertNotIn("vid42", err)

    def test_die_includes_tag(self):
        common.set_log_tag("vid42")
        try:
            err = assert_dies(self, lambda: common.die("boom"), pattern="boom")
        finally:
            common.set_log_tag(None)
        self.assertIn("[vid42]", err)


class TestRun(unittest.TestCase):
    def _proc(self, rc=0, out="", errout=""):
        return ns(returncode=rc, stdout=out, stderr=errout)

    def test_success_returns_proc(self):
        with mock.patch("vidsummarizer.common.subprocess.run",
                        return_value=self._proc(0, out="ok")) as m:
            proc = common.run(["ls"], "listing")
        self.assertEqual(proc.stdout, "ok")
        m.assert_called_once()

    def test_failure_dies_by_default_with_stderr_tail(self):
        def fake_run(cmd, **kw):
            return self._proc(1, errout="line1\nline2\nline3")
        with mock.patch("vidsummarizer.common.subprocess.run", fake_run):
            err = assert_dies(self,
                              lambda: common.run(["boom"], "boom stage"),
                              pattern="boom stage failed")
        self.assertIn("line3", err)  # the tail of the tool's stderr

    def test_failure_nonfatal_raises_runtime_error(self):
        def fake_run(cmd, **kw):
            return self._proc(1, errout="nope")
        with mock.patch("vidsummarizer.common.subprocess.run", fake_run):
            with self.assertRaises(RuntimeError):
                common.run(["boom"], "boom stage", fatal=False)

    def test_command_not_found_dies(self):
        def fake_run(cmd, **kw):
            raise FileNotFoundError(cmd[0])
        with mock.patch("vidsummarizer.common.subprocess.run", fake_run):
            assert_dies(self, lambda: common.run(["no-such-tool"], "stage x"),
                        pattern="command not found")


class TestToolDetection(unittest.TestCase):
    def test_missing_tools_lists_absent(self):
        present = {"yt-dlp", "ffmpeg", "ollama"}

        def fake_which(name, path=None):
            return "/usr/bin/" + name if name in present else None
        with mock.patch("vidsummarizer.common.shutil.which", fake_which):
            self.assertEqual(common.missing_tools(["yt-dlp", "whisper-cli"]),
                             ["whisper-cli"])
            self.assertEqual(common.missing_tools(["ffmpeg"]), [])

    def test_require_tools_ok_when_present(self):
        with mock.patch("vidsummarizer.common.shutil.which",
                        lambda name, path=None: "/usr/bin/" + name):
            err, exc = stderr_of(
                lambda: common.require_tools(["yt-dlp", "ffmpeg"]))
        self.assertIsNone(exc)
        self.assertEqual(err, "")

    def test_require_tools_dies_with_install_hints(self):
        with mock.patch("vidsummarizer.common.shutil.which", lambda n, path=None: None):
            err = assert_dies(
                self, lambda: common.require_tools(["yt-dlp", "whisper-cli"]),
                pattern="missing required external tool")
        self.assertIn("yt-dlp", err)
        self.assertIn("whisper-cli", err)
        self.assertIn("pipx install yt-dlp", err)   # install hint shown
        self.assertIn("whisper.cpp", err)
        self.assertIn("doctor", err)                # pointer to doctor


class TestBaseDir(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-basedir-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_executable_path_gives_its_directory(self):
        exe = self.dir / "vidsummarizer"
        exe.write_text("#!/bin/sh\n")
        with mock.patch("sys.argv", [str(exe), "run"]):
            self.assertEqual(common.base_dir(), self.dir)

    def test_source_checkout_gives_package_parent(self):
        # argv[0] is __main__.py or a .py file -> repo root (package parent)
        main_py = self.dir / "vidsummarizer" / "__main__.py"
        main_py.parent.mkdir()
        main_py.write_text("")
        with mock.patch("sys.argv", [str(main_py)]):
            self.assertEqual(common.base_dir(),
                             Path(common.__file__).resolve().parent.parent)

    def test_no_argv_gives_package_parent(self):
        with mock.patch("sys.argv", []):
            self.assertEqual(common.base_dir(),
                             Path(common.__file__).resolve().parent.parent)


class TestResolveWorkdir(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-wd-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.patcher = mock.patch.dict(os.environ)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        os.environ.pop("YT_WORKDIR", None)
        os.environ.pop("XDG_DATA_HOME", None)

    def test_explicit_wins_over_env(self):
        os.environ["YT_WORKDIR"] = str(self.dir / "from-env")
        wd = common.resolve_workdir(str(self.dir / "explicit"))
        self.assertEqual(wd, self.dir / "explicit")

    def test_env_used_when_no_explicit(self):
        os.environ["YT_WORKDIR"] = str(self.dir / "from-env")
        self.assertEqual(common.resolve_workdir(None), self.dir / "from-env")

    def test_writable_base_dir_used(self):
        base = self.dir / "base"
        base.mkdir()
        with mock.patch("vidsummarizer.common.base_dir",
                        return_value=base):
            wd, _ = _capture(lambda: common.resolve_workdir(None))
        self.assertEqual(wd, base)

    def test_readonly_base_dir_falls_back_to_xdg(self):
        ro = self.dir / "ro"
        ro.mkdir()
        ro.chmod(0o555)
        self.addCleanup(lambda: ro.chmod(0o755))
        xdg = self.dir / "xdg"
        xdg.mkdir()
        with mock.patch("vidsummarizer.common.base_dir", return_value=ro), \
             mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(xdg)}):
            wd, err = _capture(lambda: common.resolve_workdir(None))
        self.assertEqual(wd, xdg / "vidsummarizer")
        self.assertIn("not writable", err)
        self.assertIn("--workdir", err)          # tells the user the way out

    def test_explicit_not_checked_for_writability(self):
        ro = self.dir / "ro2"
        ro.mkdir()
        ro.chmod(0o555)
        self.addCleanup(lambda: ro.chmod(0o755))
        # explicit --workdir is honored as-is (the OS will complain later)
        self.assertEqual(common.resolve_workdir(str(ro)), ro)


if __name__ == "__main__":
    unittest.main()
