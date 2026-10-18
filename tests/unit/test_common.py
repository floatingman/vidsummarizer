"""Unit tests for ytpipeline.common (parsing, logging, subprocess wrapper,
title-based output naming, tool detection, workdir resolution)."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import assert_dies, capture as _capture, ns, stderr_of

from ytpipeline import common


class TestSafeName(unittest.TestCase):
    def test_strips_filename_hostile_characters(self):
        self.assertEqual(common.safe_name('a/b\\c:d*e?f"g<h>i|j#k^l[m]n'),
                         "abcdefghijklmn")

    def test_control_characters_stripped(self):
        self.assertEqual(common.safe_name("bad\x00\x1fname"), "badname")

    def test_collapses_whitespace_and_strips_dots(self):
        self.assertEqual(common.safe_name("  A   B .. "), "A B")

    def test_char_cap(self):
        self.assertEqual(len(common.safe_name("x" * 200)), 90)
        self.assertEqual(common.safe_name("x" * 200, max_len=7), "x" * 7)

    def test_byte_cap_for_multibyte_titles(self):
        # the byte cap must bite for titles made of 4-byte characters
        name = common.safe_name("\U0001f600" * 200)  # emoji
        self.assertLessEqual(len(name.encode("utf-8")), common.MAX_TITLE_BYTES)
        self.assertTrue(name)  # never truncated to nothing

    def test_only_hostile_characters_gives_empty(self):
        self.assertEqual(common.safe_name("///???"), "")


class TestTitledPath(unittest.TestCase):
    """The collision rules shared by transcripts/*.txt and summaries/*.md."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-name-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "out"

    def test_free_name_used_directly_and_dir_created(self):
        p = common.titled_path(self.dir, "My Talk", "abcdefghijk", ".txt")
        self.assertEqual(p, self.dir / "My Talk.txt")
        self.assertTrue(self.dir.is_dir())

    def test_same_video_keeps_its_own_file(self):
        p = common.titled_path(self.dir, "My Talk", "abcdefghijk", ".txt")
        p.write_text("video_id: abcdefghijk\n\nbody", encoding="utf-8")
        self.assertEqual(
            common.titled_path(self.dir, "My Talk", "abcdefghijk", ".txt"), p)

    def test_different_video_gets_id_suffix(self):
        p = common.titled_path(self.dir, "My Talk", "abcdefghijk", ".txt")
        p.write_text("video_id: abcdefghijk\n\nbody", encoding="utf-8")
        self.assertEqual(
            common.titled_path(self.dir, "My Talk", "lmnopqrstuv", ".txt"),
            self.dir / "My Talk (lmnopqrstuv).txt")

    def test_suffixed_name_stays_with_its_video(self):
        # once a video owns "Name (vid).txt", re-runs keep updating it
        plain = common.titled_path(self.dir, "Talk", "abcdefghijk", ".txt")
        plain.write_text("video_id: abcdefghijk\n\nbody", encoding="utf-8")
        suffixed = common.titled_path(self.dir, "Talk", "lmnopqrstuv", ".txt")
        suffixed.write_text("video_id: lmnopqrstuv\n\nbody", encoding="utf-8")
        self.assertEqual(
            common.titled_path(self.dir, "Talk", "lmnopqrstuv", ".txt"),
            suffixed)

    def test_file_without_video_id_is_treated_as_foreign(self):
        p = common.titled_path(self.dir, "Talk", "abcdefghijk", ".txt")
        p.write_text("no header at all", encoding="utf-8")
        self.assertEqual(
            common.titled_path(self.dir, "Talk", "abcdefghijk", ".txt"),
            self.dir / "Talk (abcdefghijk).txt")

    def test_no_video_id_never_disambiguates(self):
        p = common.titled_path(self.dir, "Talk", "", ".txt")
        p.write_text("no ids here", encoding="utf-8")
        self.assertEqual(common.titled_path(self.dir, "Talk", "", ".txt"), p)

    def test_quoted_frontmatter_id_is_read_too(self):
        p = common.titled_path(self.dir, "Talk", "abcdefghijk", ".md")
        p.write_text('---\nvideo_id: "abcdefghijk"\n---\n', encoding="utf-8")
        self.assertEqual(
            common.titled_path(self.dir, "Talk", "abcdefghijk", ".md"), p)
        self.assertEqual(
            common.titled_path(self.dir, "Talk", "lmnopqrstuv", ".md"),
            self.dir / "Talk (lmnopqrstuv).md")

    def test_suffixed_name_owned_by_a_third_video_gets_a_counter(self):
        # contrived: a video C whose title literally ends in " (<id>)" of B
        plain = common.titled_path(self.dir, "Base", "aaaaaaaaaaa", ".txt")
        plain.write_text("video_id: aaaaaaaaaaa\n\nA", encoding="utf-8")
        impostor = self.dir / "Base (bbbbbbbbbbb).txt"
        impostor.write_text("title: Base (bbbbbbbbbbb)\nvideo_id: ccccccccccc\n\nC",
                            encoding="utf-8")
        # video B (title "Base", id bbbbbbbbbbb) must not take C's file
        got = common.titled_path(self.dir, "Base", "bbbbbbbbbbb", ".txt")
        self.assertEqual(got, self.dir / "Base (bbbbbbbbbbb) (2).txt")
        self.assertIn("video_id: ccccccccccc",
                      impostor.read_text(encoding="utf-8"))  # untouched
        # and the assignment is stable across re-runs
        got.write_text("video_id: bbbbbbbbbbb\n\nB", encoding="utf-8")
        self.assertEqual(
            common.titled_path(self.dir, "Base", "bbbbbbbbbbb", ".txt"), got)

    def test_counter_walks_past_every_impostor(self):
        # contrived chain: plain name owned by A, the id-suffixed name by an
        # impostor C, and the (2) counter name by another impostor D
        plain = common.titled_path(self.dir, "Base", "aaaaaaaaaaa", ".txt")
        plain.write_text("video_id: aaaaaaaaaaa\n\nA", encoding="utf-8")
        for name, vid in (("Base (bbbbbbbbbbb).txt", "ccccccccccc"),
                          ("Base (bbbbbbbbbbb) (2).txt", "ddddddddddd")):
            (self.dir / name).write_text(f"title: {name[:-4]}\n"
                                         f"video_id: {vid}\n\nX",
                                         encoding="utf-8")
        got = common.titled_path(self.dir, "Base", "bbbbbbbbbbb", ".txt")
        self.assertEqual(got, self.dir / "Base (bbbbbbbbbbb) (3).txt")


    def test_exhausted_candidates_die_instead_of_overwriting(self):
        # contrived to the limit: the plain name, the id-suffixed name and
        # every counter through (99) owned by other videos -> no free name
        self.dir.mkdir(parents=True, exist_ok=True)
        names = ["Base.txt", "Base (bbbbbbbbbbb).txt"]
        names += [f"Base (bbbbbbbbbbb) ({n}).txt" for n in range(2, 100)]
        for name in names:
            (self.dir / name).write_text(
                f"video_id: ccccccccccc\n\n{name}", encoding="utf-8")
        assert_dies(self,
                    lambda: common.titled_path(self.dir, "Base",
                                               "bbbbbbbbbbb", ".txt"),
                    pattern="no free output name")
        # none of the other videos' files were taken
        self.assertTrue(all((self.dir / n).is_file() for n in names))


class TestOutputDirLock(unittest.TestCase):
    """output_dir_lock must exclude other processes, not just threads: two
    yt-pipeline runs transcribing same-titled videos rely on it."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-lock-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "transcripts"

    def test_threads_of_one_process_are_serialized(self):
        import threading
        order = []
        barrier = threading.Barrier(2)

        def worker(n):
            barrier.wait()  # both threads reach the lock at the same time
            with common.output_dir_lock(self.dir):
                order.append(("in", n))
                order.append(("out", n))

        ts = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        # the critical sections never interleave: each "in" is immediately
        # followed by its own "out"
        for i in range(0, len(order), 2):
            self.assertEqual(order[i][0], "in")
            self.assertEqual(order[i + 1][0], "out")
            self.assertEqual(order[i][1], order[i + 1][1])

    def test_second_process_blocks_until_released(self):
        import subprocess
        import sys
        import textwrap
        import time
        if common.fcntl is None:
            self.skipTest("no flock on this platform")
        from tests.helpers import REPO
        marker = self.dir.parent / "held"
        code = textwrap.dedent("""\
            import sys, time
            sys.path.insert(0, {repo!r})
            from ytpipeline.common import output_dir_lock
            with output_dir_lock({dir!r}):
                open({marker!r}, "w").close()
                time.sleep(1.5)
            """).format(repo=str(REPO), dir=str(self.dir), marker=str(marker))
        proc = subprocess.Popen([sys.executable, "-c", code])
        try:
            deadline = time.monotonic() + 10
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(marker.exists(), "child never acquired the lock")
            t0 = time.monotonic()
            with common.output_dir_lock(self.dir):
                blocked_for = time.monotonic() - t0
            self.assertGreaterEqual(blocked_for, 0.5,
                                    f"lock did not exclude the other "
                                    f"process (blocked {blocked_for:.2f}s)")
        finally:
            proc.wait(timeout=10)

    def test_lock_file_is_created_and_harmless(self):
        with common.output_dir_lock(self.dir):
            pass
        self.assertTrue((self.dir / ".yt-pipeline.lock").exists())


class TestWriteTextAtomic(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-aw-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_writes_content_and_leaves_no_temp_file(self):
        p = self.dir / "Talk.txt"
        common.write_text_atomic(p, "hello\n")
        self.assertEqual(p.read_text(encoding="utf-8"), "hello\n")
        self.assertEqual(list(self.dir.iterdir()), [p])
        # overwrites in place, still no leftovers
        common.write_text_atomic(p, "again\n")
        self.assertEqual(p.read_text(encoding="utf-8"), "again\n")
        self.assertEqual(list(self.dir.iterdir()), [p])

    def test_temp_names_never_look_like_transcripts(self):
        # the temp file must not match the *.txt scan patterns
        p = self.dir / "Talk.txt"
        common.write_text_atomic(p, "x")
        self.assertEqual([f.name for f in self.dir.glob("*.txt")],
                         ["Talk.txt"])

    def test_long_basename_still_writable(self):
        # a destination basename near the filesystem's 255-byte limit must
        # not grow an even longer temp name on top (ENAMETOOLONG)
        p = self.dir / ("L" * 250 + ".txt")
        common.write_text_atomic(p, "payload\n")
        self.assertEqual(p.read_text(encoding="utf-8"), "payload\n")
        self.assertEqual(list(self.dir.iterdir()), [p])



class TestTranscriptLookup(unittest.TestCase):
    """Locating a transcript by video id: title-named and legacy files."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-look-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "transcripts"
        self.dir.mkdir()

    def test_title_named_file_found_via_header(self):
        (self.dir / "Some Great Talk.txt").write_text(
            "title: Some Great Talk\nvideo_id: abcdefghijk\n\nbody",
            encoding="utf-8")
        self.assertEqual(common.find_transcript("abcdefghijk", [str(self.dir)]),
                         self.dir / "Some Great Talk.txt")

    def test_suffixed_title_named_file_found_via_header(self):
        (self.dir / "Some Great Talk (abcdefghijk).txt").write_text(
            "title: Some Great Talk\nvideo_id: abcdefghijk\n\nbody",
            encoding="utf-8")
        self.assertEqual(common.find_transcript("abcdefghijk", [str(self.dir)]),
                         self.dir / "Some Great Talk (abcdefghijk).txt")

    def test_legacy_id_named_file_found_by_name(self):
        (self.dir / "abcdefghijk.txt").write_text(
            "video_id: abcdefghijk\n\nold", encoding="utf-8")
        self.assertEqual(common.find_transcript("abcdefghijk", [str(self.dir)]),
                         self.dir / "abcdefghijk.txt")

    def test_headerless_legacy_file_found_by_name(self):
        (self.dir / "abcdefghijk.txt").write_text("just words", encoding="utf-8")
        self.assertEqual(common.find_transcript("abcdefghijk", [str(self.dir)]),
                         self.dir / "abcdefghijk.txt")

    def test_title_named_wins_over_legacy(self):
        (self.dir / "abcdefghijk.txt").write_text(
            "video_id: abcdefghijk\n\nold", encoding="utf-8")
        (self.dir / "Some Great Talk.txt").write_text(
            "title: Some Great Talk\nvideo_id: abcdefghijk\n\nnew",
            encoding="utf-8")
        self.assertEqual(common.find_transcript("abcdefghijk", [str(self.dir)]),
                         self.dir / "Some Great Talk.txt")

    def test_second_directory_searched(self):
        other = self.dir.parent / "elsewhere"
        other.mkdir()
        (other / "Talk.txt").write_text("video_id: abcdefghijk\n\nbody",
                                        encoding="utf-8")
        self.assertEqual(
            common.find_transcript("abcdefghijk", [str(self.dir), str(other)]),
            other / "Talk.txt")

    def test_missing_video_is_none(self):
        self.assertIsNone(common.find_transcript("abcdefghijk", [str(self.dir)]))
        self.assertIsNone(
            common.find_transcript("abcdefghijk", [str(self.dir / "nope")]))

    def test_index_skips_non_transcripts(self):
        (self.dir / "notes.md").write_text("video_id: abcdefghijk",
                                          encoding="utf-8")
        (self.dir / "subdir.txt").mkdir()
        self.assertEqual(common.transcripts_by_id(str(self.dir)), {})

    def test_owned_transcripts_lists_every_copy(self):
        # a video may legitimately have several files on disk: a legacy
        # <id>.txt, a file under an older title, and the current one
        legacy = self.dir / "abcdefghijk.txt"
        legacy.write_text("video_id: abcdefghijk\n\nold", encoding="utf-8")
        older = self.dir / "Old Title.txt"
        older.write_text("title: Old Title\nvideo_id: abcdefghijk\n\nold",
                         encoding="utf-8")
        current = self.dir / "New Title.txt"
        current.write_text("title: New Title\nvideo_id: abcdefghijk\n\nnew",
                           encoding="utf-8")
        self.assertEqual(set(common.owned_transcripts(str(self.dir),
                                                      "abcdefghijk")),
                         {legacy, older, current})
        self.assertEqual(common.owned_transcripts(str(self.dir),
                                                  "lmnopqrstuv"), [])

    def test_owned_transcripts_includes_headerless_legacy_file(self):
        legacy = self.dir / "abcdefghijk.txt"
        legacy.write_text("no header", encoding="utf-8")
        self.assertEqual(common.owned_transcripts(str(self.dir), "abcdefghijk"),
                         [legacy])

    def test_headerless_legacy_file_is_indexed_by_its_name(self):
        legacy = self.dir / "abcdefghijk.txt"
        legacy.write_text("no header at all", encoding="utf-8")
        self.assertEqual(common.transcripts_by_id(str(self.dir)),
                         {"abcdefghijk": legacy})

    def test_legacy_name_with_foreign_header_belongs_to_its_header(self):
        # a title that is literally another video's id: the file is that
        # video's title-named output, whatever its name says
        (self.dir / "abcdefghijk.txt").write_text(
            "title: abcdefghijk\nvideo_id: zzzzzzzzzzz\n\nbody",
            encoding="utf-8")
        self.assertEqual(
            common.owned_transcripts(str(self.dir), "abcdefghijk"), [])
        self.assertIsNone(
            common.find_transcript("abcdefghijk", [str(self.dir)]))

    def test_foreign_header_legacy_name_does_not_stop_the_search(self):
        (self.dir / "abcdefghijk.txt").write_text(
            "video_id: zzzzzzzzzzz\n\nbody", encoding="utf-8")
        d2 = self.dir / "d2"
        d2.mkdir()
        good = d2 / "abcdefghijk.txt"
        good.write_text("the real headerless legacy file", encoding="utf-8")
        self.assertEqual(
            common.find_transcript("abcdefghijk", [str(self.dir), str(d2)]),
            good)



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
        with mock.patch("ytpipeline.common.subprocess.run",
                        return_value=self._proc(0, out="ok")) as m:
            proc = common.run(["ls"], "listing")
        self.assertEqual(proc.stdout, "ok")
        m.assert_called_once()

    def test_failure_dies_by_default_with_stderr_tail(self):
        def fake_run(cmd, **kw):
            return self._proc(1, errout="line1\nline2\nline3")
        with mock.patch("ytpipeline.common.subprocess.run", fake_run):
            err = assert_dies(self,
                              lambda: common.run(["boom"], "boom stage"),
                              pattern="boom stage failed")
        self.assertIn("line3", err)  # the tail of the tool's stderr

    def test_failure_nonfatal_raises_runtime_error(self):
        def fake_run(cmd, **kw):
            return self._proc(1, errout="nope")
        with mock.patch("ytpipeline.common.subprocess.run", fake_run):
            with self.assertRaises(RuntimeError):
                common.run(["boom"], "boom stage", fatal=False)

    def test_command_not_found_dies(self):
        def fake_run(cmd, **kw):
            raise FileNotFoundError(cmd[0])
        with mock.patch("ytpipeline.common.subprocess.run", fake_run):
            assert_dies(self, lambda: common.run(["no-such-tool"], "stage x"),
                        pattern="command not found")


class TestToolDetection(unittest.TestCase):
    def test_missing_tools_lists_absent(self):
        present = {"yt-dlp", "ffmpeg", "ollama"}

        def fake_which(name, path=None):
            return "/usr/bin/" + name if name in present else None
        with mock.patch("ytpipeline.common.shutil.which", fake_which):
            self.assertEqual(common.missing_tools(["yt-dlp", "whisper-cli"]),
                             ["whisper-cli"])
            self.assertEqual(common.missing_tools(["ffmpeg"]), [])

    def test_require_tools_ok_when_present(self):
        with mock.patch("ytpipeline.common.shutil.which",
                        lambda name, path=None: "/usr/bin/" + name):
            err, exc = stderr_of(
                lambda: common.require_tools(["yt-dlp", "ffmpeg"]))
        self.assertIsNone(exc)
        self.assertEqual(err, "")

    def test_require_tools_dies_with_install_hints(self):
        with mock.patch("ytpipeline.common.shutil.which", lambda n, path=None: None):
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
        exe = self.dir / "yt-pipeline"
        exe.write_text("#!/bin/sh\n")
        with mock.patch("sys.argv", [str(exe), "run"]):
            self.assertEqual(common.base_dir(), self.dir)

    def test_source_checkout_gives_package_parent(self):
        # argv[0] is __main__.py or a .py file -> repo root (package parent)
        main_py = self.dir / "ytpipeline" / "__main__.py"
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
        with mock.patch("ytpipeline.common.base_dir",
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
        with mock.patch("ytpipeline.common.base_dir", return_value=ro), \
             mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(xdg)}):
            wd, err = _capture(lambda: common.resolve_workdir(None))
        self.assertEqual(wd, xdg / "yt-pipeline")
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
