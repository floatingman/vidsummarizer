"""Unit tests for ytpipeline.batch: the worker pool and its failure
containment."""

import threading
import time
import unittest

from tests.helpers import stderr_of

from ytpipeline.batch import _fmt_took, _guarded, run_batch


class TestGuarded(unittest.TestCase):
    def test_result_passed_through(self):
        self.assertEqual(_guarded("vid", lambda v: v.upper(), True), "VID")

    def test_systemexit_string_becomes_runtime_error(self):
        def die_like(v):
            raise SystemExit("video unavailable")
        with self.assertRaises(RuntimeError) as cm:
            _guarded("vid", die_like, True)
        self.assertEqual(str(cm.exception), "video unavailable")

    def test_systemexit_int_becomes_runtime_error(self):
        def die_like(v):
            raise SystemExit(1)
        with self.assertRaises(RuntimeError) as cm:
            _guarded("vid", die_like, True)
        self.assertIn("exit 1", str(cm.exception))

    def test_other_exceptions_propagate(self):
        def boom(v):
            raise ValueError("x")
        with self.assertRaises(ValueError):
            _guarded("vid", boom, True)

    def test_log_tag_set_during_run_and_cleared_after(self):
        from ytpipeline.common import _log_tag
        seen = {}

        def probe(v):
            seen["tag"] = getattr(_log_tag, "value", None)
        _guarded("vid42", probe, tag=True)
        self.assertEqual(seen["tag"], "vid42")
        self.assertIsNone(getattr(_log_tag, "value", None))

    def test_no_tag_when_single_video(self):
        from ytpipeline.common import _log_tag
        seen = {}

        def probe(v):
            seen["tag"] = getattr(_log_tag, "value", None)
        _guarded("vid", probe, tag=False)
        self.assertIsNone(seen["tag"])


class TestFmtTook(unittest.TestCase):
    def test_seconds_with_one_decimal(self):
        self.assertEqual(_fmt_took(1.234), "1.2s")
        self.assertEqual(_fmt_took(59.9), "59.9s")

    def test_minutes_formatted_as_duration(self):
        self.assertEqual(_fmt_took(61.2), "01:01")
        self.assertEqual(_fmt_took(3661), "1:01:01")


class TestRunBatch(unittest.TestCase):
    def test_all_success(self):
        results, failures = run_batch(["a", "b"], lambda v: v * 2, jobs=2)
        self.assertEqual(results, {"a": "aa", "b": "bb"})
        self.assertEqual(failures, [])

    def test_failure_is_contained(self):
        def fn(v):
            if v == "bad":
                raise RuntimeError("dead video")
            return v
        results, failures = run_batch(["good", "bad", "also"], fn, jobs=2)
        self.assertEqual(results, {"good": "good", "also": "also"})
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][0], "bad")
        self.assertIn("dead video", str(failures[0][1]))

    def test_die_in_worker_becomes_failure_not_crash(self):
        def fn(v):
            raise SystemExit("no transcript")
        results, failures = run_batch(["a", "b"], fn, jobs=2)
        self.assertEqual(len(failures), 2)
        self.assertIn("no transcript", str(failures[0][1]))

    def test_batch_log_summary_with_failures(self):
        def fn(v):
            if v == "aaaaaaaaaaa":
                raise RuntimeError("dead video")
            return v
        err, _ = stderr_of(
            lambda: run_batch(["aaaaaaaaaaa", "bbbbbbbbbbb"], fn, jobs=1))
        self.assertIn("1/2 video(s) failed", err)
        self.assertIn("aaaaaaaaaaa", err)
        self.assertIn("batch finished", err)

    def test_workers_run_in_parallel(self):
        barrier = threading.Barrier(3, timeout=15)

        def fn(v):
            barrier.wait()  # only passes if all 3 run concurrently
            return v
        results, failures = run_batch(["a", "b", "c"], fn, jobs=3)
        self.assertEqual(failures, [])
        self.assertEqual(len(results), 3)

    def test_jobs_1_is_strictly_sequential(self):
        lock = threading.Lock()
        state = {"inside": 0, "max": 0}

        def fn(v):
            with lock:
                state["inside"] += 1
                state["max"] = max(state["max"], state["inside"])
            time.sleep(0.02)
            with lock:
                state["inside"] -= 1
            return v
        results, _ = run_batch(["a", "b", "c"], fn, jobs=1)
        self.assertEqual(len(results), 3)
        self.assertEqual(state["max"], 1)

    def test_intro_logged(self):
        err, _ = stderr_of(
            lambda: run_batch(["a"], lambda v: v, jobs=1, intro="* starting"))
        self.assertIn("* starting", err)


if __name__ == "__main__":
    unittest.main()
