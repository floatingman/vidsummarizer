"""E2E tests for the benchmark / prompt-variant commands: the real CLI
(`python3 -m ytpipeline`) as a subprocess, with the in-process fake Ollama
server answering both the summarizer and the judge calls (distinguished by
the request text via the server's responder hook).

Covered: a full benchmark run writes the report/rank and exits 0; the
--min-overall quality gate fails a weak active variant; `prompts list`,
`prompts show`, `prompts new` and `prompts promote` (with and without
confirmation) manage the versioned prompt artifacts.
"""

import json
import unittest
from pathlib import Path

from tests.helpers import run_cli

REPO = Path(__file__).resolve().parent.parent.parent

GOOD_OVERVIEW = ("## Overview\n\nThe video walks through the pipeline and "
                 "mentions the number 42.")
GOOD_ACTIONABLE = ("## Actionable summary\n\nGet TLS running.\n\n"
                   "### Actionable steps\n\n1. Open port 443.\n"
                   "2. Run certbot in webroot mode.")
JUDGE_JSON = json.dumps({"faithfulness": 4, "actionability": 4,
                         "clarity": 4, "kind": "overview",
                         "notes": "stub judge"})


class BenchmarkE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.helpers import FakeOllamaServer
        cls.ollama = FakeOllamaServer()

    @classmethod
    def tearDownClass(cls):
        cls.ollama.stop()

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-be2e-")
        self.addCleanup(self._tmp.cleanup)
        self.workdir = Path(self._tmp.name)
        self.ollama.reset()
        self.ollama.responder = self.respond
        # a private copy of the prompt variants: promote below must never
        # touch the checkout's prompts/ directory
        self.pdir = self.workdir / "prompts"
        self.pdir.mkdir()
        for f in sorted((REPO / "prompts").glob("*.json")):
            (self.pdir / f.name).write_text(f.read_text(encoding="utf-8"))
        (self.pdir / "active.txt").write_text("v1-baseline\n")
        from tests.helpers import fake_env
        self.env = fake_env(self.workdir)

    def respond(self, body):
        user = body["messages"][-1]["content"]
        if "Rate the SUMMARY" in user:
            return JUDGE_JSON
        return GOOD_ACTIONABLE if "certbot" in user else GOOD_OVERVIEW

    def host_args(self):
        return ["--host", self.ollama.host, "--no-autostart"]

    # -- benchmark ---------------------------------------------------------- #

    def test_benchmark_writes_report_and_ranks(self):
        r = run_cli(["benchmark", "--host", self.ollama.host,
                     "--no-autostart", "--prompts", "v1-baseline",
                     "--prompts-dir", str(self.pdir),
                     "--out", str(self.workdir / "res")], self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Prompt benchmark report", r.stdout)
        self.assertIn("v1-baseline", r.stdout)
        out = self.workdir / "res"
        report = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("## Comparison", report)
        data = json.loads((out / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["results"]), 3)  # 1 variant x 3 cases
        self.assertTrue((out / "summaries" / "v1-baseline" /
                         "howto-tls.md").exists())
        for row in data["variants"]:
            self.assertEqual(row["auto_failed"], 0, report)

    def test_benchmark_min_overall_gate_can_fail(self):
        r = run_cli(["benchmark", "--host", self.ollama.host,
                     "--no-autostart", "--prompts-dir", str(self.pdir),
                     "--out", str(self.workdir / "res"),
                     "--min-overall", "4.9"], self.env)
        self.assertEqual(r.returncode, 1)  # stub judge scores 4/4/4
        self.assertIn("below the required 4.90", r.stderr)
        # the report is still written for inspection
        self.assertTrue((self.workdir / "res" / "report.md").exists())

    def test_benchmark_unknown_variant_fails(self):
        r = run_cli(["benchmark", "--host", self.ollama.host,
                     "--no-autostart", "--prompts-dir", str(self.pdir),
                     "--prompts", "ghost"], self.env)
        self.assertEqual(r.returncode, 1)
        self.assertIn("unknown prompt variant", r.stderr)

    # -- prompts ------------------------------------------------------------ #

    def test_prompts_list_and_show(self):
        r = run_cli(["prompts", "list", "--prompts-dir", str(self.pdir)],
                    self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("v1-baseline", r.stdout)
        self.assertIn("[active]", r.stdout)
        r = run_cli(["prompts", "show", "v1-baseline",
                     "--prompts-dir", str(self.pdir)], self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('"final"', r.stdout)  # the JSON variant file

    def test_prompts_new_creates_a_comparable_variant(self):
        r = run_cli(["prompts", "new", "tighter", "--from", "v1-baseline",
                     "--prompts-dir", str(self.pdir)], self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        path = self.pdir / "tighter.json"
        self.assertTrue(path.exists())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["name"], "tighter")
        base = json.loads((self.pdir / "v1-baseline.json")
                          .read_text(encoding="utf-8"))
        self.assertEqual(data["prompts"], base["prompts"])
        # and the new variant is immediately benchmarkable
        r = run_cli(["benchmark", "--host", self.ollama.host,
                     "--no-autostart", "--prompts-dir", str(self.pdir),
                     "--prompts", "v1-baseline,tighter",
                     "--out", str(self.workdir / "res")], self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("| tighter", r.stdout)

    def test_prompts_promote_needs_confirmation(self):
        r = run_cli(["prompts", "new", "tighter", "--prompts-dir",
                     str(self.pdir)], self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        r = run_cli(["prompts", "promote", "tighter", "--prompts-dir",
                     str(self.pdir)], self.env, input="n\n")
        self.assertEqual(r.returncode, 1)
        self.assertEqual((self.pdir / "active.txt").read_text().strip(),
                         "v1-baseline")  # unchanged
        r = run_cli(["prompts", "promote", "tighter", "--prompts-dir",
                     str(self.pdir)], self.env, input="y\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((self.pdir / "active.txt").read_text().strip(),
                         "tighter")
        # 'list' reflects the new active variant
        r = run_cli(["prompts", "list", "--prompts-dir", str(self.pdir)],
                    self.env)
        self.assertIn("tighter", r.stdout.split("[active]")[0].splitlines()[-1])

    def test_prompts_promote_yes_skips_confirmation(self):
        r = run_cli(["prompts", "promote", "v1-baseline", "--yes",
                     "--prompts-dir", str(self.pdir)], self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("already the active", r.stdout)

    def test_promote_writes_into_workdir_when_dir_is_bundled(self):
        # no --prompts-dir: reads the bundled/checkout variants and writes
        # <workdir>/prompts, materialized from them on first use
        for f in self.pdir.iterdir():
            f.unlink()
        r = run_cli(["prompts", "promote", "short", "--yes",
                     "--workdir", str(self.workdir)], self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        materialized = self.workdir / "prompts" / "active.txt"
        self.assertEqual(materialized.read_text().strip(), "short")
        self.assertTrue((self.workdir / "prompts" / "v1-baseline.json")
                        .exists())  # the bundled variants came along
        # and the materialized copy is what a later run resolves
        r = run_cli(["prompts", "list", "--workdir", str(self.workdir)],
                    self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("(active: short)", r.stdout)


class TestZipappBenchmark(unittest.TestCase):
    """The built executable must benchmark with the prompt variants AND
    the benchmark set packed inside it: an empty workdir (no prompts/, no
    benchmarks/) resolves both from the zipapp. Regression guards for the
    packed-set 'transcript not found' crash and the prompts-materialization
    NotADirectoryError crash from the executable."""

    EXE = REPO / "dist" / "yt-pipeline"

    def setUp(self):
        if not self.EXE.exists():
            self.skipTest("dist/yt-pipeline not built (run ./build.sh)")
        import tempfile
        from tests.helpers import FakeOllamaServer, fake_env
        self.ollama = FakeOllamaServer()
        self.addCleanup(self.ollama.stop)
        self.ollama.responder = self.respond
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-bzip-")
        self.addCleanup(self._tmp.cleanup)
        self.workdir = Path(self._tmp.name)
        self.env = fake_env(self.workdir)

    def respond(self, body):
        user = body["messages"][-1]["content"]
        if "Rate the SUMMARY" in user:
            return JUDGE_JSON
        return GOOD_ACTIONABLE if "certbot" in user else GOOD_OVERVIEW

    def run_exe(self, *args, timeout=240):
        import subprocess
        import sys
        return subprocess.run(
            [sys.executable, str(self.EXE), *args,
             "--workdir", str(self.workdir)],
            env=self.env, capture_output=True, text=True, errors="replace",
            timeout=timeout)

    def test_benchmark_uses_the_packed_set_and_variants(self):
        r = self.run_exe("benchmark", "--prompts", "v1-baseline",
                         "--host", self.ollama.host, "--no-autostart")
        self.assertEqual(r.returncode, 0, r.stderr)
        # the packed set inside the executable was used: the empty workdir
        # has no benchmarks/ directory of its own
        self.assertIn("benchset", r.stdout)
        self.assertIn("3 cases", r.stdout)
        results = self.workdir / "benchmarks" / "results"
        runs = sorted(results.iterdir())
        self.assertTrue(runs, "no benchmark report written")
        report = (runs[-1] / "report.md").read_text(encoding="utf-8")
        self.assertIn("## Comparison", report)
        self.assertIn("v1-baseline", report)

    def test_prompts_new_materializes_the_packed_variants(self):
        r = self.run_exe("prompts", "new", "tighter")
        self.assertEqual(r.returncode, 0, r.stderr)
        pdir = self.workdir / "prompts"
        self.assertTrue((pdir / "tighter.json").exists(), r.stdout)
        names = sorted(p.name for p in pdir.glob("*.json"))
        self.assertEqual(names, ["actionable.json", "short.json",
                                 "study-guide.json", "tighter.json",
                                 "v1-baseline.json"])


if __name__ == "__main__":
    unittest.main()
