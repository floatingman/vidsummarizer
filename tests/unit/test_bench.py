"""Unit tests for ytpipeline.bench: the benchmark set, automatic quality
checks, the LLM-judge answer parsing, scoring/aggregation, the report, and
a full benchmark run against the in-process fake Ollama server."""

import contextlib
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests.helpers import FakeOllamaServer, ns

from ytpipeline import bench, prompts as P
from ytpipeline.bench import (JUDGE_DIMS, aggregate, auto_checks, judge_summary,
                              load_benchset, overall_score, parse_scores,
                              render_report, run_benchmark, write_report)

REPO = Path(__file__).resolve().parent.parent.parent
FIXTURE_TRANSCRIPT = (REPO / "tests" / "fixtures" / "transcript" /
                      "sample.txt").read_text(encoding="utf-8")

GOOD_OVERVIEW = ("## Overview\n\nThe video explains the pipeline end to end "
                 "and mentions the number 42.")
GOOD_ACTIONABLE = ("## Actionable summary\n\nSet up TLS with certbot.\n\n"
                   "### Actionable steps\n\n1. Open port 443 in the "
                   "firewall.\n2. Run certbot with the webroot plugin.")


def case(kind="overview", must=(), name="c", body="Transcript text."):
    return {"name": name, "kind": kind, "must_contain": list(must),
            "body": body}


class StubOllama:
    model = "stub-model"

    def __init__(self, responses=(GOOD_OVERVIEW,)):
        self.responses = list(responses)
        self.calls = []

    def chat(self, system, user, options, think=False, retries=3):
        self.calls.append(system)
        content = self.responses.pop(0) if len(self.responses) > 1 \
            else self.responses[0]
        return content, {"s": 0.1, "prompt_tok": 5, "gen_tok": 5, "tok_s": 50.0}


# --------------------------------------------------------------------------- #
# benchmark set
# --------------------------------------------------------------------------- #

class TestBenchset(unittest.TestCase):
    def test_default_set_loads_from_the_repo(self):
        bs = load_benchset(None)
        self.assertEqual(bs["name"], "default")
        self.assertEqual([c["name"] for c in bs["cases"]],
                         ["sample-short", "howto-tls", "overview-history"])

    def test_cases_have_bodies_and_kinds(self):
        for c in load_benchset(None)["cases"]:
            self.assertTrue(c["body"].strip())
            self.assertIn(c["kind"], ("actionable", "overview"))

    def test_sample_case_is_the_test_fixture(self):
        # the benchmark set reuses the summarize-stage test fixture
        sample = next(c for c in load_benchset(None)["cases"]
                      if c["name"] == "sample-short")
        self.assertEqual(
            Path(sample["path"]).read_text(encoding="utf-8"),
            FIXTURE_TRANSCRIPT)

    def test_kinds_cover_both_summary_modes(self):
        kinds = {c["kind"] for c in load_benchset(None)["cases"]}
        self.assertEqual(kinds, {"actionable", "overview"})

    def test_missing_case_file_dies(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "set.json"
            bad.write_text(json.dumps(
                {"cases": [{"file": "nope.txt", "kind": "overview"}]}))
            from tests.helpers import assert_dies
            assert_dies(self, lambda: load_benchset(str(bad)),
                        pattern="transcript not found")

    def test_bad_kind_dies(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "set.json"
            bad.write_text(json.dumps(
                {"cases": [{"file": "x.txt", "kind": "poetry"}]}))
            from tests.helpers import assert_dies
            assert_dies(self, lambda: load_benchset(str(bad)),
                        pattern="kind must be")


# --------------------------------------------------------------------------- #
# automatic checks
# --------------------------------------------------------------------------- #

class TestAutoChecks(unittest.TestCase):
    def test_good_overview_passes(self):
        r = auto_checks(case(must=["42"]), GOOD_OVERVIEW, 500)
        self.assertEqual(r["failures"], [])
        self.assertEqual(r["total"], 7)  # 6 + one fact check

    def test_good_actionable_passes(self):
        r = auto_checks(case(kind="actionable", must=["certbot", "443"]),
                        GOOD_ACTIONABLE, 500)
        self.assertEqual(r["failures"], [])

    def test_word_limit(self):
        long = "## Overview\n\n" + " ".join(f"w{i}" for i in range(600))
        r = auto_checks(case(), long, 500)
        self.assertTrue(any("word limit" in f for f in r["failures"]))

    def test_kind_mismatch(self):
        r = auto_checks(case(kind="actionable"), GOOD_OVERVIEW, 500)
        self.assertTrue(any("summary kind" in f for f in r["failures"]))

    def test_missing_fact(self):
        r = auto_checks(case(must=["quantum"]), GOOD_OVERVIEW, 500)
        self.assertEqual(r["failures"], ["content: expected fact 'quantum' "
                                         "not in summary"])

    def test_fact_match_is_case_insensitive(self):
        r = auto_checks(case(kind="actionable", must=["CERTBOT"]),
                        GOOD_ACTIONABLE, 500)
        self.assertEqual(r["failures"], [])

    def test_code_fence_rejected(self):
        r = auto_checks(case(), "```markdown\n" + GOOD_OVERVIEW + "\n```", 500)
        self.assertTrue(any("code fence" in f for f in r["failures"]))

    def test_horizontal_rule_rejected(self):
        r = auto_checks(case(), GOOD_OVERVIEW + "\n\n---\n\ntail", 500)
        self.assertTrue(any("horizontal rule" in f for f in r["failures"]))

    def test_h1_rejected(self):
        r = auto_checks(case(), "# Title\n\n" + GOOD_OVERVIEW, 500)
        self.assertTrue(any("H1" in f for f in r["failures"]))

    def test_bad_top_heading_rejected(self):
        r = auto_checks(case(), "## Random stuff\n\ntext", 500)
        self.assertTrue(any("unknown top heading" in f for f in r["failures"]))


# --------------------------------------------------------------------------- #
# judge answer parsing
# --------------------------------------------------------------------------- #

class TestParseScores(unittest.TestCase):
    def test_clean_json(self):
        s = parse_scores('{"faithfulness": 5, "actionability": 4, '
                         '"clarity": 3, "kind": "overview", "notes": "ok"}')
        self.assertEqual((s["faithfulness"], s["actionability"], s["clarity"]),
                         (5, 4, 3))
        self.assertEqual(s["kind"], "overview")

    def test_fenced_json(self):
        s = parse_scores('```json\n{"faithfulness": 4, "actionability": 4, '
                         '"clarity": 4, "kind": "actionable"}\n```')
        self.assertEqual(s["faithfulness"], 4)

    def test_json_with_surrounding_talk(self):
        s = parse_scores('Here you go:\n{"faithfulness": 2, '
                         '"actionability": 2, "clarity": 2, "kind": "x"}\n'
                         'Hope this helps!')
        self.assertEqual(s["faithfulness"], 2)
        self.assertIsNone(s["kind"])  # unknown kind is reported as None

    def test_scores_clamped_to_1_5(self):
        s = parse_scores('{"faithfulness": 9, "actionability": 0, '
                         '"clarity": -3, "kind": "overview"}')
        self.assertEqual((s["faithfulness"], s["actionability"], s["clarity"]),
                         (5, 1, 1))

    def test_missing_dimension_is_none(self):
        self.assertIsNone(parse_scores('{"faithfulness": 5}'))

    def test_garbage_is_none(self):
        self.assertIsNone(parse_scores("I liked the summary a lot!"))
        self.assertIsNone(parse_scores(""))
        self.assertIsNone(parse_scores("[1, 2, 3]"))


class TestJudgeSummary(unittest.TestCase):
    def bopts(self):
        return ns(max_num_ctx=32768)

    def test_good_answer(self):
        stub = StubOllama([json.dumps({"faithfulness": 5, "actionability": 5,
                                       "clarity": 4, "kind": "overview",
                                       "notes": "n"})])
        s = judge_summary(stub, case(body="T."), GOOD_OVERVIEW, self.bopts())
        self.assertEqual(s["faithfulness"], 5)

    def test_one_retry_then_success(self):
        junk = StubOllama(["not json at all",
                           json.dumps({"faithfulness": 3, "actionability": 3,
                                       "clarity": 3, "kind": "overview"})])
        s = judge_summary(junk, case(body="T."), GOOD_OVERVIEW, self.bopts())
        self.assertEqual(s["faithfulness"], 3)
        self.assertEqual(len(junk.calls), 2)

    def test_two_bad_answers_give_up(self):
        junk = StubOllama(["nope", "still nope"])
        s = judge_summary(junk, case(body="T."), GOOD_OVERVIEW, self.bopts())
        self.assertIsNone(s)
        self.assertEqual(len(junk.calls), 2)

    def test_prompt_carries_transcript_and_summary(self):
        stub = StubOllama([json.dumps({"faithfulness": 1, "actionability": 1,
                                       "clarity": 1, "kind": "overview"})])
        judge_summary(stub, case(body="UNIQUE-TRANSCRIPT-MARKER"),
                      "SUMMARY-MARKER", self.bopts())


class TestJudgePrompt(unittest.TestCase):
    def test_prompt_contains_both_texts(self):
        # indirect: build the prompt exactly as judge_summary does
        prompt = bench.JUDGE_PROMPT.format(transcript="TRANS9",
                                           summary="SUMM9")
        self.assertIn("TRANS9", prompt)
        self.assertIn("SUMM9", prompt)
        for dim in JUDGE_DIMS:
            self.assertIn(dim, prompt)


# --------------------------------------------------------------------------- #
# scoring / aggregation / report
# --------------------------------------------------------------------------- #

def result(variant, scores, failures=(), words=100):
    return {"case": "c", "kind": "overview", "variant": variant,
            "words": words, "summary": "s",
            "auto_failures": list(failures), "auto_total": 7,
            "scores": scores}


SCORES_HI = {"faithfulness": 5, "actionability": 5, "clarity": 5,
             "kind": "overview", "notes": ""}
SCORES_LO = {"faithfulness": 3, "actionability": 3, "clarity": 3,
             "kind": "overview", "notes": ""}


class TestAggregate(unittest.TestCase):
    def test_overall_is_mean_minus_auto_penalty(self):
        self.assertEqual(overall_score(result("a", SCORES_HI)), 5.0)
        self.assertAlmostEqual(
            overall_score(result("a", SCORES_HI, failures=["x"])),
            5.0 - bench.AUTO_PENALTY)
        self.assertIsNone(overall_score(result("a", None)))

    def test_ranking_best_first(self):
        rows = aggregate([
            result("lo", SCORES_LO),
            result("hi", SCORES_HI),
            result("mid", SCORES_LO, failures=[]),
        ])
        self.assertEqual([r["variant"] for r in rows], ["hi", "lo", "mid"])
        self.assertEqual(rows[0]["overall"], 5.0)
        self.assertEqual(rows[0]["auto_total"], 7)

    def test_auto_failures_break_ties(self):
        rows = aggregate([
            result("dirty", SCORES_HI, failures=["a", "b"]),
            result("clean", SCORES_HI),
        ])
        self.assertEqual([r["variant"] for r in rows], ["clean", "dirty"])

    def test_unjudged_variants_sort_last(self):
        rows = aggregate([result("dead", None), result("alive", SCORES_LO)])
        self.assertEqual(rows[0]["variant"], "alive")
        self.assertIsNone(rows[1]["overall"])

    def test_report_renders_rows_and_promote_hint(self):
        results = [result("lo", SCORES_LO), result("hi", SCORES_HI)]
        rows = aggregate(results)
        meta = {"timestamp": "t", "n_cases": 1, "set_dir": "s",
                "set_name": "default", "model": "m", "judge_model": "j",
                "max_words": 500, "active": "lo"}
        report = render_report(rows, results, meta)
        self.assertIn("| hi ", report)
        self.assertIn("| lo **(active)**", report)
        self.assertIn("Winner: `hi`", report)
        self.assertIn("yt-pipeline prompts promote hi", report)

    def test_report_notes_when_active_already_wins(self):
        results = [result("lo", SCORES_HI), result("hi", SCORES_LO)]
        rows = aggregate(results)
        meta = {"timestamp": "t", "n_cases": 1, "set_dir": "s",
                "set_name": "default", "model": "m", "judge_model": "j",
                "max_words": 500, "active": "lo"}
        report = render_report(rows, results, meta)
        self.assertIn("is also the best", report)
        self.assertNotIn("Winner", report)

    def test_write_report_artifacts(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            results = [result("hi", SCORES_HI), result("lo", SCORES_LO)]
            rows = aggregate(results)
            meta = {"timestamp": "t", "n_cases": 1, "set_dir": "s",
                    "set_name": "default", "model": "m", "judge_model": "j",
                    "max_words": 500, "active": "hi"}
            out = Path(tmp) / "run1"
            write_report(out, rows, results, meta,
                         {("hi", "c"): "summary text", ("lo", "c"): "other"})
            self.assertTrue((out / "report.md").exists())
            data = json.loads((out / "results.json").read_text())
            self.assertEqual([v["variant"] for v in data["variants"]],
                             ["hi", "lo"])
            self.assertNotIn("summary", data["results"][0])  # not dumped
            self.assertTrue((out / "summaries" / "hi" / "c.md")
                            .read_text().startswith("summary text"))


# --------------------------------------------------------------------------- #
# the whole benchmark, against the fake Ollama HTTP server
# --------------------------------------------------------------------------- #

class TestRunBenchmark(unittest.TestCase):
    """A full deterministic benchmark over a temp prompts dir with two
    variants; the fake Ollama answers summaries and judge JSON via a
    responder keyed on the request text."""

    TIGHT_MARK = "TIGHT-VARIANT-MARKER"

    @classmethod
    def setUpClass(cls):
        cls.server = FakeOllamaServer()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-bm-")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.workdir = self.tmp / "workdir"
        self.workdir.mkdir()
        self.make_variants()
        self.server.reset()
        self.server.responder = self.responder
        self.args = ns(
            model="qwen3.6", judge_model="qwen3.6", host=self.server.host,
            timeout=30, no_autostart=True, think=False, max_words=500,
            chunk_words=1500, notes_words=220, max_num_ctx=32768,
            prompts_dir=str(self.tmp / "prompts"), prompts=None,
            set=str(REPO / "benchmarks" / "set.json"),
            out=str(self.tmp / "results"), min_overall=None,
            workdir=str(self.workdir))

    def make_variants(self):
        base = json.loads((REPO / "prompts" / "v1-baseline.json")
                          .read_text(encoding="utf-8"))
        plain = dict(base, name="plain")
        tight = dict(base, name="tight",
                     description="tighter variant for tests",
                     prompts=dict(base["prompts"],
                                  final=base["prompts"]["final"]
                                  + f"\n\n({self.TIGHT_MARK})"))
        pdir = self.tmp / "prompts"
        pdir.mkdir()
        (pdir / "plain.json").write_text(json.dumps(plain, indent=2))
        (pdir / "tight.json").write_text(json.dumps(tight, indent=2))
        (pdir / "active.txt").write_text("plain\n")

    def responder(self, body):
        user = body["messages"][-1]["content"]
        if "Rate the SUMMARY" in user:  # judge call: the tight variant's
            # summaries carry its marker into the judged text
            hi = self.TIGHT_MARK in user
            scores = (5, 5, 5) if hi else (3, 3, 3)
            return json.dumps({"faithfulness": scores[0],
                               "actionability": scores[1],
                               "clarity": scores[2],
                               "kind": "overview", "notes": "stub"})
        self.tight_output = self.TIGHT_MARK in user
        if "certbot" in user:  # the actionable case
            return GOOD_ACTIONABLE + (
                f"\n\n_{self.TIGHT_MARK}_" if self.tight_output else "")
        return GOOD_OVERVIEW + (
            f"\n\n_{self.TIGHT_MARK}_" if self.tight_output else "")

    def run_bm(self, **over):
        args = SimpleNamespace(**{**vars(self.args), **over})
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = run_benchmark(args)
        return rc, out.getvalue(), err.getvalue()

    def test_full_run_reports_and_ranks(self):
        self.tight_output = False
        rc, out, err = self.run_bm()
        self.assertEqual(rc, 0)
        self.assertIn("| tight", out)
        self.assertIn("| plain **(active)**", out)
        self.assertIn("Winner: `tight`", out)
        self.assertIn("yt-pipeline prompts promote tight", out)
        results_dir = self.tmp / "results"
        self.assertTrue((results_dir / "report.md").exists())
        data = json.loads((results_dir / "results.json").read_text())
        self.assertEqual([v["variant"] for v in data["variants"]],
                         ["tight", "plain"])
        self.assertEqual(data["active"], "plain")
        # 2 variants x 3 cases: summaries saved per variant/case
        for v in ("tight", "plain"):
            for c in ("sample-short", "howto-tls", "overview-history"):
                self.assertTrue((results_dir / "summaries" / v / f"{c}.md")
                                .exists())
        # every summary passed the automatic checks in this stubbed run
        for row in data["variants"]:
            self.assertEqual(row["auto_failed"], 0)

    def test_selected_variants_only(self):
        self.tight_output = False
        rc, out, _ = self.run_bm(prompts="plain")
        self.assertEqual(rc, 0)
        self.assertNotIn("| tight", out)
        self.assertIn("| plain", out)

    def test_unknown_variant_dies(self):
        from tests.helpers import assert_dies
        # run directly (no capture): die() must escape to the test runner
        self.tight_output = False
        args = SimpleNamespace(**{**vars(self.args), "prompts": "nope"})
        assert_dies(self, lambda: run_benchmark(args),
                    pattern="unknown prompt variant")

    def test_min_overall_gate_fails_when_active_is_weak(self):
        self.tight_output = False
        rc, out, err = self.run_bm(min_overall=3.9)  # active 'plain' scores 3
        self.assertEqual(rc, 1)
        self.assertIn("below the required 3.90", err)

    def test_min_overall_gate_passes_for_strong_active(self):
        (self.tmp / "prompts" / "active.txt").write_text("tight\n")
        self.tight_output = False
        rc, _, _ = self.run_bm(min_overall=4.5)
        self.assertEqual(rc, 0)

    def test_min_overall_gate_fails_when_active_not_run(self):
        from tests.helpers import assert_dies
        self.tight_output = False
        args = SimpleNamespace(**{**vars(self.args), "prompts": "tight",
                                  "min_overall": 3.0})
        assert_dies(self, lambda: run_benchmark(args),
                    pattern="was not part of this run")


# --------------------------------------------------------------------------- #
# deterministic quality guard for the default suite
# --------------------------------------------------------------------------- #

class TestQualityGuard(unittest.TestCase):
    """The automatic half of the benchmark, run on every `make test`: the
    ACTIVE prompt variant, summarized here with a deterministic stub model,
    must produce summaries that pass every automatic check on every case of
    the benchmark set. Catches format regressions (word cap, heading
    structure, kind selection, lost facts) alongside the other tests; the
    LLM-judged half is `make bench-real`."""

    def test_active_variant_passes_auto_checks_on_every_case(self):
        from ytpipeline.summarize import summarize_transcript
        bs = load_benchset(None)
        pset = P.load_active(P.resolve_prompts_dir(None, None))
        opts = ns(out=None, model="stub", host="x", timeout=30,
                  no_autostart=True, think=False, max_words=500, lang=None,
                  chunk_words=0, notes_words=220, max_num_ctx=32768,
                  force=False, stdout=False, prompts=pset)
        for c in bs["cases"]:
            canned = GOOD_ACTIONABLE if c["kind"] == "actionable" \
                else GOOD_OVERVIEW + " The number 42 appears."
            stub = StubOllama([canned])
            _, info = summarize_transcript(stub, c["path"], opts)
            self.assertEqual(info["prompt"], pset.name)
            self.assertEqual(info["summary"], canned)  # no mangling
            checks = auto_checks(c, info["summary"], 500)
            self.assertEqual(checks["failures"], [],
                             f"{c['name']}: {checks['failures']}")


if __name__ == "__main__":
    unittest.main()
