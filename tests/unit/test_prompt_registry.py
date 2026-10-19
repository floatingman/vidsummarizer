"""Registry tests for the prompt-variant artifacts: directory resolution,
the active.txt pointer, promotion/materialization flows, and (regression
guard) that the shipped v1-baseline is byte-identical to the prompts
yt-summarize historically hardcoded in summarize.py."""

import os
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import assert_dies

from vidsummarizer import prompts as P

REPO_PROMPTS = Path(__file__).resolve().parent.parent.parent / "prompts"


# --- regression guard ------------------------------------------------------ #
# The exact prompt wording of yt-summarize 1.0.0 / vidsummarizer 2.0-2.1, kept
# here as literals. v1-baseline.json must stay byte-identical to it; editing
# the baseline is a deliberate act that updates this test too.
ORIG_SYSTEM = 'You are a careful, precise summarizer for a personal note-taking app. You only state facts found in the transcript; you never invent details, numbers, names or claims. You write clean, compact markdown.'

ORIG_FINAL = 'Summarize the video transcript below for someone who has not watched it.\n\nDecide what kind of video it is:\n- If the transcript contains ACTIONABLE information (steps, instructions, how-to advice, concrete recommendations the viewer can act on), the summary must capture that actionable content: a short context paragraph, then a numbered list "### Actionable steps" where each item is a concrete step, recommendation or decision the viewer can take, with the details needed to act on it (tools, dosages, settings, thresholds, caveats mentioned in the video).\n- Otherwise, give an OVERVIEW of the content: 2-4 short paragraphs covering the main points and their meaning.\n\nHard rules:\n- {lang_instr}\n- The WHOLE answer must be at most {max_words} words; prefer fewer.\n- Start with exactly one top heading line, nothing before it:\n  "## Actionable summary" for an actionable video, or "## Overview" otherwise.\n  You may add "### Actionable steps" (and at most one other "###" subheading) below it.\n- No preamble, no closing remarks, no title of the video, no horizontal rules, no code fences.\n- Do not use headings above "##". Plain markdown only (bold, lists).\n- Ground every statement in the transcript. If the video presents claims as the speaker\'s opinion, keep that framing.\n- If the transcript contains [MM:SS] timestamps you may cite the most relevant ones sparingly, like "(12:34)".\n\nTranscript:\n\n{transcript}'

ORIG_NOTES = 'You are given part {i} of {n} of a longer video transcript. Extract compact notes from it that a summarizer can use later.\n\nCapture, as a plain markdown bullet list (no headings):\n- the main points made in this part;\n- any actionable steps, instructions or concrete recommendations, with the details needed to act on them (tools, names, numbers, settings, caveats);\n- important facts, definitions or examples worth keeping.\n\nRules:\n- {lang_instr}\n- At most {notes_words} words.\n- Do not add anything not in this part of the transcript.\n\nTranscript part {i} of {n}:\n\n{chunk}'

ORIG_REDUCE = 'Below are notes taken from the parts of a video transcript (the notes appear in order). Turn them into the final summary for someone who has not watched the video.\n\nDecide what kind of video it is:\n- If the notes contain ACTIONABLE information (steps, instructions, how-to advice, concrete recommendations the viewer can act on), the summary must capture that actionable content: a short context paragraph, then a numbered list "### Actionable steps" where each item is a concrete step, recommendation or decision the viewer can take, with the details needed to act on it.\n- Otherwise, give an OVERVIEW of the content: 2-4 short paragraphs covering the main points and their meaning.\n\nHard rules:\n- {lang_instr}\n- The WHOLE answer must be at most {max_words} words; prefer fewer.\n- Deduplicate: the notes repeat things across parts; each point appears once.\n- Start with exactly one top heading line, nothing before it:\n  "## Actionable summary" for an actionable video, or "## Overview" otherwise.\n  You may add "### Actionable steps" (and at most one other "###" subheading) below it.\n- No preamble, no closing remarks, no video title, no horizontal rules, no code fences.\n- Ground every statement in the notes.\n\nNotes:\n\n{notes}'

ORIG_CONDENSE = 'Your answer below is {actual} words, over the hard limit of {max_words} words. Rewrite it to at most {max_words} words (prefer clearly fewer), keeping the same language, the same first "##" heading, the same structure and the same key content, just tighter. Output only the rewritten markdown, nothing else.\n\nAnswer:\n\n{text}'

ORIG = {"system": ORIG_SYSTEM, "final": ORIG_FINAL, "notes": ORIG_NOTES,
        "reduce": ORIG_REDUCE, "condense": ORIG_CONDENSE}


class TestShippedBaseline(unittest.TestCase):
    def test_repo_prompts_dir_found_by_default_resolution(self):
        d = P.resolve_prompts_dir(None, None)
        self.assertIn("v1-baseline", P.variant_names(d))

    def test_v1_baseline_is_the_historical_wording(self):
        pset = P.load_variant(P.resolve_prompts_dir(None, None), "v1-baseline")
        self.assertFalse(pset.is_style)  # the baseline is not a style
        for key, orig in ORIG.items():
            self.assertEqual(getattr(pset, key), orig,
                             f"v1-baseline '{key}' drifted from the "
                             "historical prompt wording")

    def test_default_active_is_v1_baseline_without_pointer_file(self):
        d = P.resolve_prompts_dir(None, None)
        self.assertEqual(P.active_name(d), P.DEFAULT_VARIANT)

    def test_baseline_templates_format_with_real_values(self):
        pset = P.load_variant(P.resolve_prompts_dir(None, None), "v1-baseline")
        self.assertIn("Transcript", pset.final.format(
            lang_instr="Write the summary in English.", max_words=500,
            transcript="Hello world."))
        pset.notes.format(i=1, n=2, lang_instr="x", notes_words=220,
                          chunk="text")
        pset.reduce.format(lang_instr="x", max_words=500, notes="n")
        pset.condense.format(actual=600, max_words=500, text="t")


class TestDirResolution(unittest.TestCase):
    def test_explicit_dir_used(self):
        d = P.resolve_prompts_dir(str(REPO_PROMPTS), None)
        self.assertIn("v1-baseline", P.variant_names(d))

    def test_explicit_dir_without_variants_dies(self):
        import tempfile
        with tempfile.TemporaryDirectory() as empty:
            assert_dies(self, lambda: P.resolve_prompts_dir(empty, None),
                        pattern="--prompts-dir")

    def test_env_var_used(self):
        with mock.patch.dict(os.environ, {"YT_PROMPTS_DIR": str(REPO_PROMPTS)}):
            d = P.resolve_prompts_dir(None, None)
            self.assertIn("v1-baseline", P.variant_names(d))

    def test_env_var_missing_dir_dies(self):
        with mock.patch.dict(os.environ, {"YT_PROMPTS_DIR": "/no/such/dir"}):
            assert_dies(self, lambda: P.resolve_prompts_dir(None, None),
                        pattern=r"\$YT_PROMPTS_DIR")

    def test_workdir_prompts_preferred_when_present(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp) / "prompts"
            pdir.mkdir()
            (pdir / "w.json").write_text(
                (REPO_PROMPTS / "v1-baseline.json").read_text()
                .replace('"v1-baseline"', '"w"', 1))
            d = P.resolve_prompts_dir(None, tmp)
            self.assertEqual(P.variant_names(d), ["w"])


class TestActivePointer(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-reg-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.copy = self.dir / "copy.json"
        self.copy.write_text(
            (REPO_PROMPTS / "v1-baseline.json").read_text()
            .replace('"v1-baseline"', '"copy"', 1), encoding="utf-8")

    def test_pointer_round_trip(self):
        self.assertEqual(P.active_name(self.dir), P.DEFAULT_VARIANT)
        P.set_active(self.dir, "copy")
        self.assertEqual(P.active_name(self.dir), "copy")
        self.assertEqual(P.load_active(self.dir).name, "copy")

    def test_pointer_to_missing_variant_dies(self):
        (self.dir / "active.txt").write_text("ghost\n")
        assert_dies(self, lambda: P.load_active(self.dir),
                    pattern="no prompt variant 'ghost'")

    def test_set_active_validates_first(self):
        assert_dies(self, lambda: P.set_active(self.dir, "nope"),
                    pattern="no prompt variant 'nope'")


class TestWritableDir(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-wr-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.workdir = self.dir / "wd"
        self.workdir.mkdir()
        (self.dir / "copy.json").write_text(
            (REPO_PROMPTS / "v1-baseline.json").read_text()
            .replace('"v1-baseline"', '"copy"', 1), encoding="utf-8")

    def test_explicit_dir_is_used_as_is(self):
        w = P.writable_prompts_dir(self.dir, str(self.workdir),
                                   explicit=str(self.dir))
        self.assertEqual(Path(w), self.dir)

    def test_workdir_prompts_used_when_it_has_variants(self):
        pdir = self.workdir / "prompts"
        pdir.mkdir()
        (pdir / "copy.json").write_text(
            (self.dir / "copy.json").read_text(), encoding="utf-8")
        w = P.writable_prompts_dir(self.dir, str(self.workdir))
        self.assertEqual(w, pdir)

    def test_fresh_workdir_materializes_the_read_dir(self):
        # the read dir is writable but not the workdir's own prompts/: a
        # fresh workdir gets its own copy, the read dir stays untouched
        w = P.writable_prompts_dir(self.dir, str(self.workdir))
        self.assertEqual(w, self.workdir / "prompts")
        self.assertEqual(P.variant_names(w), ["copy"])
        P.set_active(w, "copy")
        self.assertEqual(P.load_active(w).name, "copy")
        self.assertFalse((self.dir / "active.txt").exists())

    def test_unreadable_source_materializes_bundled_variants(self):
        # like a path inside the zipapp archive: not readable as a directory
        ghost = self.dir / "packed"
        w = P.writable_prompts_dir(ghost, str(self.workdir))
        self.assertEqual(w, self.workdir / "prompts")
        self.assertIn("v1-baseline", P.variant_names(w))


class TestNewVariant(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-nv-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        (self.dir / "base.json").write_text(
            (REPO_PROMPTS / "v1-baseline.json").read_text()
            .replace('"v1-baseline"', '"base"', 1), encoding="utf-8")

    def test_new_variant_copies_templates(self):
        path = P.new_variant(self.dir, self.dir, "tighter", "base")
        pset = P.load_variant(self.dir, "tighter")
        self.assertEqual(pset.final, P.load_variant(self.dir, "base").final)
        self.assertIn("copied from base", pset.description)
        self.assertTrue(path.is_file())

    def test_new_variant_rejects_same_name(self):
        assert_dies(self, lambda: P.new_variant(self.dir, self.dir,
                                                "base", "base"),
                    pattern="different name")

    def test_new_variant_rejects_bad_names(self):
        assert_dies(self, lambda: P.new_variant(self.dir, self.dir,
                                                "Bad Name", "base"),
                    pattern="invalid variant name")


if __name__ == "__main__":
    unittest.main()
