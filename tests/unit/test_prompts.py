"""Unit tests for ytpipeline.prompts: variant file validation, loading,
style metadata (the --style framework) and the shipped prompt files."""

import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import assert_dies

from ytpipeline import prompts as P
from ytpipeline.prompts import (PromptSet, load_variant, styles_in,
                                validate_variant, variant_names, write_variant)

REPO_PROMPTS = Path(__file__).resolve().parent.parent.parent / "prompts"


def base_variant(**over):
    """A minimal valid variant file payload."""
    data = {
        "name": "t-variant",
        "version": 1,
        "description": "a test variant",
        "prompts": {
            "system": "sys",
            "final": "final {transcript} {lang_instr} {max_words}",
            "notes": "notes {i} {n} {lang_instr} {notes_words} {chunk}",
            "reduce": "reduce {lang_instr} {max_words} {notes}",
            "condense": "condense {actual} {max_words} {text}",
        },
    }
    data.update(over)
    return data


class TestValidateVariant(unittest.TestCase):
    def test_minimal_variant_is_valid(self):
        self.assertEqual(validate_variant(base_variant()), base_variant())

    def test_style_block_accepted(self):
        data = base_variant(style={"label": "Mini", "summary": "mini style",
                                   "max_words": 42,
                                   "fallback_heading": "## Mini"})
        self.assertEqual(validate_variant(data), data)

    def test_style_block_may_be_partial(self):
        data = base_variant(style={"max_words": 300})
        self.assertEqual(validate_variant(data), data)

    def test_style_partial_fields_still_checked(self):
        bad = [
            ("'style' must be an object", base_variant(style=["nope"])),
            ("style 'label'", base_variant(style={"label": "  "})),
            ("style 'max_words'", base_variant(style={"max_words": 0})),
            ("style 'max_words'", base_variant(style={"max_words": "500"})),
            ("style 'max_words'", base_variant(style={"max_words": True})),
            ("style 'fallback_heading'",
             base_variant(style={"fallback_heading": ""})),
        ]
        for pattern, data in bad:
            with self.subTest(pattern=pattern, data=data.get("style")):
                with self.assertRaises(ValueError) as cm:
                    validate_variant(data)
                self.assertIn(pattern, str(cm.exception))

    def test_missing_keys_rejected(self):
        for key in ("name", "version", "description", "prompts"):
            data = base_variant()
            del data[key]
            with self.assertRaises(ValueError) as cm:
                validate_variant(data)
            self.assertIn(key, str(cm.exception))


class TestPromptSetStyleDefaults(unittest.TestCase):
    def make(self, style=None):
        return PromptSet("t", 1, "d", "", base_variant()["prompts"],
                         style=style)

    def test_plain_variant_defaults(self):
        ps = self.make(None)
        self.assertFalse(ps.is_style)
        self.assertEqual(ps.style_label, "t")       # falls back to the name
        self.assertIsNone(ps.max_words)             # -> tool default (500)
        self.assertEqual(ps.fallback_heading, "## Overview")
        self.assertNotIn("style", ps.as_dict())     # round trip stays clean

    def test_style_variant_metadata(self):
        ps = self.make({"label": "Mini", "max_words": 42,
                        "fallback_heading": "## Mini"})
        self.assertTrue(ps.is_style)
        self.assertEqual(ps.style_label, "Mini")
        self.assertEqual(ps.max_words, 42)
        self.assertEqual(ps.fallback_heading, "## Mini")
        self.assertEqual(ps.as_dict()["style"]["max_words"], 42)

    def test_as_dict_round_trip_preserves_style(self):
        data = base_variant(style={"label": "Mini", "max_words": 42})
        ps = PromptSet(data["name"], 1, "d", "", data["prompts"],
                       style=data["style"])
        again = ps.as_dict()
        self.assertEqual(again["style"], data["style"])
        validate_variant(again)  # still a valid variant file


class TestStylesIn(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-styles-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        write_variant(self.dir, base_variant(name="plain"))
        write_variant(self.dir, base_variant(name="styled",
                                             style={"label": "Styled"}))

    def test_only_style_variants_listed(self):
        styled = styles_in(self.dir)
        self.assertEqual([s.name for s in styled], ["styled"])
        self.assertEqual(styled[0].style_label, "Styled")

    def test_variant_names_lists_everything(self):
        self.assertEqual(variant_names(self.dir), ["plain", "styled"])


class TestShippedPromptFiles(unittest.TestCase):
    """The *.json files shipped in prompts/ must all load and validate, and
    the three documented styles must carry their documented metadata."""

    STYLES = {  # name -> (label, word cap, fallback heading)
        "actionable": ("Actionable", 500, "## Actionable summary"),
        "short": ("Short", 150, "## Overview"),
        "study-guide": ("Study guide", 700, "## Key concepts"),
    }

    def test_all_shipped_files_validate(self):
        files = sorted(REPO_PROMPTS.glob("*.json"))
        self.assertGreaterEqual(len(files), 4)
        for f in files:
            with self.subTest(file=f.name):
                data = json.loads(f.read_text(encoding="utf-8"))
                validate_variant(data, source=f.name)
                self.assertEqual(data["name"], f.stem)

    def test_shipped_styles_present_with_metadata(self):
        for name, (label, cap, heading) in self.STYLES.items():
            with self.subTest(style=name):
                ps = load_variant(REPO_PROMPTS, name)
                self.assertTrue(ps.is_style, f"{name} lacks a style block")
                self.assertEqual(ps.style_label, label)
                self.assertEqual(ps.max_words, cap)
                self.assertEqual(ps.fallback_heading, heading)

    def test_baseline_is_not_a_style(self):
        ps = load_variant(REPO_PROMPTS, "v1-baseline")
        self.assertFalse(ps.is_style)
        self.assertIsNone(ps.max_words)

    def test_shipped_styles_in_lists_them(self):
        names = [s.name for s in styles_in(REPO_PROMPTS)]
        self.assertEqual(names, sorted(self.STYLES))


class TestNewVariantKeepsStyle(unittest.TestCase):
    """'prompts new' copies via as_dict(); a style copy stays a style."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-newstyle-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        write_variant(self.dir, base_variant(name="styled",
                                             style={"label": "Styled",
                                                    "max_words": 42}))

    def test_copy_preserves_style_block(self):
        P.new_variant(self.dir, self.dir, "styled2", "styled")
        ps = load_variant(self.dir, "styled2")
        self.assertTrue(ps.is_style)
        self.assertEqual(ps.max_words, 42)

    def test_load_variant_bad_name_dies(self):
        assert_dies(self, lambda: load_variant(self.dir, "Not A Name"),
                    pattern="invalid prompt variant name")


if __name__ == "__main__":
    unittest.main()
