"""Opt-in tests against the REAL external tools (no fakes where it matters).

These are excluded from `make test` and run via `make test-real`:

  YT_TEST_REAL=1 python3 -m unittest discover -s tests/real -t .

- test_real_whisper_transcribe: the real whisper-cli + a real whisper GGML
  model transcribe the bundled 7s speech fixture (tests/fixtures/audio).
  yt-dlp stays fake (canned metadata + copying the fixture mp3), so no
  network is touched. Requires: whisper-cli on PATH and a model at
  $YT_TEST_WHISPER_MODEL or dist/models/ggml-small.bin.

- test_live (YT_E2E_LIVE=1 additionally, `make test-live`): the real
  yt-dlp downloads a real 19-second YouTube video ("Me at the zoo") and the
  real pipeline transcribes it. Needs network access and the model.
"""

import os
import shutil
import unittest
from pathlib import Path

from tests.helpers import FIXTURES, REPO, fake_env, run_cli

VID = "dQw4w9WgXcQ"
OPT_IN = os.environ.get("YT_TEST_REAL") == "1"
LIVE = OPT_IN and os.environ.get("YT_E2E_LIVE") == "1"


def find_whisper_model():
    """A real whisper GGML model, or None (skip)."""
    cand = [os.environ.get("YT_TEST_WHISPER_MODEL"),
            str(REPO / "dist" / "models" / "ggml-small.bin"),
            os.path.expanduser("~/.cache/whisper/ggml-small.bin")]
    for c in cand:
        try:
            if c and Path(c).stat().st_size > 10_000_000:
                return c
        except OSError:
            continue
    return None


@unittest.skipUnless(OPT_IN, "opt-in: run with make test-real (YT_TEST_REAL=1)")
class TestRealWhisper(unittest.TestCase):
    def setUp(self):
        if not shutil.which("whisper-cli"):
            self.skipTest("whisper-cli not on PATH")
        self.model = find_whisper_model()
        if not self.model:
            self.skipTest("no whisper GGML model found "
                          "(set YT_TEST_WHISPER_MODEL or put one in "
                          "dist/models/ggml-small.bin)")
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-real-")
        self.addCleanup(self._tmp.cleanup)
        self.workdir = Path(self._tmp.name)

    def test_real_whisper_transcribes_the_speech_fixture(self):
        # fake yt-dlp (canned metadata + the bundled mp3), REAL whisper-cli
        env = fake_env(self.workdir, only=("yt-dlp",))
        r = run_cli(["transcribe", VID, "-j", "1",
                     "-m", self.model, "-t", "4",
                     "--workdir", str(self.workdir)], env, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr)
        # title-based naming: the transcript is named after the video title
        files = list((self.workdir / "transcripts").glob("*.txt"))
        self.assertEqual(len(files), 1)
        out = files[0]
        self.assertIn(f"video_id: {VID}", out.read_text(encoding="utf-8"))
        text = out.read_text(encoding="utf-8")
        # real transcription of the real speech fixture: substantial body
        header, body = text.split("\n\n", 1)
        self.assertRegex(header, r"(?m)^language: \w+")
        self.assertGreaterEqual(len(body.split()), 8)
        # the fixture speech says these words; whisper should find most
        lowered = body.lower()
        hits = sum(w in lowered for w in ("hello", "test", "recording",
                                          "pipeline", "fox", "dog"))
        self.assertGreaterEqual(hits, 3,
                                f"transcript did not contain the expected "
                                f"speech:\n{text}")


@unittest.skipUnless(LIVE, "network test: run with make test-live "
                           "(YT_E2E_LIVE=1)")
class TestLiveYouTube(unittest.TestCase):
    """A real 19-second video through fetch + transcribe (live network)."""

    LIVE_VID = "jNQXAC9IVRw"  # "Me at the zoo" (19s)

    def setUp(self):
        if not shutil.which("yt-dlp"):
            self.skipTest("yt-dlp not on PATH")
        if not shutil.which("whisper-cli"):
            self.skipTest("whisper-cli not on PATH")
        self.model = find_whisper_model()
        if not self.model:
            self.skipTest("no whisper GGML model found")
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-live-")
        self.addCleanup(self._tmp.cleanup)
        self.workdir = Path(self._tmp.name)

    def test_live_fetch_and_transcribe(self):
        env = dict(os.environ)  # everything real
        for var in ("YT_WORKDIR", "YT_TRANSCRIPTS_DIR", "YT_JOBS"):
            env.pop(var, None)
        r = run_cli(["transcribe", self.LIVE_VID, "-j", "1",
                     "-m", self.model, "--timestamps",
                     "--workdir", str(self.workdir)], env, timeout=900)
        self.assertEqual(r.returncode, 0, r.stderr)
        # the real video's real title names the file; find it by its header
        files = list((self.workdir / "transcripts").glob("*.txt"))
        self.assertEqual(len(files), 1)
        text = files[0].read_text(encoding="utf-8")
        self.assertIn(f"video_id: {self.LIVE_VID}", text)
        self.assertIn("title:", text)
        self.assertRegex(text, r"\[\d\d:\d\d\] ")  # timestamps requested
        self.assertGreaterEqual(len(text.split()), 10)


if __name__ == "__main__":
    unittest.main()
