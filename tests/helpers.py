"""Shared helpers for the yt-pipeline tests.

- path constants (repo root, fixtures, fake-tool bin dir)
- stderr-capturing assert helpers for code that calls die()/log()/warn()
- FakeOllamaServer: an in-process HTTP server speaking the small part of the
  Ollama API the pipeline uses (version/tags/chat), for unit and E2E tests
- subprocess helpers to run the real CLI (`python3 -m ytpipeline`) with the
  fake yt-dlp/whisper-cli/ffmpeg/ollama binaries shadowing PATH
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent
FIXTURES = TESTS / "fixtures"
FAKE_BIN = TESTS / "fakes"

# Env vars that change the tool's behavior; cleared for every CLI run so
# tests are deterministic regardless of the developer's environment.
LEAKY_ENV_VARS = ("YT_WORKDIR", "YT_TRANSCRIPTS_DIR", "YT_JOBS",
                  "OLLAMA_HOST", "OLLAMA_MODEL")


# --------------------------------------------------------------------------- #
# small in-process helpers
# --------------------------------------------------------------------------- #

def ns(**kwargs):
    """SimpleNamespace builder for the opts/args objects the code passes."""
    return SimpleNamespace(**kwargs)


def stderr_of(fn):
    """Run fn() capturing stderr; return (stderr_text, exception_or_None)."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        try:
            fn()
            return buf.getvalue(), None
        except BaseException as e:  # test helper: keep whatever escapes
            return buf.getvalue(), e


def assert_dies(testcase, fn, pattern=None, code=1):
    """Assert fn() exits through die(); return the captured stderr text."""
    err, exc = stderr_of(fn)
    testcase.assertIsInstance(
        exc, SystemExit, f"expected SystemExit, got {exc!r}; stderr:\n{err}")
    testcase.assertEqual(exc.code, code)
    if pattern:
        testcase.assertRegex(err, pattern)
    return err


def capture(fn):
    """Run fn() with stderr swallowed (the tool logs a lot); return
    (fn_result, stderr_text). Exceptions propagate to the caller."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        result = fn()
    return result, buf.getvalue()


# --------------------------------------------------------------------------- #
# fake Ollama HTTP server (unit + E2E)
# --------------------------------------------------------------------------- #

class _FakeOllamaHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence request logging
        pass

    @property
    def fake(self):
        return self.server.fake

    def _json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/version":
            self._json({"version": "0.0.0-fake"})
        elif path == "/api/tags":
            self._json({"models": [{"name": n} for n in self.fake.models]})
        else:
            self._json({"error": f"no such route: {path}"}, status=404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/api/chat":
            self._json({"error": f"no such route: {path}"}, status=404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, status=400)
            return
        with self.fake.lock:
            self.fake.chat_requests.append(body)
            fail = self.fake.fail_queue.pop(0) if self.fake.fail_queue else None
            content = self.fake.next_content()
        if fail is not None:
            self._json(fail.get("body", {"error": "boom"}),
                       status=fail.get("status", 500))
            return
        self._json({
            "model": body.get("model", "?"),
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 17,
            "eval_count": 42,
            "eval_duration": 42_000_000_000,
        })


class FakeOllamaServer:
    """In-process HTTP server that answers the Ollama API calls yt-pipeline
    makes. Test customization:

      .models           model names reported by /api/tags
      .contents         queue of /api/chat response contents (popped in order;
                        the last entry sticks, and .default_content is used
                        when empty)
      .fail_queue       queue of {"status": int, "body": dict} failures to
                        return for /api/chat before succeeding again
      .chat_requests    every /api/chat request body, in order
    """

    def __init__(self, models=("qwen3.6",)):
        self.lock = threading.Lock()
        self.models = list(models)
        self.chat_requests = []
        self.contents = []
        self.fail_queue = []
        self.default_content = ("## Overview\n\nCanned overview: the fixture "
                                "video shows the pipeline working end to end.")
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
        self._httpd.fake = self
        self.host = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True)
        self._thread.start()

    def next_content(self):
        if not self.contents:
            return self.default_content
        if len(self.contents) == 1:
            return self.contents[0]
        return self.contents.pop(0)

    def reset(self):
        with self.lock:
            self.chat_requests.clear()
            self.contents.clear()
            self.fail_queue.clear()

    def stop(self):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# running the real CLI in a subprocess (E2E)
# --------------------------------------------------------------------------- #

def fake_env(workdir, extra_path=None, no_system_path=False, only=None,
             **overrides):
    """Environment for a fake-tools CLI run.

    The tests/fakes directory is put first on PATH so the fake yt-dlp,
    whisper-cli, ffmpeg and ollama shadow any real ones. only=("yt-dlp",)
    shadows just the named tools (used by tests/real to mix a fake
    yt-dlp with the real whisper-cli). no_system_path=True drops the rest
    of PATH entirely (for "tools missing" scenarios: pass
    extra_path=<an empty dir>).
    """
    env = dict(os.environ)
    for var in LEAKY_ENV_VARS:
        env.pop(var, None)
    if only is None:
        prefix = str(FAKE_BIN)
    else:
        bindir = Path(workdir) / "fakebin"
        bindir.mkdir(parents=True, exist_ok=True)
        for tool in only:
            dst = bindir / tool
            if not dst.exists():
                dst.symlink_to(FAKE_BIN / tool)
        prefix = str(bindir)
    if no_system_path:
        env["PATH"] = str(extra_path) if extra_path else "/nonexistent"
    else:
        parts = [prefix]
        if extra_path:
            parts.append(str(extra_path))
        if os.environ.get("PATH"):
            parts.append(os.environ["PATH"])
        env["PATH"] = os.pathsep.join(parts)
    env["YT_FAKE_LOG"] = str(Path(workdir) / "fake-tool-calls.log")
    env["YT_FAKE_FIXTURES"] = str(FIXTURES)
    env.update(overrides)
    return env


def tool_log(workdir):
    """Parsed fake-tool call log: [{tool, argv}, ...] in call order."""
    path = Path(workdir) / "fake-tool-calls.log"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def run_cli(args, env, timeout=180):
    """Run `python3 -m ytpipeline ARGS` from the repo root."""
    return subprocess.run(
        [sys.executable, "-m", "ytpipeline", *args],
        cwd=str(REPO), env=env, capture_output=True, text=True,
        errors="replace", timeout=timeout)


class TempWorkdirMixin(unittest.TestCase):
    """TestCase with a self.workdir (tmp) recreated per test."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="ytp-test-")
        self.addCleanup(self._tmp.cleanup)
        self.workdir = Path(self._tmp.name)
