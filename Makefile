# vidsummarizer test suite (Python stdlib unittest only; see README "Testing").
#
#   make test        the default suite: unit + E2E, seconds, fully offline
#                    and deterministic (external tools replaced by fakes)
#   make test-unit   unit tests only
#   make test-e2e    end-to-end tests only
#   make test-real   opt-in: real whisper-cli + local model transcribe the
#                    bundled speech fixture (no network)
#   make test-live   opt-in: a real 19s YouTube video through the real
#                    pipeline (needs network + model)
#   make bench-real  opt-in: real local LLM benchmark of the prompt variants
#                    (needs ollama + model; quality gate on the active one)
#   make build       build dist/vidsummarizer (zipapp)

PYTHON ?= python3

.PHONY: test test-unit test-e2e test-real test-live bench-real build

test: test-unit test-e2e

test-unit:
	$(PYTHON) -m unittest discover -s tests/unit -t .

test-e2e:
	$(PYTHON) -m unittest discover -s tests/e2e -t .

test-real:
	YT_TEST_REAL=1 $(PYTHON) -m unittest discover -s tests/real -t .

test-live:
	YT_TEST_REAL=1 YT_E2E_LIVE=1 $(PYTHON) -m unittest discover -s tests/real -t .

bench-real:
	$(PYTHON) -m vidsummarizer benchmark --min-overall 3.5

build:
	./build.sh
