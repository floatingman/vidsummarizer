#!/usr/bin/env bash
# Build dist/vidsummarizer: a single self-contained executable file.
#
# The executable is a Python "zipapp" (.pyz): the whole vidsummarizer package is
# packed into one file with a '#!/usr/bin/env python3' shebang, so it runs
# directly (./vidsummarizer ...) on any machine that has Python 3.9+ installed.
# Nothing else from this checkout is needed at runtime; external tools
# (yt-dlp, ffmpeg, whisper-cli, ollama) and model files are detected on the
# target machine at runtime.
#
# Usage:  ./build.sh          (or: PYTHON=python3.11 ./build.sh)

set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}

if ! "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "build.sh: need Python 3.9+ (found: $("$PYTHON" --version))" >&2
    exit 1
fi

VERSION=$("$PYTHON" -c "import sys; sys.path.insert(0, '.'); from vidsummarizer import VERSION; print(VERSION)")

rm -rf build dist
mkdir -p build/vidsummarizer/prompts build/vidsummarizer/benchset/transcripts dist

# Stage only the package (no README, .git, data dirs) into the archive root.
cp vidsummarizer/*.py build/vidsummarizer/

# The summarizer's prompt variants and the benchmark set are data of the
# package itself, so the executable stays self-contained.
cp prompts/*.json build/vidsummarizer/prompts/
cp benchmarks/set.json build/vidsummarizer/benchset/
cp benchmarks/transcripts/*.txt build/vidsummarizer/benchset/transcripts/

# Ship the MIT notice inside the one-file distribution: the archive is all a
# recipient gets, and the license requires the notice in all copies.
cp LICENSE build/

"$PYTHON" -m zipapp build \
    --python "/usr/bin/env python3" \
    --main "vidsummarizer.cli:main" \
    --compress \
    --output "dist/vidsummarizer"

chmod +x dist/vidsummarizer

# Smoke test the freshly built executable.
./dist/vidsummarizer --version
./dist/vidsummarizer prompts list >/dev/null
echo "built dist/vidsummarizer (vidsummarizer $VERSION, $(du -h dist/vidsummarizer | cut -f1))"
echo "transfer it to another machine with e.g.:"
echo "  scp dist/vidsummarizer user@host:~/vidsummarizer/   # then: ssh user@host 'cd ~/vidsummarizer && ./vidsummarizer doctor'"
