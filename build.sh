#!/usr/bin/env bash
# Build dist/yt-pipeline: a single self-contained executable file.
#
# The executable is a Python "zipapp" (.pyz): the whole ytpipeline package is
# packed into one file with a '#!/usr/bin/env python3' shebang, so it runs
# directly (./yt-pipeline ...) on any machine that has Python 3.9+ installed.
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

VERSION=$("$PYTHON" -c "import sys; sys.path.insert(0, '.'); from ytpipeline import VERSION; print(VERSION)")

rm -rf build dist
mkdir -p build/ytpipeline dist

# Stage only the package (no README, .git, data dirs) into the archive root.
cp ytpipeline/*.py build/ytpipeline/

"$PYTHON" -m zipapp build \
    --python "/usr/bin/env python3" \
    --main "ytpipeline.cli:main" \
    --compress \
    --output "dist/yt-pipeline"

chmod +x dist/yt-pipeline

# Smoke test the freshly built executable.
./dist/yt-pipeline --version
echo "built dist/yt-pipeline (yt-pipeline $VERSION, $(du -h dist/yt-pipeline | cut -f1))"
echo "transfer it to another machine with e.g.:"
echo "  scp dist/yt-pipeline user@host:~/yt-pipeline/   # then: ssh user@host 'cd ~/yt-pipeline && ./yt-pipeline doctor'"
