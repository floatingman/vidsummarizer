#!/usr/bin/env python3
"""Render the 'Changes since <tag>' section for the release notes.

Usage: format-changelog.py PREV_TAG            (appends markdown to stdout)

Prints every commit between PREV_TAG and HEAD as a bullet with its full
message: the subject line, then the body as indented text (body bullets
become nested bullets). Used by .github/workflows/release.yml; deterministic,
offline, stdlib only.
"""

import subprocess
import sys

SEP = "\x1e"  # record separator between commits
UNIT = "\x1f"  # unit separator between subject / short hash / body


def changelog(prev: str) -> str:
    out = subprocess.run(
        ["git", "log", "--no-merges",
         f"--pretty=format:%s{UNIT}%h{UNIT}%b{SEP}", f"{prev}..HEAD"],
        capture_output=True, text=True, check=True).stdout
    lines = []
    for entry in out.split(SEP):
        entry = entry.strip("\n")
        if not entry.strip():
            continue
        subject, sha, body = entry.split(UNIT, 2)
        lines.append(f"* {subject} ({sha})")
        for line in body.splitlines():
            s = line.strip()
            if not s or s == "---":
                continue
            lines.append(("  " + s) if s.startswith(("- ", "* "))
                         else ("    " + s))
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} PREV_TAG")
    text = changelog(sys.argv[1])
    if text:
        print(text)
