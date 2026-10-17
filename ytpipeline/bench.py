"""Benchmark the summarizer's prompt variants and compare their scores.

`yt-pipeline benchmark` runs the summarizer (a local model via Ollama, like
the rest of the pipeline) over a fixed set of transcripts with one or more
prompt variants, then scores every produced summary:

- automatic checks (deterministic, offline): word cap, markdown/Obsidian
  structure, the right summary kind for the video, presence of key facts;
- an LLM-as-judge rating by a *local* model against a rubric:
  faithfulness to the transcript, actionability, clarity (1-5 each).

The result is a comparison report (markdown + JSON + the generated
summaries) under <workdir>/benchmarks/results/<timestamp>/, naming the
winner and how to promote it. Promotion itself is a separate, manual step
(`yt-pipeline prompts promote NAME`), by design: the report informs, the
human decides. `--min-overall SCORE` turns a run into a quality gate (used
by `make bench-real`).

The judge prompts are part of the measuring instrument (like a human
reviewer's rubric), not of the thing being measured, so they live here as
constants rather than in the versioned prompt variants.
"""

import importlib.resources
import json
import re
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from . import VERSION
from .common import die, log, resolve_workdir, warn
from .prompts import (active_name, list_variants, load_variant,
                      resolve_prompts_dir, variant_names)
from .summarize import Ollama, count_words, gen_options, parse_transcript

TOOL = f"yt-pipeline {VERSION} (benchmark)"

# Each failed automatic check subtracts AUTO_PENALTY from the case's overall
# score (mean of the three judge dimensions) before averaging per variant.
AUTO_PENALTY = 0.25


# --------------------------------------------------------------------------- #
# The judge (local LLM, strict rubric, JSON answer)
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM = (
    "You are a strict, impartial judge of summaries. You evaluate each "
    "summary only against the transcript it claims to summarize, using the "
    "rubric you are given. You answer with JSON only, no other text."
)

JUDGE_PROMPT = """Rate the SUMMARY of the TRANSCRIPT below on three dimensions, each an integer from 1 to 5:

- faithfulness: 5 = every statement in the summary is supported by the transcript, nothing invented, nothing distorted; 1 = major inventions or contradictions.
- actionability: 5 = the concrete, usable content of the video (steps, instructions, recommendations, tools, numbers, caveats) survives in the summary at a useful level of detail; 1 = the useful content is lost or reduced to vague generalities.
- clarity: 5 = well structured, easy to scan, clean markdown, right length for the material; 1 = rambling, hard to read or badly formatted.

Also report which kind of summary it is.

Answer with JSON only, in exactly this shape, no other text:
{{"faithfulness": 1, "actionability": 1, "clarity": 1, "kind": "actionable" or "overview", "notes": "one short sentence"}}

TRANSCRIPT:

{transcript}

SUMMARY:

{summary}"""

JUDGE_DIMS = ("faithfulness", "actionability", "clarity")


# --------------------------------------------------------------------------- #
# Benchmark set
# --------------------------------------------------------------------------- #

def _packaged_set():
    """The benchmarks/ copy packed next to the package (inside the zipapp)."""
    try:
        d = importlib.resources.files("ytpipeline").joinpath("benchset")
        return d if (d / "set.json").is_file() else None
    except Exception:
        return None


def _workdir_set():
    """<workdir>/benchmarks (repo root in a source checkout)."""
    d = Path(resolve_workdir()) / "benchmarks"
    return d if (d / "set.json").is_file() else None


def resolve_set_dir(explicit=None):
    """--set FILE/DIR > the packed copy (zipapp) > <workdir>/benchmarks."""
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_dir():
            if not (p / "set.json").is_file():
                die(f"--set {explicit}: no set.json in that directory")
            return p
        if p.name != "set.json":
            p = p.parent / "set.json"
        if not p.is_file():
            die(f"--set {explicit}: file not found")
        return p.parent
    for cand in (_packaged_set(), _workdir_set()):
        if cand is not None:
            return cand
    die("no benchmark set found (looked for benchmarks/set.json in the work "
        "directory and inside the executable); pass --set FILE")


def load_benchset(explicit=None):
    """Return {"name", "dir", "cases": [case dicts, paths and bodies filled]}."""
    d = resolve_set_dir(explicit)
    try:
        data = json.loads((d / "set.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        die(f"cannot read benchmark set {d / 'set.json'}: {e}")
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        die(f"{d / 'set.json'}: needs a non-empty 'cases' list")
    cases = []
    for i, case in enumerate(data["cases"], 1):
        for key in ("file", "kind"):
            if key not in case:
                die(f"{d / 'set.json'}: case {i} is missing '{key}'")
        kind = case["kind"]
        if kind not in ("actionable", "overview"):
            die(f"{d / 'set.json'}: case {i}: kind must be 'actionable' or "
                f"'overview', not {kind!r}")
        ref = Path(case["file"])
        if ref.is_absolute():
            path = ref
            if not path.is_file():
                die(f"{d / 'set.json'}: case {i}: transcript not found: {path}")
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError as e:
                die(f"{d / 'set.json'}: case {i}: cannot read {path}: {e}")
        elif isinstance(d, Path):
            path = Path(str(d)) / ref
            if not path.is_file():
                die(f"{d / 'set.json'}: case {i}: transcript not found: {path}")
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError as e:
                die(f"{d / 'set.json'}: case {i}: cannot read {path}: {e}")
        else:
            # the packed set inside the zipapp: d is a Traversable, so
            # Path(str(d)) is not a real filesystem path. Read through
            # the Traversable API, then materialize the transcript so
            # the rest of the run (summarize_transcript) gets a real
            # file it can open.
            try:
                raw = (d / ref.as_posix()).read_text(encoding="utf-8")
            except (OSError, KeyError, ValueError) as e:
                # KeyError: py3.9's zipfile.Path raises it for a missing
                # entry instead of FileNotFoundError
                die(f"{d / 'set.json'}: case {i}: cannot read "
                    f"{case['file']}: {e}")
            tmp = tempfile.mkdtemp(prefix="ytp-benchset-")
            path = Path(tmp) / ref.name
            path.write_text(raw, encoding="utf-8")
        meta, body = parse_transcript(raw)
        if not body:
            die(f"{d / 'set.json'}: case {i}: transcript is empty: {path}")
        cases.append({
            "name": case.get("name") or path.stem,
            "file": case["file"], "path": str(path), "kind": kind,
            "must_contain": [str(t) for t in case.get("must_contain", [])],
            "meta": meta, "body": body, "body_words": count_words(body),
        })
    return {"name": data.get("name", "default"), "dir": d, "cases": cases}


# --------------------------------------------------------------------------- #
# Automatic checks (deterministic, offline)
# --------------------------------------------------------------------------- #

KNOWN_TOP_HEADINGS = ("## Overview", "## Actionable summary")
FENCE_RE = re.compile(r"```")
HR_RE = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", re.M)
H1_RE = re.compile(r"^#\s+\S", re.M)


def auto_checks(case, summary, max_words, expected_heading=None):
    """Deterministic quality checks on one summary. Returns
    {"failures": [msg, ...], "total": int} (total = number of checks run).

    expected_heading overrides the case-kind default for style variants
    whose top heading is neither '## Overview' nor '## Actionable
    summary' (e.g. a study-guide style)."""
    failures = []
    words = count_words(summary)
    if words > max_words:
        failures.append(f"word limit: {words} words > {max_words}")
    first = next((ln.strip() for ln in summary.splitlines() if ln.strip()), "")
    known = KNOWN_TOP_HEADINGS
    if expected_heading and expected_heading not in known:
        known = KNOWN_TOP_HEADINGS + (expected_heading,)
    if not first.startswith("## "):
        failures.append(f"structure: does not start with a '## ' heading "
                        f"(starts with {first[:30]!r})")
    elif first not in known:
        failures.append(f"structure: unknown top heading {first!r}")
    if FENCE_RE.search(summary):
        failures.append("structure: contains a code fence")
    if HR_RE.search(summary):
        failures.append("structure: contains a horizontal rule")
    if H1_RE.search(summary):
        failures.append("structure: contains an H1 '#' heading")
    expected = expected_heading or (
        "## Actionable summary" if case["kind"] == "actionable"
        else "## Overview")
    if first != expected:
        failures.append(f"summary kind: expected {expected!r} for a "
                        f"{case['kind']} video, got {first!r}")
    for term in case.get("must_contain", []):
        if term.lower() not in summary.lower():
            failures.append(f"content: expected fact {term!r} not in summary")
    # 1 word limit + 4 structure + 1 kind + len(must_contain) fact checks
    total = 6 + len(case.get("must_contain", []))
    return {"failures": failures, "total": total}


# --------------------------------------------------------------------------- #
# LLM-as-judge
# --------------------------------------------------------------------------- #

def parse_scores(text):
    """Parse the judge's JSON answer; None when it is unusable."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    scores = {}
    for dim in JUDGE_DIMS:
        if dim not in obj:
            return None
        try:
            scores[dim] = min(5, max(1, int(obj[dim])))
        except (TypeError, ValueError):
            return None
    kind = obj.get("kind", "")
    scores["kind"] = kind if kind in ("actionable", "overview") else None
    scores["notes"] = str(obj.get("notes", ""))[:500]
    return scores


def judge_summary(ollama, case, summary, bopts):
    """Rate one summary with the local judge model (case needs 'body').
    Returns a scores dict (JUDGE_DIMS + kind + notes), or None when the
    judge answer is unusable twice in a row."""
    prompt = JUDGE_PROMPT.format(transcript=case["body"], summary=summary)
    options = gen_options(count_words(prompt), 80, bopts, 0.0)
    for attempt in (1, 2):  # one re-ask; judges sometimes decorate JSON
        content, _ = ollama.chat(JUDGE_SYSTEM, prompt, options, think=False)
        scores = parse_scores(content)
        if scores is not None:
            return scores
        log(f"    judge answer was not valid JSON (attempt {attempt}/2)")
    return None


# --------------------------------------------------------------------------- #
# Running the benchmark
# --------------------------------------------------------------------------- #

def bench_opts_ns(args, workdir):
    """The opts namespace summarize_transcript/gen_options need."""
    return SimpleNamespace(
        out=None, model=args.model, host=args.host, timeout=args.timeout,
        no_autostart=args.no_autostart, think=args.think,
        max_words=args.max_words, lang=None, style=None,
        chunk_words=args.chunk_words,
        notes_words=args.notes_words, max_num_ctx=args.max_num_ctx,
        force=False, stdout=False, prompts_dir=args.prompts_dir,
        workdir=workdir)


def summarize_case(ollama, pset, case, bopts):
    """Summarize one case with one variant pinned; returns (summary, info).
    Without an explicit --max-words the variant's own style cap applies
    (same rule as a normal run), so the check below scores fairly."""
    from .summarize import summarize_transcript
    per_case = SimpleNamespace(**{**vars(bopts), "prompts": pset})
    _, info = summarize_transcript(ollama, case["path"], per_case)
    return info["summary"], info


def run_one(ollama, judge, pset, case, bopts, label):
    """Summarize + score one (variant, case) pair; returns a result dict."""
    from .summarize import DEFAULT_MAX_WORDS
    summary, info = summarize_case(ollama, pset, case, bopts)
    cap = bopts.max_words if bopts.max_words is not None \
        else (pset.max_words or DEFAULT_MAX_WORDS)
    # a style variant with its own top heading is checked against that
    heading = pset.fallback_heading if pset.is_style else None
    auto = auto_checks(case, summary, cap, expected_heading=heading)
    scores = None
    try:
        scores = judge_summary(judge, case, summary, bopts)
    except SystemExit:
        raise
    except Exception as e:  # a broken judge call must not kill the run
        warn(f"{label}: judge call failed: {e}")
    if auto["failures"]:
        log(f"  {label}: {len(auto['failures'])} automatic check(s) failed: "
            + "; ".join(auto["failures"]))
    return {
        "case": case["name"], "kind": case["kind"], "variant": pset.name,
        "words": info["words"], "summary": summary,
        "auto_failures": auto["failures"], "auto_total": auto["total"],
        "scores": scores,
    }


def overall_score(result):
    """Per-case overall: mean of the three judge dimensions minus a penalty
    per failed automatic check. None when the judge produced nothing."""
    s = result["scores"]
    if not s:
        return None
    mean = sum(s[d] for d in JUDGE_DIMS) / len(JUDGE_DIMS)
    return round(mean - AUTO_PENALTY * len(result["auto_failures"]), 3)


def aggregate(results):
    """Per-variant summary rows, sorted best first."""
    variants = {}
    for r in results:
        variants.setdefault(r["variant"], []).append(r)
    rows = []
    for name, rs in variants.items():
        judged = [overall_score(r) for r in rs if overall_score(r) is not None]
        dims = {}
        for dim in JUDGE_DIMS:
            vals = [r["scores"][dim] for r in rs if r["scores"]]
            dims[dim] = round(statistics.mean(vals), 2) if vals else None
        rows.append({
            "variant": name,
            "cases": len(rs),
            "judged": len(judged),
            "overall": round(statistics.mean(judged), 3) if judged else None,
            "faithfulness": dims["faithfulness"],
            "actionability": dims["actionability"],
            "clarity": dims["clarity"],
            "auto_failed": sum(len(r["auto_failures"]) for r in rs),
            "auto_total": sum(r["auto_total"] for r in rs),
            "mean_words": int(statistics.mean([r["words"] for r in rs])),
        })
    rows.sort(key=lambda row: (
        row["overall"] is None,                      # judged variants first
        -(row["overall"] or 0.0),                    # higher overall first
        row["auto_failed"],                          # fewer check failures
        -(row["faithfulness"] or 0.0),               # then faithfulness
        row["variant"]))
    return rows


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def render_report(rows, results, meta):
    """The human comparison report (markdown)."""
    lines = [
        "# Prompt benchmark report",
        "",
        f"- run: {meta['timestamp']}  (yt-pipeline {VERSION})",
        f"- set: {meta['set_name']} ({meta['n_cases']} cases) from "
        f"`{meta['set_dir']}`",
        f"- summarizer model: {meta['model']}, judge model: "
        f"{meta['judge_model']} (local, via Ollama)",
        f"- prompt variants: {', '.join(r['variant'] for r in rows)}"
        f" (active: {meta['active']})",
        f"- word cap: {meta['max_words']}",
        "",
        "## Comparison",
        "",
        "| Variant | Overall | Faithful | Actionable | Clear | Auto checks "
        "| Avg words |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        active_mark = " **(active)**" if row["variant"] == meta["active"] \
            else ""
        lines.append(
            f"| {row['variant']}{active_mark} | {_fmt(row['overall'])} "
            f"| {_fmt(row['faithfulness'])} | {_fmt(row['actionability'])} "
            f"| {_fmt(row['clarity'])} "
            f"| {row['auto_total'] - row['auto_failed']}/{row['auto_total']} "
            f"| {row['mean_words']} |")
    lines += [
        "",
        f"`Overall` = mean of the three judge dimensions (1-5 each) per "
        f"case, minus {AUTO_PENALTY} per failed automatic check (word cap, "
        "markdown structure, summary kind, expected facts), then averaged "
        "per variant.",
        "",
    ]
    winner = rows[0] if rows else None
    if winner and winner["overall"] is not None:
        if winner["variant"] == meta["active"]:
            lines.append(f"**The active variant `{winner['variant']}` is "
                         "also the best in this run.**")
        else:
            lines.append(
                f"**Winner: `{winner['variant']}`** (overall "
                f"{_fmt(winner['overall'])}, "
                f"{winner['auto_total'] - winner['auto_failed']}/"
                f"{winner['auto_total']} automatic checks). To make it the "
                "default for future runs:")
            lines.append("")
            lines.append(f"    yt-pipeline prompts promote {winner['variant']}")
    elif winner:
        lines.append("**No variant produced judgeable summaries in this "
                     "run; check the per-case details.**")
    lines.append("")
    lines.append("## Per-case detail")
    for row in rows:
        lines.append("")
        lines.append(f"### `{row['variant']}`")
        for r in [x for x in results if x["variant"] == row["variant"]]:
            s = r["scores"]
            score_line = "judge FAILED" if not s else ", ".join(
                f"{d[:1].upper()} {s[d]}" for d in JUDGE_DIMS)
            overall = overall_score(r)
            lines.append(
                f"- **{r['case']}** ({r['kind']}, {r['words']} words, "
                f"overall {_fmt(overall)})  \n"
                f"  checks: {r['auto_total'] - len(r['auto_failures'])}/"
                f"{r['auto_total']}, {score_line}"
                + (f"  \n  auto failures: {'; '.join(r['auto_failures'])}"
                   if r["auto_failures"] else "")
                + (f"  \n  judge: {s['notes']}" if s and s.get("notes") else ""))
    return "\n".join(lines) + "\n"


def write_report(out_dir, rows, results, meta, summaries):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = render_report(rows, results, meta)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    payload = {
        "tool": TOOL, "timestamp": meta["timestamp"],
        "set": {"name": meta["set_name"], "dir": str(meta["set_dir"])},
        "summarizer_model": meta["model"], "judge_model": meta["judge_model"],
        "max_words": meta["max_words"], "active": meta["active"],
        "variants": rows,
        "results": [{k: v for k, v in r.items() if k != "summary"}
                    for r in results],
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    for (variant, case), summary in sorted(summaries.items()):
        d = out_dir / "summaries" / variant
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{case}.md").write_text(summary + "\n", encoding="utf-8")
    return report


# --------------------------------------------------------------------------- #
# Entry point (the 'benchmark' command)
# --------------------------------------------------------------------------- #

def run_benchmark(args):
    """Run the whole benchmark; returns the process exit code."""
    workdir = args.workdir or str(resolve_workdir())
    args.workdir = workdir
    benchset = load_benchset(args.set)

    prompts_dir = resolve_prompts_dir(args.prompts_dir, workdir)
    if args.prompts:
        wanted = [v.strip() for v in args.prompts.split(",") if v.strip()]
        unknown = [v for v in wanted if v not in variant_names(prompts_dir)]
        if unknown:
            die(f"unknown prompt variant(s): {', '.join(unknown)} "
                f"(have: {', '.join(variant_names(prompts_dir))})")
        psets = [load_variant(prompts_dir, v) for v in wanted]
    else:
        psets = list_variants(prompts_dir)
    active = active_name(prompts_dir)

    log(f"benchmark set: {benchset['name']} "
        f"({len(benchset['cases'])} cases) from {benchset['dir']}")
    log(f"prompt variants: {', '.join(p.name for p in psets)} "
        f"(active: {active})")

    bopts = bench_opts_ns(args, workdir)
    ollama = Ollama(args.host, args.model, args.timeout,
                    no_autostart=args.no_autostart)
    ollama.check_model()
    if args.judge_model == args.model:
        judge = ollama
    else:
        judge = Ollama(args.host, args.judge_model, args.timeout,
                       no_autostart=args.no_autostart)
        judge.check_model()

    results, summaries = [], {}
    for pset in psets:
        for case in benchset["cases"]:
            label = f"{pset.name} / {case['name']}"
            log(f"[{label}] summarizing ({case['body_words']} words) ...")
            result = run_one(ollama, judge, pset, case, bopts, label)
            results.append(result)
            summaries[(pset.name, case["name"])] = result["summary"]

    rows = aggregate(results)
    meta = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "n_cases": len(benchset["cases"]),
        "set_dir": str(benchset["dir"]), "set_name": benchset["name"],
        "model": args.model, "judge_model": args.judge_model,
        "max_words": args.max_words if args.max_words is not None
        else "per variant (its style cap, else 500)",
        "active": active,
    }
    out_dir = Path(args.out) if args.out else \
        Path(workdir) / "benchmarks" / "results" / meta["timestamp"]
    report = write_report(out_dir, rows, results, meta, summaries)

    print(report)
    log(f"report: {out_dir / 'report.md'}")

    if rows and rows[0]["overall"] is not None \
            and rows[0]["variant"] != active:
        log(f"hint: promote the winner with  "
            f"yt-pipeline prompts promote {rows[0]['variant']}")

    if args.min_overall is not None:
        row = next((r for r in rows if r["variant"] == active), None)
        if row is None:
            die(f"--min-overall: the active variant {active!r} was not "
                "part of this run")
        # gate verdicts are results, not crashes: the report was written,
        # so warn and return non-zero instead of dying
        if row["overall"] is None:
            warn(f"quality gate: active prompt variant {active!r} produced "
                 "no judgeable summaries")
            return 1
        if row["overall"] < args.min_overall:
            warn(f"quality gate: active prompt variant {active!r} scored "
                 f"{row['overall']:.2f} overall, below the required "
                 f"{args.min_overall:.2f}")
            return 1
        if row["auto_failed"]:
            warn(f"quality gate: active prompt variant {active!r} failed "
                 f"{row['auto_failed']} automatic check(s); see the report")
            return 1
    return 0
