"""Versioned prompt artifacts for the summarizer.

The summarizer's five prompt templates no longer live as hardcoded strings
in summarize.py. They are JSON "variant" files, so they can be edited,
version-controlled, compared and promoted:

    prompts/<name>.json

Each variant file looks like this::

    {
      "name": "v1-baseline",          # must equal the file name (minus .json)
      "version": 1,
      "description": "what this variant tries differently",
      "created": "2026-02-27",
      "style": {                       # OPTIONAL: marks the variant as a
        "label": "Short",             # summary *style* selectable with
        "summary": "one short ...",   # --style NAME (see README)
        "max_words": 150,             # optional: the style's own word cap
        "fallback_heading": "## Overview"   # optional: heading to add if
      },                               # the model outputs none
      "prompts": {
        "system":   "...",            # system prompt, static text
        "final":    "... {transcript} {lang_instr} {max_words}",
        "notes":    "... {i} {n} {lang_instr} {notes_words} {chunk}",
        "reduce":   "... {lang_instr} {max_words} {notes}",
        "condense": "... {actual} {max_words} {text}"
      }
    }

A variant WITH a "style" block is a summary style: run/summarize select
it per run with `--style NAME`. Without --style the active variant is
used, so the default summary behavior never changes just because styles
exist. All "style" keys are optional; a style's max_words only applies
when --max-words was not given.

The templates are Python format strings using exactly the placeholders the
summarizer fills in; anything else is rejected at load time.

Where the files live, in order:

1. --prompts-dir DIR           (summarize/run/benchmark/prompts commands)
2. $YT_PROMPTS_DIR
3. <workdir>/prompts            if it contains any *.json
4. the copy packed into the executable (vidsummarizer/prompts in the zipapp)
5. <repo>/prompts               when running from a source checkout

The *active* variant (the one plain `run`/`summarize` use) is named in the
file <prompts-dir>/active.txt, one line, no extension. Without that file the
shipped default (DEFAULT_VARIANT) is used. Writes always go to --prompts-dir
or <workdir>/prompts (materialized from the read location if needed), so a
read-only packed executable never blocks `prompts promote` / `prompts new`.
"""

import importlib.resources
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .common import base_dir, die, log

DEFAULT_VARIANT = "v1-baseline"
ACTIVE_FILE = "active.txt"

# the five templates every variant must provide
TEMPLATE_KEYS = ("system", "final", "notes", "reduce", "condense")

# every placeholder any template may use; a superset dict is used for
# validation, so unknown placeholders fail fast instead of at summarization
PLACEHOLDERS = {
    "lang_instr": "Write the summary in English.",
    "max_words": "500",
    "transcript": "text",
    "notes": "text",
    "i": "1",
    "n": "2",
    "notes_words": "220",
    "chunk": "text",
    "actual": "600",
    "text": "text",
}

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

# optional "style" block of a variant file: key -> (type, minimal check)
STYLE_FIELDS = {
    "label": (str, "non-empty"),
    "summary": (str, "non-empty"),
    "max_words": (int, ">= 1"),
    "fallback_heading": (str, "non-empty"),
}


class PromptSet:
    """One loaded, validated prompt variant."""

    def __init__(self, name, version, description, created, prompts,
                 path=None, style=None):
        self.name = name
        self.version = version
        self.description = description
        self.created = created
        self.path = path  # file the variant was loaded from (diagnostics)
        self.style = dict(style or {})  # style metadata block, {} if none
        self.system = prompts["system"]
        self.final = prompts["final"]
        self.notes = prompts["notes"]
        self.reduce = prompts["reduce"]
        self.condense = prompts["condense"]

    def __repr__(self):
        return f"<PromptSet {self.name!r} v{self.version}>"

    # -- style metadata (all default sensibly for plain variants) --------- #

    @property
    def is_style(self):
        """True when the variant declares style metadata (--style-able)."""
        return bool(self.style)

    @property
    def style_label(self):
        """Human-readable style name for messages (file name if unset)."""
        return self.style.get("label") or self.name

    @property
    def max_words(self):
        """The style's own word cap, or None (-> the tool default)."""
        return self.style.get("max_words")

    @property
    def fallback_heading(self):
        """Heading to prepend when the model output has no '##' heading."""
        return self.style.get("fallback_heading") or "## Overview"

    def as_dict(self):
        data = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created": self.created,
            "prompts": {k: getattr(self, k) for k in TEMPLATE_KEYS},
        }
        if self.style:
            data["style"] = dict(self.style)
        return data


# --------------------------------------------------------------------------- #
# Locating the prompts directory
# --------------------------------------------------------------------------- #

def _has_variants(d):
    try:
        return any(f.name.endswith(".json") for f in d.iterdir())
    except (OSError, AttributeError, TypeError, ValueError):
        return False


def _packaged_dir():
    """The prompts/ copy packed next to the package (inside the zipapp)."""
    try:
        d = importlib.resources.files("vidsummarizer").joinpath("prompts")
        return d if _has_variants(d) else None
    except Exception:
        return None


def _repo_dir():
    """<repo>/prompts when running from a source checkout."""
    d = Path(base_dir()) / "prompts"
    return d if d.is_dir() and _has_variants(d) else None


def resolve_prompts_dir(explicit=None, workdir=None):
    """Find the directory holding the prompt variants (see module docstring).
    Returns a path-like (Path for real dirs, Traversable inside a zipapp)."""
    for label, value in (("--prompts-dir", explicit),
                         ("$YT_PROMPTS_DIR", os.environ.get("YT_PROMPTS_DIR"))):
        if value:
            d = Path(value).expanduser()
            if not _has_variants(d):
                die(f"{label} {value}: not a directory holding *.json "
                    "prompt variant files")
            return d
    if workdir:
        wd = Path(workdir) / "prompts"
        if _has_variants(wd):
            return wd
    for cand in (_packaged_dir(), _repo_dir()):
        if cand is not None:
            return cand
    die("no prompt variant files found (looked in <workdir>/prompts, the "
        "executable and the checkout's prompts/ directory); re-install or "
        "pass --prompts-dir DIR")


def writable_prompts_dir(read_dir, workdir, explicit=None):
    """Directory writes (active.txt, new variants) may go to.

    --prompts-dir when given; otherwise <workdir>/prompts, materialized
    with a copy of the readable variants when it holds none yet (so a
    packed, read-only executable and a temporary workdir both work). In a
    plain checkout <workdir> is the repo root, so writes land in the very
    prompts/ the variants are versioned in.
    """
    if explicit:
        return Path(explicit)
    dest = Path(workdir) / "prompts"
    if _has_variants(dest):
        return dest
    src = read_dir if read_dir is not None and _has_variants(read_dir) \
        else (_packaged_dir() or _repo_dir())
    dest.mkdir(parents=True, exist_ok=True)
    if src is not None:
        for f in sorted(src.iterdir(), key=lambda f: f.name):
            if f.name.endswith(".json"):
                # copy through the Traversable API: src may live inside
                # the zipapp, where str(f) is not a real filesystem
                # path and shutil.copyfile cannot read it
                (dest / f.name).write_text(f.read_text(encoding="utf-8"),
                                           encoding="utf-8")
        log(f"copied the prompt variants to {dest}")
    return dest


# --------------------------------------------------------------------------- #
# Loading / validating variants
# --------------------------------------------------------------------------- #

def validate_variant(data, source="<data>"):
    """Check the parsed JSON of a variant file; raise ValueError on issues."""
    if not isinstance(data, dict):
        raise ValueError(f"{source}: not a JSON object")
    for key in ("name", "version", "description", "prompts"):
        if key not in data:
            raise ValueError(f"{source}: missing key '{key}'")
    name = data["name"]
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValueError(f"{source}: 'name' must be lowercase letters, "
                         f"digits and dashes (got {name!r})")
    if not isinstance(data["version"], int) or data["version"] < 1:
        raise ValueError(f"{source}: 'version' must be an integer >= 1")
    if not isinstance(data["description"], str) or not data["description"].strip():
        raise ValueError(f"{source}: 'description' must be a non-empty string")
    prompts = data["prompts"]
    if not isinstance(prompts, dict):
        raise ValueError(f"{source}: 'prompts' must be an object")
    for key in TEMPLATE_KEYS:
        if key not in prompts:
            raise ValueError(f"{source}: missing prompt '{key}'")
        value = prompts[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{source}: prompt '{key}' must be a non-empty "
                             "string")
        if key == "system":
            continue  # static text, never formatted
        try:
            value.format(**PLACEHOLDERS)
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(
                f"{source}: prompt '{key}' uses a placeholder this tool does "
                f"not fill in ({e}); known placeholders: "
                f"{', '.join(sorted(PLACEHOLDERS))}") from e
    style = data.get("style")
    if style is not None:
        if not isinstance(style, dict):
            raise ValueError(f"{source}: 'style' must be an object")
        for key, (typ, rule) in STYLE_FIELDS.items():
            if key not in style:
                continue
            value = style[key]
            ok = isinstance(value, typ) and not isinstance(value, bool)
            if ok and rule == "non-empty":
                ok = bool(str(value).strip())
            if ok and key == "max_words":
                ok = value >= 1
            if not ok:
                raise ValueError(f"{source}: style '{key}' must be "
                                 f"{typ.__name__} {rule} (got {value!r})")
    return data


def variant_names(d):
    """Sorted variant names (file stems) in directory d."""
    try:
        names = [f.name[:-5] for f in d.iterdir() if f.name.endswith(".json")]
    except (OSError, AttributeError, TypeError, ValueError) as e:
        die(f"cannot read prompt directory {d}: {e}")
    return sorted(names)


def _read_json(f):
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        die(f"cannot read prompt variant {f.name}: {e}")


def load_variant(d, name):
    """Load and validate one variant by name from directory d."""
    if not NAME_RE.match(name or ""):
        die(f"invalid prompt variant name {name!r} (lowercase letters, "
            "digits and dashes)")
    f = d / (name + ".json")
    try:
        exists = f.is_file()
    except (OSError, AttributeError):
        exists = False
    if not exists:
        die(f"no prompt variant '{name}' in {d} "
            f"(have: {', '.join(variant_names(d)) or 'none'})")
    data = validate_variant(_read_json(f), source=f.name)
    if data["name"] != name:
        die(f"{f.name}: 'name' is {data['name']!r} but the file must be "
            f"named {data['name']}.json")
    return PromptSet(data["name"], data["version"], data["description"],
                     data.get("created", ""), data["prompts"], path=str(f),
                     style=data.get("style"))


def list_variants(d):
    return [load_variant(d, n) for n in variant_names(d)]


def styles_in(d):
    """The variants that declare style metadata (selectable via --style)."""
    return [v for v in list_variants(d) if v.is_style]


def active_name(d):
    """Name of the active variant (active.txt if present, else default)."""
    f = d / ACTIVE_FILE
    try:
        name = f.read_text(encoding="utf-8").strip()
    except (OSError, AttributeError, KeyError):
        # no active.txt -> the default variant. KeyError: py3.9's
        # zipfile.Path (importlib.resources inside the zipapp) raises it
        # for a missing entry; newer versions raise FileNotFoundError.
        return DEFAULT_VARIANT
    return name or DEFAULT_VARIANT


def load_active(d):
    return load_variant(d, active_name(d))


def write_variant(dirpath, pset_or_dict):
    """Write a variant file (name taken from the data itself)."""
    data = pset_or_dict.as_dict() if isinstance(pset_or_dict, PromptSet) \
        else pset_or_dict
    validate_variant(data, source=data.get("name", "<variant>") + ".json")
    path = Path(dirpath) / (data["name"] + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def set_active(dirpath, name):
    """Point active.txt at `name`; returns the file written."""
    load_variant(dirpath, name)  # must exist and validate first
    path = Path(dirpath) / ACTIVE_FILE
    with open(path, "w", encoding="utf-8") as f:
        f.write(name + "\n")
    return path


def new_variant(read_dir, write_dir, name, from_name):
    """Scaffold prompts/<name>.json as a copy of the `from_name` variant."""
    if not NAME_RE.match(name or ""):
        die(f"invalid variant name {name!r} (lowercase letters, digits and "
            "dashes)")
    if name == from_name:
        die(f"the new variant needs a different name than {from_name!r}")
    base = load_variant(read_dir, from_name)
    data = base.as_dict()
    data.update({
        "name": name,
        "description": f"copied from {from_name}; edit and benchmark me",
        "created": datetime.now(timezone.utc).date().isoformat(),
    })
    return write_variant(write_dir, data)
