"""Verify that every path referenced by the project docs exists.

Checks markdown link targets (relative only) and repo-relative paths written in
backticks, across README.md, docs/*.md and any skills/**/SKILL.md. Placeholders
(`{arm}`, `*`, `<id>`, `...`) are skipped. Exit code 1 if anything is missing.

Run:  python scripts/check_docs.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *(sorted((ROOT / "docs").glob("*.md"))),
        *(sorted((ROOT / "skills").rglob("SKILL.md")))]

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
CODE_RE = re.compile(r"`([^`\s]+)`")
SKIP_PREFIX = ("http://", "https://", "mailto:", "#", "~")
PLACEHOLDERS = ("{", "}", "*", "<", ">", "...")
EXT = (".md", ".py", ".json", ".yaml", ".yml", ".txt", ".csv", ".jsonl", ".png",
       ".gif", ".parquet", ".npz", ".sh", ".joblib", ".html")
IGNORE_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache",
               "node_modules", ".ipynb_checkpoints"}
BASENAMES: set[str] = set()


def basename_index() -> set[str]:
    """Every file basename in the repo (bare names are matched against it)."""
    names: set[str] = set()
    for p in ROOT.rglob("*"):
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        if p.is_file():
            names.add(p.name)
    return names


def looks_like_path(token: str) -> bool:
    if any(p in token for p in PLACEHOLDERS):
        return False
    if any(token.startswith(p) for p in SKIP_PREFIX):
        return False
    if token.endswith(EXT):
        return True
    return token.startswith(("src/", "docs/", "outputs/", "data/", "scripts/", "models/",
                             "prompts/", "queue/", "schemas/", "configs/", "tests/", "logs/"))


def resolve(doc: Path, token: str) -> Path:
    # A markdown link to a path containing spaces must be percent-encoded
    # ("Laya%20Decision-...md"), so decode it before touching the filesystem —
    # and drop any ?query, the way a browser would.
    token = unquote(token.split("#", 1)[0].split("?", 1)[0]).strip()
    if token.startswith("/"):
        return ROOT / token.lstrip("/")
    # markdown links are relative to the doc; backticked paths are repo-relative
    if token.startswith(("src/", "docs/", "outputs/", "data/", "scripts/", "models/",
                         "prompts/", "queue/", "schemas/", "configs/", "tests/", "logs/")):
        return ROOT / token
    cand = (doc.parent / token)
    return cand if cand.exists() else ROOT / token


def main() -> int:
    global BASENAMES
    BASENAMES = basename_index()
    missing: list[str] = []
    checked = 0
    for doc in DOCS:
        if not doc.exists():
            missing.append(f"{doc.relative_to(ROOT)} (document itself missing)")
            continue
        text = doc.read_text(encoding="utf-8")
        rel = doc.relative_to(ROOT)
        for match in LINK_RE.finditer(text):
            target = match.group(1).strip()
            if any(target.startswith(p) for p in SKIP_PREFIX):
                continue
            checked += 1
            if not resolve(doc, target).exists():
                missing.append(f"{rel}: link -> {target}")
        for match in CODE_RE.finditer(text):
            token = match.group(1).strip()
            if not looks_like_path(token):
                continue
            if " " in token:
                continue
            checked += 1
            if "/" not in token:          # bare filename: match by basename anywhere
                if token in BASENAMES:
                    continue
            elif resolve(doc, token).exists():
                continue
            missing.append(f"{rel}: path -> {token}")
    print(f"checked {checked} path reference(s) in {len(DOCS)} document(s)")
    if missing:
        print("MISSING:")
        for item in missing:
            print("  -", item)
        return 1
    print("all referenced paths exist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
