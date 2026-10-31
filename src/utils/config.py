"""Configuration loading and canonical project paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


class Cfg(dict):
    """dict with attribute access; nested dicts are wrapped lazily."""

    def __getattr__(self, item: str) -> Any:
        try:
            v = self[item]
        except KeyError as e:  # pragma: no cover - defensive
            raise AttributeError(f"config has no key {item!r}") from e
        return Cfg(v) if isinstance(v, dict) else v


class Paths:
    root = ROOT
    data = ROOT / "data"
    raw = ROOT / "data" / "raw"
    processed = ROOT / "data" / "processed"
    eval = ROOT / "data" / "eval"
    models = ROOT / "models"
    outputs = ROOT / "outputs"
    rationales = ROOT / "outputs" / "rationales"
    evidence = ROOT / "outputs" / "evidence"
    faithfulness = ROOT / "outputs" / "faithfulness"
    figs = ROOT / "outputs" / "figs"
    laya = ROOT / "outputs" / "laya"
    cache = ROOT / "cache" / "llm"
    queue = ROOT / "queue"
    logs = ROOT / "logs"
    prompts = ROOT / "prompts"
    schemas = ROOT / "schemas"
    configs = ROOT / "configs"
    docs = ROOT / "docs"


ALL_DIRS = [
    Paths.data, Paths.raw, Paths.processed, Paths.eval, Paths.models, Paths.outputs,
    Paths.rationales, Paths.evidence, Paths.faithfulness, Paths.figs, Paths.laya,
    Paths.cache,
    Paths.queue, Paths.logs, Paths.prompts, Paths.schemas, Paths.configs, Paths.docs,
    ROOT / "scripts",
]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> Cfg:
    """Load YAML config; also loads .env (never overriding real env vars)."""
    load_dotenv(ROOT / ".env", override=False)
    p = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Cfg(raw)
    cfg["_config_path"] = str(p)
    ensure_dirs()
    return cfg


def env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name, default)
    return v if v not in ("", None) else default


def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
