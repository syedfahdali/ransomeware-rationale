"""Disk cache for LLM calls (spec section 6.1).

Key = sha256(backend + model + system + user + temperature + seed + schema).
Reruns cost nothing: a cache hit returns the stored LLMResult.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from src.utils.config import Paths


def make_key(backend: str, model: str, system: str, user: str,
             temperature: float, seed: int | None, schema: dict | None) -> str:
    h = hashlib.sha256()
    payload = {
        "backend": backend, "model": model, "system": system, "user": user,
        "temperature": round(float(temperature), 6), "seed": seed,
        "schema": json.dumps(schema, sort_keys=True) if schema else None,
    }
    h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


class DiskCache:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or Paths.cache)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / f"{key[:2]}" / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self.path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    def put(self, key: str, record: dict) -> None:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        record = dict(record)
        record.setdefault("cached_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        tmp.replace(p)
