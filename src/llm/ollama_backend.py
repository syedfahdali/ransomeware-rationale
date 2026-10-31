"""Ollama backend (spec section 6.2b): local or remote Ollama server.

NOT used in this study's main run (the host has 8 GB RAM and the user asked
not to download local models). Kept runnable for portability: for this machine
only tiny models (~1-3B, 4-bit) are viable, e.g. `llama3.2:3b` or
`qwen2.5:3b`; expect slow, lower-quality output. A stronger machine or Ollama
cloud only needs a different OLLAMA_HOST / model name.
"""
from __future__ import annotations

import time

import requests

from src.llm.base import LLMError, LLMResult, result_from_cache
from src.llm.cache import DiskCache, make_key
from src.utils.config import env
from src.utils.logging import setup_logger

log = setup_logger("llm_ollama")


class OllamaBackend:
    name = "ollama"

    def __init__(self, host: str | None = None, model: str | None = None, timeout: float = 600.0):
        self.host = (host or env("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or env("LLM_MODEL", "") or "llama3.2:3b"
        self.timeout = timeout
        self.cache = DiskCache()

    def generate(self, system: str, user: str, *, temperature: float = 0.2,
                 max_tokens: int = 500, seed: int | None = None,
                 json_schema: dict | None = None) -> LLMResult:
        key = make_key(self.name, self.model, system, user, temperature, seed, json_schema)
        hit = self.cache.get(key)
        if hit:
            return result_from_cache(hit)
        payload = {
            "model": self.model, "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if seed is not None:
            payload["options"]["seed"] = seed
        if json_schema is not None:
            payload["format"] = json_schema
        t0 = time.time()
        last: Exception | None = None
        for attempt in range(4):
            try:
                r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
                r.raise_for_status()
                body = r.json()
                result = LLMResult(
                    text=body["message"]["content"], model=self.model, backend=self.name,
                    latency_s=time.time() - t0,
                    prompt_tokens=body.get("prompt_eval_count"),
                    completion_tokens=body.get("eval_count"), cache_hit=False)
                self.cache.put(key, result.__dict__)  # noqa: SLF001
                return result
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(min(30.0, 2.0 ** attempt))
        raise LLMError(f"ollama call failed: {last}")
