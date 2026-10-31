"""Remote OpenAI-compatible backend (spec section 6.2a) — DEFAULT in the spec.

Works with OpenRouter / Groq / Together / vLLM / llama-server / any
`/v1/chat/completions` endpoint. Configure via `.env`:

    LLM_BASE_URL=https://openrouter.ai/api/v1
    LLM_API_KEY=...
    LLM_MODEL=<open-weight instruct model id>
    LLM_RPM=20

Model IDs are looked up from the provider's /models endpoint rather than
hard-coded (free-tier IDs change often):

    python -m src.llm.openai_compat --list-models
"""
from __future__ import annotations

import argparse
import json
import time

import requests

from src.llm.base import LLMError, LLMResult, result_from_cache
from src.llm.cache import DiskCache, make_key
from src.utils.config import env
from src.utils.logging import setup_logger

log = setup_logger("llm_openai_compat")


class RateLimiter:
    """Simple spacing limiter: at most `rpm` requests per minute."""

    def __init__(self, rpm: int):
        self.min_interval = 60.0 / max(rpm, 1)
        self.last = 0.0

    def wait(self) -> None:
        now = time.time()
        delta = now - self.last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self.last = time.time()


class OpenAICompatBackend:
    name = "openai_compat"

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, rpm: int | None = None, timeout: float = 120.0):
        self.base_url = (base_url or env("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or env("LLM_API_KEY", "")
        self.model = model or env("LLM_MODEL", "")
        self.rpm = rpm or int(env("LLM_RPM", "20"))
        self.timeout = timeout
        self.json_mode = (env("LLM_JSON_MODE", "auto") or "auto").lower()
        self.cache = DiskCache()
        self.limiter = RateLimiter(self.rpm)
        if not self.base_url:
            raise LLMError("LLM_BASE_URL is empty; set it in .env")
        if not self.model:
            raise LLMError("LLM_MODEL is empty; run `python -m src.llm.openai_compat "
                           "--list-models` and pick an open-weight instruct model")

    # ------------------------------------------------------------------ #
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def list_models(self) -> list[str]:
        r = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
        return sorted(m.get("id", "?") for m in data)

    # ------------------------------------------------------------------ #
    def generate(self, system: str, user: str, *, temperature: float = 0.2,
                 max_tokens: int = 500, seed: int | None = None,
                 json_schema: dict | None = None) -> LLMResult:
        key = make_key(self.name, self.model, system, user, temperature, seed, json_schema)
        hit = self.cache.get(key)
        if hit:
            return result_from_cache(hit)

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        allow_json = json_schema is not None and self.json_mode != "off"
        if allow_json:
            payload["response_format"] = {"type": "json_object"}

        t0 = time.time()
        last_err: Exception | None = None
        for attempt in range(6):
            try:
                self.limiter.wait()
                r = requests.post(f"{self.base_url}/chat/completions",
                                  headers=self._headers(), json=payload, timeout=self.timeout)
                if r.status_code == 400 and "response_format" in r.text and allow_json:
                    log.warning("provider rejected response_format; retrying without JSON mode")
                    payload.pop("response_format", None)
                    allow_json = False
                    continue
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                body = r.json()
                text = body["choices"][0]["message"]["content"]
                usage = body.get("usage", {}) or {}
                result = LLMResult(
                    text=text, model=self.model, backend=self.name,
                    latency_s=time.time() - t0,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    cache_hit=False)
                self.cache.put(key, result.__dict__)  # noqa: SLF001
                return result
            except Exception as e:  # noqa: BLE001
                last_err = e
                sleep = min(60.0, 2.0 ** attempt)
                log.warning("LLM call failed (attempt %d/6): %s -- retrying in %.1fs",
                            attempt + 1, e, sleep)
                time.sleep(sleep)
        raise LLMError(f"LLM call failed after 6 attempts: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()
    if args.list_models:
        b = OpenAICompatBackend(model="__probe__")
        for mid in b.list_models():
            print(mid)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
