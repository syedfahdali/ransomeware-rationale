"""LLM backend adapter interface (spec section 6.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class LLMResult:
    text: str
    model: str
    backend: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_hit: bool = False
    meta: dict = field(default_factory=dict)


def result_from_cache(record: dict) -> LLMResult:
    """Rebuild an LLMResult from a cache record (marking cache_hit)."""
    return LLMResult(
        text=record.get("text", ""), model=record.get("model", ""),
        backend=record.get("backend", ""), latency_s=float(record.get("latency_s") or 0.0),
        prompt_tokens=record.get("prompt_tokens"),
        completion_tokens=record.get("completion_tokens"), cache_hit=True,
        meta=record.get("meta") or {})


class WaitingForAgent(Exception):
    """Raised by the agent_queue backend when prompts await agent answers."""

    def __init__(self, request_id: str, kind: str = "generate"):
        super().__init__(f"WAITING_FOR_AGENT:{request_id}")
        self.request_id = request_id
        self.kind = kind


class LLMError(Exception):
    """Fatal backend error (no key, unreachable, malformed response...)."""


@runtime_checkable
class LLMBackend(Protocol):
    name: str
    model: str

    def generate(self, system: str, user: str, *, temperature: float = 0.2,
                 max_tokens: int = 500, seed: int | None = None,
                 json_schema: dict | None = None) -> LLMResult: ...


def get_backend(cfg, force: str | None = None):
    """Factory for the configured backend (agent_queue | openai_compat | ollama)."""
    from src.utils.config import env

    name = (force or env("LLM_BACKEND") or str(cfg.llm.backend)).strip()
    if name == "agent_queue":
        from src.llm.agent_queue import AgentQueueBackend
        aq = dict(cfg.llm.get("agent_queue", {}) or {})
        return AgentQueueBackend(batch_size=int(aq.get("batch_size", 25)),
                                 model_name=str(env("LLM_MODEL") or aq.get("model_name", "freebuff-agent")))
    if name == "openai_compat":
        from src.llm.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend()
    if name == "ollama":
        from src.llm.ollama_backend import OllamaBackend
        return OllamaBackend()
    raise LLMError(f"unknown LLM_BACKEND={name!r} "
                   "(expected agent_queue | openai_compat | ollama)")
