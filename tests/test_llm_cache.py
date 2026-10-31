"""LLM cache + agent_queue protocol tests (spec phase 4 acceptance)."""
from __future__ import annotations

import json

import pytest

from src.llm.agent_queue import AgentQueueBackend
from src.llm.base import LLMError, WaitingForAgent
from src.llm.cache import DiskCache, make_key


def test_disk_cache_roundtrip(tmp_path):
    cache = DiskCache(tmp_path)
    key = make_key("agent_queue", "m", "sys", "usr", 0.2, 42, None)
    assert cache.get(key) is None
    cache.put(key, {"text": "hello", "model": "m", "backend": "agent_queue",
                    "latency_s": 1.0, "prompt_tokens": 2, "completion_tokens": 3,
                    "cache_hit": False})
    got = cache.get(key)
    assert got["text"] == "hello"


def test_agent_queue_roundtrip(tmp_path):
    import uuid
    tag = uuid.uuid4().hex
    backend = AgentQueueBackend(batch_size=5, model_name="test-agent", queue_dir=tmp_path,
                                cache_root=tmp_path / "cache")
    with pytest.raises(WaitingForAgent):
        backend.generate(f"SYS-{tag}", f"USER-{tag}", temperature=0.2, seed=42)
    status = backend.emit_batch()
    assert status["n_in_active_batch"] == 1
    req = json.loads((tmp_path / "requests.jsonl").read_text().strip())
    (tmp_path / "responses.jsonl").write_text(
        json.dumps({"id": req["id"], "text": "My prose answer."}) + "\n")
    res = backend.generate(f"SYS-{tag}", f"USER-{tag}", temperature=0.2, seed=42)
    assert res.text == "My prose answer."
    assert res.backend == "agent_queue" and res.cache_hit is False
    res2 = backend.generate(f"SYS-{tag}", f"USER-{tag}", temperature=0.2, seed=42)
    assert res2.cache_hit is True


def test_agent_queue_schema_validation(tmp_path):
    import uuid
    tag = uuid.uuid4().hex
    backend = AgentQueueBackend(batch_size=5, model_name="test-agent", queue_dir=tmp_path,
                                cache_root=tmp_path / "cache")
    schema = {"type": "object", "required": ["verdict", "claims"],
              "properties": {"verdict": {"enum": ["ransomware", "benign"]},
                             "claims": {"type": "array"}}}
    with pytest.raises(WaitingForAgent):
        backend.generate("S2", f"USER-SCHEMA-{tag}", json_schema=schema)
    backend.emit_batch()
    req = json.loads((tmp_path / "requests.jsonl").read_text().strip())
    # first a bad answer
    (tmp_path / "responses.jsonl").write_text(
        json.dumps({"id": req["id"], "json": {"verdict": "not-a-verdict", "claims": []}}) + "\n")
    with pytest.raises(LLMError):
        backend.generate("S2", f"USER-SCHEMA-{tag}", json_schema=schema)
    # then a good one
    (tmp_path / "responses.jsonl").write_text(
        json.dumps({"id": req["id"], "json": {"verdict": "ransomware", "claims": []}}) + "\n")
    res = backend.generate("S2", f"USER-SCHEMA-{tag}", json_schema=schema)
    assert json.loads(res.text)["verdict"] == "ransomware"


def test_estimate_counts(tmp_path):
    backend = AgentQueueBackend(batch_size=5, model_name="test-agent", queue_dir=tmp_path,
                                cache_root=tmp_path / "cache")
    specs = [{"system": "s1", "user": "u1", "temperature": 0.2, "seed": 1, "json_schema": None},
             {"system": "s2", "user": "u2", "temperature": 0.2, "seed": 1, "json_schema": None}]
    est = backend.estimate(specs)
    assert est["total"] == 2 and est["new_calls"] == 2
