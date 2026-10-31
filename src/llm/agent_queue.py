"""`agent_queue` backend (spec section 6.3).

No API key and no local model: the pipeline writes pending prompts to
`queue/requests.jsonl` and exits with code 20 / `WAITING_FOR_AGENT`. The coding
agent (Freebuff) reads each request, appends its own answers to
`queue/responses.jsonl` (one JSON object per line, `{"id": ..., "text": ...}`
and/or `{"id": ..., "json": {...}}`), and re-runs the same command. Answered
requests are consumed cache-style and the run continues; unanswered ones are
re-batched (default 25 per batch).

Caveat recorded in the report: the "LLM" here is whatever model powers the
coding agent, which may be closed-weight. Runs are marked
`generator=agent_queue:<agent name>` and are NOT presented as open-weight.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.llm.base import LLMError, LLMResult, WaitingForAgent, result_from_cache
from src.llm.cache import DiskCache, make_key
from src.utils.config import Paths
from src.utils.logging import setup_logger

log = setup_logger("agent_queue")

HOWTO = """# agent_queue — how to answer pending prompts

The pipeline is paused (`WAITING_FOR_AGENT`, exit code 20). Each line of
`queue/requests.jsonl` is one prompt waiting for an answer:

    {"id": "<16 hex>", "kind": "generate|repair|extract", "arm": "P0|P2|S2|S2R|XL",
     "sample_id": "...", "system": "...", "user": "...", "schema": {...} | null}

Append ONE answer line per request to `queue/responses.jsonl`:

    {"id": "<same id>", "text": "<plain prose answer>"}          # prose modes (P0/P2)
    {"id": "<same id>", "json": { ... }}                        # structured modes (S2/S2R/extract)

Rules
- `text` and `json` are both accepted; include the one the request expects
  (`schema` non-null => structured answer; the object is validated against it).
- Do NOT edit `requests.jsonl` ids; ids are sha256-based and deterministic, so
  re-running the same command re-uses existing answers and caches.
- Then re-run the exact same command. When all pending ids are answered the
  pipeline continues to the next batch automatically.
"""


class AgentQueueBackend:
    name = "agent_queue"

    def __init__(self, batch_size: int = 25, model_name: str = "freebuff-agent",
                 queue_dir: Path | None = None, cache_root: Path | None = None):
        self.batch_size = int(batch_size)
        self.model = model_name
        self.dir = Path(queue_dir or Paths.queue)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cache = DiskCache(cache_root)
        self.requests_path = self.dir / "requests.jsonl"
        self.archive_path = self.dir / "requests_archive.jsonl"
        self.responses_path = self.dir / "responses.jsonl"
        self._resp_mtime = -1.0
        self._responses: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        out = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.warning("bad JSON in %s line %d: %s", path.name, line_no, e)
        return out

    def _load_responses(self, force: bool = False) -> dict[str, dict]:
        if not self.responses_path.exists():
            self._responses = {}
            return self._responses
        mtime = self.responses_path.stat().st_mtime
        if force or mtime != self._resp_mtime:
            resp = {}
            for rec in self._read_jsonl(self.responses_path):
                rid = rec.get("id")
                if rid:
                    resp[rid] = rec
            self._responses = resp
            self._resp_mtime = mtime
            log.info("loaded %d agent responses", len(resp))
        return self._responses

    def _validate(self, schema: dict | None, text: str, obj: dict | None) -> dict | None:
        """Validate structured answers; prose answers (schema=None) pass through."""
        import jsonschema
        if schema is None:
            return obj
        if obj is None:
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as e:
                raise LLMError(f"agent answer is not valid JSON: {e}") from e
        try:
            jsonschema.validate(obj, schema)
        except jsonschema.ValidationError as e:
            raise LLMError(f"agent answer violates schema: {e.message}") from e
        return obj

    # ------------------------------------------------------------------ #
    def generate(self, system: str, user: str, *, temperature: float = 0.2,
                 max_tokens: int = 500, seed: int | None = None,
                 json_schema: dict | None = None, kind: str = "generate",
                 arm: str = "", sample_id: str = "") -> LLMResult:
        key = make_key(self.name, self.model, system, user, temperature, seed, json_schema)
        hit = self.cache.get(key)
        if hit:
            return result_from_cache(hit)

        rid = key[:16]
        resp = self._load_responses().get(rid)
        if resp is not None:
            if "json" in resp and resp["json"] is not None:
                obj = self._validate(json_schema, "", resp["json"])
                text = resp.get("text") or json.dumps(obj, ensure_ascii=False)
            else:
                text = resp.get("text", "")
                obj = self._validate(json_schema, text, None)
                text = resp.get("text", text)
            req = self._find_request(rid) or {}
            latency = max(time.time() - float(req.get("emitted_at", time.time())), 0.0)
            result = LLMResult(
                text=text, model=self.model, backend=self.name, latency_s=latency,
                prompt_tokens=len(system + user) // 4,
                completion_tokens=len(text) // 4, cache_hit=False,
                meta={"request_id": rid, "generator": f"agent_queue:{self.model}",
                      "elapsed_since_request_s": latency, "kind": kind, "arm": arm})
            payload = result.__dict__.copy()
            self.cache.put(key, payload)
            return result

        self._ensure_archived(rid, system, user, temperature, max_tokens, seed,
                              json_schema, kind, arm, sample_id)
        raise WaitingForAgent(rid, kind=kind)

    # ------------------------------------------------------------------ #
    def _ensure_archived(self, rid: str, system: str, user: str, temperature: float,
                         max_tokens: int, seed: int | None, schema: dict | None,
                         kind: str, arm: str, sample_id: str) -> None:
        archived = {r.get("id") for r in self._read_jsonl(self.archive_path)}
        if rid in archived:
            return
        record = {
            "id": rid, "kind": kind, "arm": arm, "sample_id": sample_id,
            "system": system, "user": user, "schema": schema,
            "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
            "emitted_at": time.time(), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with self.archive_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _find_request(self, rid: str) -> dict | None:
        for rec in self._read_jsonl(self.archive_path):
            if rec.get("id") == rid:
                return rec
        return None

    # ------------------------------------------------------------------ #
    def unanswered(self) -> list[dict]:
        responses = self._load_responses(force=True)
        pending = [r for r in self._read_jsonl(self.archive_path) if r.get("id") not in responses]
        seen, out = set(), []
        for r in pending:
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(r)
        return out

    def emit_batch(self) -> dict:
        """Write the next batch of unanswered requests; return counts."""
        pending = self.unanswered()
        batch = pending[: self.batch_size]
        with self.requests_path.open("w", encoding="utf-8") as fh:
            for rec in batch:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.write_howto()
        status = {"n_unanswered_total": len(pending), "n_in_active_batch": len(batch),
                  "batch_size": self.batch_size,
                  "responses_file": str(self.responses_path),
                  "requests_file": str(self.requests_path),
                  "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        (self.dir / "status.json").write_text(json.dumps(status, indent=2))
        return status

    def write_howto(self) -> None:
        (self.dir / "HOWTO.md").write_text(HOWTO)

    # ------------------------------------------------------------------ #
    def estimate(self, specs: list[dict]) -> dict:
        """Pre-flight cost estimate for a list of prompt specs.

        Each spec: {system, user, temperature, seed, json_schema}.
        """
        responses = self._load_responses(force=True)
        n_cached = n_answered = n_new = 0
        for s in specs:
            key = make_key(self.name, self.model, s["system"], s["user"],
                           s.get("temperature", 0.2), s.get("seed"), s.get("json_schema"))
            if self.cache.get(key):
                n_cached += 1
            elif key[:16] in responses:
                n_answered += 1
            else:
                n_new += 1
        return {"total": len(specs), "cached": n_cached, "answered": n_answered,
                "new_calls": n_new}
