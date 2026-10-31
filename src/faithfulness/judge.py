"""Optional LLM-judge cross-check (spec section 8.1 D). OFF by default.

Purpose: validate that the deterministic verifier isn't missing paraphrased
claims, by asking a DIFFERENT model whether each claim is entailed by the
evidence table, then reporting Cohen's kappa agreement on 30 samples.

On this host only the agent_queue backend exists (one generator), so a truly
different judge model is unavailable; the stage is skipped and the report says
so. If LLM_JUDGE_BASE_URL / LLM_JUDGE_MODEL / LLM_JUDGE_API_KEY are set, the
judge runs against that separate endpoint.
"""
from __future__ import annotations

import json

from src.generation.prompts import render_evidence_table, parse_json_response
from src.utils.config import env, Paths
from src.utils.logging import setup_logger

log = setup_logger("judge")

SYSTEM = ("You are a strict entailment checker. Given an evidence table and a claim, answer "
          "whether the evidence entails the claim. Return ONLY JSON: "
          '{"entailed": true|false, "reason": "..."}')


def judge_backend_available() -> tuple[bool, str]:
    base = env("LLM_JUDGE_BASE_URL")
    model = env("LLM_JUDGE_MODEL")
    if base and model:
        return True, f"configured: {base} / {model}"
    return False, ("no separate judge model configured (LLM_JUDGE_BASE_URL/LLM_JUDGE_MODEL "
                   "unset) and the run generator is agent_queue, so a different model is "
                   "unavailable")


def run_judge(samples, cfg, n_samples: int = 30) -> dict:
    """samples: list of {evidence, per_claim}. Returns agreement stats or a skip note."""
    if not bool(cfg.faithfulness.get("judge_enabled", False)):
        return {"status": "skipped", "reason": "judge_enabled=false (default)"}
    ok, reason = judge_backend_available()
    if not ok:
        return {"status": "skipped", "reason": reason}
    from src.llm.openai_compat import OpenAICompatBackend
    backend = OpenAICompatBackend(base_url=env("LLM_JUDGE_BASE_URL"),
                                  api_key=env("LLM_JUDGE_API_KEY"),
                                  model=env("LLM_JUDGE_MODEL"))
    agree, total = 0, 0
    per_claim = []
    for s in samples[:n_samples]:
        ev = s["evidence"]
        for c in s["per_claim"]:
            prompt = (f"Evidence table:\n{render_evidence_table(ev)}\n\nClaim: {c['text']}\n\n"
                      "Is this claim entailed by the evidence?")
            r = backend.generate(SYSTEM, prompt, temperature=0.0, max_tokens=200, seed=0)
            try:
                ent = bool(parse_json_response(r.text).get("entailed"))
            except Exception:  # noqa: BLE001
                continue
            deterministic_ok = c["status"] == "supported"
            agree += int(ent == deterministic_ok)
            total += 1
            per_claim.append({"sample_id": s.get("sample_id"), "feature": c["feature"],
                              "judge_entailed": ent, "deterministic_supported": deterministic_ok})
    out = {"status": "ran", "n_claims_judged": total,
           "agreement": (agree / total) if total else None, "per_claim": per_claim}
    (Paths.outputs / "judge_crosscheck.json").write_text(json.dumps(out, indent=2))
    return out
