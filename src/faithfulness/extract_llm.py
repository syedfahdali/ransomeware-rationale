"""LLM claim-extraction fallback (spec section 8.2, stage 2).

Used only for sentences the rule extractor cannot parse. Always logs which
extractor produced each claim.
"""
from __future__ import annotations

import json

from src.faithfulness.extract_rules import (extract_claims, split_sentences,
                                            unparsed_feature_sentences)
from src.generation.prompts import render_evidence_table
from src.utils.config import Paths
from src.utils.logging import setup_logger

log = setup_logger("extract_llm")

SYSTEM = (
    "You extract verifiable claims from an analyst's rationale about a classifier verdict. "
    "Return ONLY JSON matching the schema. Each claim names one feature from the evidence, the "
    "direction the rationale attributes to it (toward_ransomware/toward_benign), the qualitative "
    "level word if used (very_low/low/typical/high/very_high or null), the stated numeric value "
    "if any (or null), and the sentence text. Do not invent claims the rationale does not make."
)


def canonicalize(name: str, feature_names: list[str]) -> str:
    low = name.strip().lower()
    for f in feature_names:
        if f.lower() == low:
            return f
    return name.strip()


def extract_with_llm(prose: str, ev: dict, backend, temperature: float = 0.0) -> list[dict]:
    # Dedicated extraction schema (claims only). Using rationale.schema.json here made
    # every fallback answer fail validation on the generator-only `verdict` field,
    # silently degrading prose arms to rules-only extraction.
    schema_path = Paths.schemas / "extraction.schema.json"
    schema = json.loads(schema_path.read_text()) if schema_path.exists() else \
        json.loads((Paths.schemas / "rationale.schema.json").read_text())
    user = "\n".join([
        "Evidence table:", render_evidence_table(ev), "",
        "Rationale to extract claims from:", prose, "",
        "Return the JSON object now.",
    ])
    result = backend.generate(SYSTEM, user, temperature=temperature, max_tokens=700,
                              seed=0, json_schema=schema, kind="extract",
                              sample_id=ev["sample_id"])
    from src.generation.prompts import parse_json_response
    obj = parse_json_response(result.text)
    claims = []
    for c in obj.get("claims", []):
        c = dict(c)
        c["feature"] = canonicalize(str(c.get("feature", "")), ev["all_feature_names"])
        c["extractor"] = "llm"
        claims.append(c)
    return claims


def extract_claims_smart(prose: str, ev: dict, backend=None, allow_llm: bool = True,
                         ctx: dict | None = None) -> tuple[list[dict], dict]:
    """Rules first; LLM fallback for unparsed content. Returns (claims, info)."""
    ctx = ctx if ctx is not None else {"calls": 0, "cap": 60}
    rules = extract_claims(prose, ev["all_feature_names"])
    info = {"n_rules": len(rules), "n_llm": 0, "fallback": False,
            "extractors": sorted({c.get("extractor") for c in rules})}
    unparsed = unparsed_feature_sentences(prose, ev["all_feature_names"], rules)
    need_fallback = (len(rules) == 0 and len(split_sentences(prose)) > 0) or len(unparsed) > 0
    if not allow_llm or backend is None or not need_fallback:
        return rules, info
    if ctx["calls"] >= ctx["cap"]:
        log.warning("LLM-extractor budget exhausted (%d); keeping rules-only", ctx["cap"])
        return rules, info

    ctx["calls"] += 1
    try:
        llm_claims = extract_with_llm(prose, ev, backend)
    except Exception as e:  # noqa: BLE001 - includes WaitingForAgent (propagates up)
        from src.llm.base import WaitingForAgent
        if isinstance(e, WaitingForAgent):
            raise
        log.warning("LLM extractor failed for %s: %s", ev["sample_id"], e)
        return rules, info

    covered = {c["feature"].lower() for c in rules}
    merged = list(rules)
    for c in llm_claims:
        if c["feature"].lower() not in covered:
            merged.append(c)
            covered.add(c["feature"].lower())
    info.update({"fallback": True, "n_llm": len(merged) - len(rules),
                 "extractors": sorted({c.get("extractor") or "rules" for c in merged})})
    return merged, info
