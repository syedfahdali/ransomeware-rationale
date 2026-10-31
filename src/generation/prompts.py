"""Prompt construction (spec section 7).

The evidence table is the only thing the LLM sees: rounded values and
qualitative words (never raw SHAP floats), per spec pitfall #1.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.utils.config import Paths

VERSION_FILES = {
    "v0_zero_shot": "system_v0.txt",
    "v1_rules": "system_v1.txt",
    "v2_rules_fewshot": "system_v2.txt",
}

STRUCTURED_INSTRUCTION = (
    "Respond with ONE JSON object and nothing else (no markdown fences, no commentary) "
    "matching this JSON Schema:\n{schema}"
)


def load_system(version: str) -> str:
    fname = VERSION_FILES.get(version)
    if not fname:
        raise ValueError(f"unknown prompt version {version!r}")
    return (Paths.prompts / fname).read_text()


def load_fewshot() -> list[dict]:
    p = Paths.prompts / "fewshot.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def fmt_value(v: float) -> str:
    v = float(v)
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    if a == 0:
        return "0"
    return f"{v:.4g}"


def render_evidence_table(ev: dict) -> str:
    p = float(ev["probability_ransomware"])
    lines = [
        f"VERDICT: {ev['verdict']}",
        f"CALIBRATED PROBABILITY: {p:.3f}  (decision threshold {float(ev['threshold']):.3f})",
    ]
    if abs(p - 0.5) < 0.10:
        lines.append("NOTE: the probability is near the threshold -- the verdict is uncertain; say so.")
    directions = {f["direction"] for f in ev["top_features"]}
    loser = "toward_benign" if ev["verdict"] == "ransomware" else "toward_ransomware"
    if ev["verdict"] == "ransomware" and loser in directions:
        lines.append("NOTE: the evidence is mixed -- some listed features push toward benign.")
    if ev["verdict"] == "benign" and loser in directions:
        lines.append("NOTE: the evidence is mixed -- some listed features push toward ransomware.")
    lines.append("")
    lines.append("RANKED EVIDENCE (most influential first). You may only mention these features:")
    for f in ev["top_features"]:
        lines.append(
            f"{f['rank']}. {f['name']} -- value {fmt_value(f['value'])} | "
            f"level {f['qualitative_level']} | {f['percentile_vs_train']:.1f}th percentile of "
            f"training data | direction {f['direction']} | {f['description']}"
        )
    return "\n".join(lines)


def build_user(ev: dict, mode: str, version: str, fewshot: list[dict] | None = None,
               feedback: str | None = None) -> str:
    """mode: 'P' free prose, 'S' structured JSON."""
    parts: list[str] = []
    if fewshot and version == "v2_rules_fewshot":
        for i, ex in enumerate(fewshot[:2], start=1):
            parts.append(f"=== EXAMPLE {i} (verified faithful) ===")
            parts.append(ex["evidence_table"])
            ans = ex["answer_structured"] if mode == "S" else ex["answer_prose"]
            parts.append(("ANSWER: " if mode == "P" else "ANSWER (JSON): ")
                         + (ans if isinstance(ans, str) else json.dumps(ans)))
            parts.append("")
    parts.append("=== EVIDENCE FOR THE SAMPLE TO EXPLAIN ===")
    parts.append(render_evidence_table(ev))
    if mode == "S":
        schema = json.loads((Paths.schemas / "rationale.schema.json").read_text())
        parts.append("")
        parts.append(STRUCTURED_INSTRUCTION.format(
            schema=json.dumps(schema, ensure_ascii=False)))
    else:
        parts.append("")
        parts.append("Write 3-5 sentences of prose explaining this verdict.")
    if feedback:
        parts.append("")
        parts.append("=== VERIFIER FEEDBACK ON YOUR PREVIOUS ANSWER (fix every issue) ===")
        parts.append(feedback)
    return "\n".join(parts)


def parse_json_response(text: str) -> dict:
    """Defensive JSON parse: strip code fences / leading prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if len(t.split("```")) > 1 else t
        if t.startswith("json"):
            t = t[4:]
    t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            return json.loads(t[start:end + 1])
        raise
