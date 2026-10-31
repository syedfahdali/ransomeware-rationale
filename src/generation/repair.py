"""Repair loop feedback (spec section 9): deterministic text from failed checks."""
from __future__ import annotations

from src.generation.prompts import fmt_value


def build_feedback(score: dict, ev: dict) -> str:
    lines: list[str] = []
    by_name = {f["name"]: f for f in ev["top_features"]}
    for r in score["per_claim"]:
        name = r["feature"]
        e = by_name.get(name)
        if r["status"] in ("unsupported", "out_of_topk"):
            lines.append(f"- The claim about '{name}' is not in the evidence table. Remove it.")
            continue
        if r["status"] != "contradicted" or e is None:
            continue
        if "direction_ok" in r["failed_checks"]:
            lines.append(f"- Claim about '{name}' says {r['stated_direction']} but the evidence "
                         f"says {e['direction']}.")
        if "level_ok" in r["failed_checks"]:
            lines.append(f"- Claim about '{name}' says level {r['stated_level']} but the evidence "
                         f"says level {e['qualitative_level']}.")
        if "value_ok" in r["failed_checks"]:
            lines.append(f"- Claim about '{name}' states {r['stated_value']} but the evidence "
                         f"value is {fmt_value(e['value'])}.")
        if "conflicting_duplicates" in r["failed_checks"]:
            lines.append(f"- You made conflicting statements about '{name}'. Say it once, correctly.")
    if not score["verdict_ok"]:
        lines.append(f"- State the verdict and probability exactly as given: "
                     f"{ev['verdict']}, {float(ev['probability_ransomware']):.3f}.")
    if score.get("no_claims"):
        lines.append("- Your answer contained no verifiable claims about the listed features. "
                     "Base every sentence on the evidence table.")
    if score.get("policy_violations"):
        for v in score["policy_violations"]:
            lines.append(f"- Policy violation: {v}.")
    if not lines:
        lines.append("- Minor issues only; keep the rationale grounded in the evidence table.")
    lines.append("Rewrite the rationale using ONLY the evidence table and these corrections. "
                 "Keep roughly the same length.")
    return "\n".join(lines)
