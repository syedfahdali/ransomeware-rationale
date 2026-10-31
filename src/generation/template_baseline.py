"""Arm T: deterministic template rationale (spec section 13.3).

Faithful by construction; a sanity ceiling for faithfulness and a floor for
fluency. If this does not score FS ~= 1.0 the VERIFIER has a bug.
"""
from __future__ import annotations

from src.generation.prompts import fmt_value


def template_rationale(ev: dict) -> dict:
    p = float(ev["probability_ransomware"])
    verdict = ev["verdict"]
    sentences = [f"The classifier predicts {verdict} with probability {p:.3f}."]
    claims = []
    for f in ev["top_features"]:
        toward = "toward ransomware" if f["direction"] == "toward_ransomware" else "toward benign"
        text = (f"{f['name']} is {f['qualitative_level']} compared with training data "
                f"(value {fmt_value(f['value'])}, {f['percentile_vs_train']:.0f}th percentile), "
                f"which pushed the decision {toward}.")
        sentences.append(text)
        claims.append({
            "feature": f["name"],
            "direction": f["direction"],
            "level": f["qualitative_level"],
            "value_stated": float(f["value"]),
            "text": text,
            "extractor": "given",
        })
    prose = " ".join(sentences)
    return {
        "verdict": verdict,
        "probability_stated": p,
        "confidence_statement": ("The probability is near the decision threshold; the verdict is "
                                 "uncertain." if abs(p - 0.5) < 0.10 else None),
        "claims": claims,
        "summary": prose,
    }


def template_prose(rationale: dict) -> str:
    return rationale["summary"]
