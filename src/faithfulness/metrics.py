"""Aggregate faithfulness metrics (spec section 8.1 B).

Composite Faithfulness Score and its weight sensitivity analysis. All
components are always reported separately; the composite never hides them.

    FS(alpha, w) = verdict_ok * claim_precision**alpha * ((1-w) + w*evidence_coverage)

Main setting: alpha=0.5, w=0.5 (documented in the report).
"""
from __future__ import annotations

import numpy as np

COMPONENTS = ["claim_precision", "claim_precision_lenient", "hallucination_rate",
              "contradiction_rate", "evidence_coverage", "FS"]


def fs_with_weights(score: dict, alpha: float = 0.5, w: float = 0.5) -> float:
    if score.get("no_claims"):
        return 0.0
    return float(score["verdict_ok"]) * (float(score["claim_precision"]) ** alpha) * \
        ((1 - w) + w * float(score["evidence_coverage"]))


def sensitivity_table(scores: list[dict],
                      alphas=(0.25, 0.5, 0.75, 1.0),
                      weights=(0.0, 0.25, 0.5, 0.75, 1.0)) -> list[dict]:
    out = []
    for a in alphas:
        for w in weights:
            vals = [fs_with_weights(s, a, w) for s in scores]
            out.append({"alpha": a, "coverage_weight": w, "mean_FS": float(np.mean(vals)),
                        "n": len(vals)})
    return out


def aggregate(scores: list[dict]) -> dict:
    if not scores:
        return {"n": 0}
    agg = {"n": len(scores)}
    for comp in COMPONENTS:
        agg[comp] = float(np.mean([s[comp] for s in scores]))
    agg["FS_pass_rate"] = float(np.mean([bool(s["FS_pass"]) for s in scores]))
    agg["top1_mentioned_rate"] = float(np.mean([bool(s["top1_mentioned"]) for s in scores]))
    agg["verdict_consistency_rate"] = float(np.mean([bool(s["verdict_ok"]) for s in scores]))
    agg["no_claims_rate"] = float(np.mean([bool(s.get("no_claims")) for s in scores]))
    agg["policy_violation_rate"] = float(np.mean([bool(s.get("policy_violations")) for s in scores]))
    agg["remediation_rate"] = float(np.mean([
        "remediation advice" in (s.get("policy_violations") or []) for s in scores]))
    agg["family_speculation_rate"] = float(np.mean([
        "malware-family speculation" in (s.get("policy_violations") or []) for s in scores]))
    agg["mean_claims"] = float(np.mean([s["n_claims"] for s in scores]))
    agg["total_hallucinations"] = int(sum(s["status_counts"]["unsupported"] for s in scores))
    agg["total_contradictions"] = int(sum(s["status_counts"]["contradicted"] for s in scores))
    agg["total_claims"] = int(sum(s["n_claims"] for s in scores))
    return agg
