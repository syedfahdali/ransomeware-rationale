"""Metric validity: synthetic perturbations of faithful rationales (spec 8.3.2).

Detection recall must be >= 95% for all four corruption types:
(a) flip one direction, (b) swap a feature for an out-of-top-k one,
(c) change a number by 30%, (d) drop the top feature.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from src.faithfulness.verify import score_rationale
from src.generation.template_baseline import template_rationale

FEATURES = ["handles.nhandles", "malfind.ninjections", "svcscan.nservices",
            "callbacks.nanonymous", "psxview.not_in_pslist", "dlllist.ndlls",
            "modules.nmodules", "handles.nport", "ldrmodules.not_in_load", "malfind.commitCharge"]


def synthetic_evidence(rng: np.random.RandomState, sid: str) -> dict:
    names = list(rng.choice(FEATURES, size=6, replace=False))
    top = []
    for rank, name in enumerate(names, 1):
        direction = "toward_ransomware" if rng.rand() < 0.7 else "toward_benign"
        level = rng.choice(["very_low", "low", "typical", "high", "very_high"])
        top.append({
            "rank": rank, "name": str(name), "value": float(rng.randint(1, 50000)),
            "shap": float(rng.uniform(0.1, 3.0)), "direction": direction,
            "percentile_vs_train": float(rng.uniform(0, 100)),
            "train_median": float(rng.randint(1, 10000)), "z_score": float(rng.normal()),
            "description": "synthetic", "qualitative_level": str(level),
        })
    # ensure descending |shap|
    top.sort(key=lambda f: -abs(f["shap"]))
    for i, f in enumerate(top, 1):
        f["rank"] = i
    p = float(rng.uniform(0.6, 0.99))
    return {
        "sample_id": sid, "verdict": "ransomware", "predicted_label": 1,
        "probability_ransomware": p, "threshold": 0.5, "base_value": 0.0,
        "top_features": top, "shap_mass_covered_by_top_k": 0.9, "k": 6,
        "all_feature_names": FEATURES,
    }


def perturb(rationale: dict, ev: dict, kind: str, rng: np.random.RandomState) -> dict:
    r = copy.deepcopy(rationale)
    topk = [f["name"] for f in ev["top_features"]]
    if kind == "flip_direction":
        c = r["claims"][0]
        c["direction"] = ("toward_benign" if c["direction"] == "toward_ransomware"
                          else "toward_ransomware")
    elif kind == "out_of_topk_feature":
        outside = [f for f in ev["all_feature_names"] if f not in topk]
        if not outside:
            pytest.skip("no out-of-top-k features available")
        r["claims"][1]["feature"] = str(rng.choice(outside))
    elif kind == "value_30pct":
        c = r["claims"][0]
        base = c.get("value_stated") or ev["top_features"][0]["value"]
        c["value_stated"] = float(base) * 1.30
    elif kind == "drop_top":
        r["claims"] = [c for c in r["claims"] if c["feature"] != topk[0]]
    return r


def detection_rate(kind: str, n: int = 100, seed: int = 42) -> float:
    rng = np.random.RandomState(seed)
    detected = 0
    for i in range(n):
        ev = synthetic_evidence(rng, f"synthetic_{i:03d}")
        base = template_rationale(ev)
        before = score_rationale(base, ev)
        after = score_rationale(perturb(base, ev, kind, rng), ev)
        detected += int(after["FS"] < before["FS"] or (before["FS_pass"] and not after["FS_pass"]))
    return detected / n


@pytest.mark.parametrize("kind", ["flip_direction", "out_of_topk_feature", "value_30pct", "drop_top"])
def test_perturbation_detection_recall(kind):
    rate = detection_rate(kind)
    assert rate >= 0.95, f"{kind}: detection recall {rate:.2f} < 0.95"


def test_template_is_perfect_on_synthetic():
    rng = np.random.RandomState(7)
    for i in range(20):
        ev = synthetic_evidence(rng, f"synthetic_{i:03d}")
        s = score_rationale(template_rationale(ev), ev)
        assert s["FS"] >= 0.99 and s["FS_pass"]
