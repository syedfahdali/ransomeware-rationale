"""Verifier unit tests (spec phase 6 acceptance: test_verify.py passes)."""
from __future__ import annotations

import json

import pytest

from src.faithfulness.verify import normalize_level, parse_number, score_rationale, verify_claim
from src.generation.template_baseline import template_rationale


def make_evidence():
    return {
        "sample_id": "test_000001",
        "verdict": "ransomware",
        "predicted_label": 1,
        "probability_ransomware": 0.973,
        "threshold": 0.5,
        "base_value": -0.12,
        "k": 3,
        "shap_mass_covered_by_top_k": 0.9,
        "all_feature_names": ["handles.nhandles", "malfind.ninjections", "svcscan.nservices",
                              "callbacks.nanonymous"],
        "top_features": [
            {"rank": 1, "name": "handles.nhandles", "value": 18234.0, "shap": 1.84,
             "direction": "toward_ransomware", "percentile_vs_train": 97.3,
             "train_median": 5230.0, "z_score": 2.9, "description": "x",
             "qualitative_level": "very_high"},
            {"rank": 2, "name": "malfind.ninjections", "value": 5.0, "shap": 0.9,
             "direction": "toward_ransomware", "percentile_vs_train": 90.0,
             "train_median": 1.0, "z_score": 1.2, "description": "x",
             "qualitative_level": "high"},
            {"rank": 3, "name": "svcscan.nservices", "value": 300.0, "shap": -0.4,
             "direction": "toward_benign", "percentile_vs_train": 50.0,
             "train_median": 300.0, "z_score": 0.0, "description": "x",
             "qualitative_level": "typical"},
        ],
    }


def test_supported_claim():
    ev = make_evidence()
    c = {"feature": "handles.nhandles", "direction": "toward_ransomware",
         "level": "very_high", "value_stated": 18234.0, "text": "t"}
    assert verify_claim(c, ev).status == "supported"
    # value with thousands separator and k-suffix tolerance
    c2 = dict(c, value_stated="18,234")
    assert verify_claim(c2, ev).status == "supported"


def test_contradicted_claims():
    ev = make_evidence()
    bad_dir = {"feature": "handles.nhandles", "direction": "toward_benign", "text": "t"}
    assert verify_claim(bad_dir, ev).status == "contradicted"
    bad_level = {"feature": "handles.nhandles", "level": "low", "text": "t"}
    assert verify_claim(bad_level, ev).status == "contradicted"
    bad_val = {"feature": "handles.nhandles", "value_stated": 24000.0, "text": "t"}
    call = verify_claim(bad_val, ev)
    assert call.status == "contradicted" and "value_ok" in call.failed_checks


def test_value_tolerance_boundary():
    ev = make_evidence()
    ok = {"feature": "handles.nhandles", "value_stated": 18234 * 1.049, "text": "t"}
    bad = {"feature": "handles.nhandles", "value_stated": 18234 * 1.06, "text": "t"}
    assert verify_claim(ok, ev).status == "supported"
    assert verify_claim(bad, ev).status == "contradicted"


def test_out_of_topk_and_hallucinated():
    ev = make_evidence()
    oot = {"feature": "callbacks.nanonymous", "direction": "toward_ransomware", "text": "t"}
    assert verify_claim(oot, ev).status == "out_of_topk"
    hal = {"feature": "made.up.feature", "direction": "toward_ransomware", "text": "t"}
    assert verify_claim(hal, ev).status == "unsupported"


def test_duplicates_and_no_claims():
    ev = make_evidence()
    dup = {"verdict": "ransomware", "probability_stated": 0.973, "claims": [
        {"feature": "handles.nhandles", "direction": "toward_ransomware", "text": "a"},
        {"feature": "handles.nhandles", "direction": "toward_ransomware", "text": "b"},
    ]}
    s = score_rationale(dup, ev)
    assert s["n_claims"] == 1 and s["claim_precision"] == 1.0
    conflict = {"verdict": "ransomware", "probability_stated": 0.973, "claims": [
        {"feature": "handles.nhandles", "direction": "toward_ransomware", "text": "a"},
        {"feature": "handles.nhandles", "direction": "toward_benign", "text": "b"},
    ]}
    s2 = score_rationale(conflict, ev)
    assert s2["contradiction_rate"] > 0 and not s2["FS_pass"]
    empty = {"verdict": "ransomware", "claims": []}
    s3 = score_rationale(empty, ev)
    assert s3["no_claims"] and s3["FS"] == 0.0


def test_verdict_checks():
    ev = make_evidence()
    wrong_verdict = {"verdict": "benign", "claims": []}
    assert score_rationale(wrong_verdict, ev)["verdict_ok"] is False
    wrong_prob = {"verdict": "ransomware", "probability_stated": 0.60, "claims": []}
    assert score_rationale(wrong_prob, ev)["verdict_ok"] is False
    ok_prob = {"verdict": "ransomware", "probability_stated": 0.975, "claims": []}
    assert score_rationale(ok_prob, ev)["verdict_ok"] is True
    missing_prob = {"verdict": "ransomware", "claims": []}
    assert score_rationale(missing_prob, ev)["verdict_ok"] is True


def test_level_and_number_parsing():
    assert normalize_level("Very High") == "very_high"
    assert normalize_level("elevated") == "high"
    assert normalize_level(None) is None
    assert parse_number("1,234") == 1234.0
    assert parse_number("2.5k") == 2500.0
    assert parse_number("3M") == 3e6
    assert parse_number("abc") is None


def test_template_arm_scores_perfectly():
    ev = make_evidence()
    score = score_rationale(template_rationale(ev), ev)
    assert score["FS"] >= 0.99
    assert score["FS_pass"] is True
    assert score["claim_precision"] == 1.0
    assert score["evidence_coverage"] == pytest.approx(1.0)


def test_schema_validation_of_template():
    import jsonschema
    from src.utils.config import Paths
    ev = make_evidence()
    schema = json.loads((Paths.schemas / "rationale.schema.json").read_text())
    jsonschema.validate(template_rationale(ev), schema)
