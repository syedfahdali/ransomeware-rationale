"""Rule-extractor tests: what a claim carries must match what the prose says.

These guard the binding rules that decide which number/level/direction belongs to
which feature mention — the places where a wrong attachment would silently turn a
hallucinating rationale into a "supported" one.
"""
from __future__ import annotations

from src.faithfulness.extract_rules import extract_claims, find_numbers, split_sentences
from src.faithfulness.verify import score_rationale

FEATURES = ["handles.nmutant", "malfind.commitCharge", "handles.nfile", "pslist.avg_threads",
            "svcscan.nservices"]


def evidence():
    return {
        "sample_id": "test_000001",
        "verdict": "benign",
        "predicted_label": 0,
        "probability_ransomware": 0.054,
        "threshold": 0.5,
        "k": 2,
        "shap_mass_covered_by_top_k": 1.0,
        "all_feature_names": FEATURES,
        "top_features": [
            {"rank": 1, "name": "handles.nmutant", "value": 282.0, "shap": -0.75,
             "direction": "toward_benign", "percentile_vs_train": 47.8, "train_median": 300.0,
             "z_score": -0.1, "description": "x", "qualitative_level": "typical"},
            {"rank": 2, "name": "malfind.commitCharge", "value": 9215.0, "shap": 0.37,
             "direction": "toward_ransomware", "percentile_vs_train": 95.5, "train_median": 1000.0,
             "z_score": 1.9, "description": "x", "qualitative_level": "very_high"},
        ],
    }


def test_value_keyword_binds_across_a_long_phrase():
    """'(value 9,215)' after 'X is very_high compared with training data' still binds."""
    prose = ("handles.nmutant is typical compared with training data (value 282.0), "
             "which pushed the decision toward benign.")
    claims = extract_claims(prose, FEATURES)
    assert len(claims) == 1
    c = claims[0]
    assert c["feature"] == "handles.nmutant"
    assert c["level"] == "typical"
    assert c["direction"] == "toward_benign"
    assert c["values_all"] == [282.0]


def test_value_keyword_catches_a_wrong_number():
    ev = evidence()
    prose = ("The classifier predicts benign with probability 0.054. handles.nmutant is typical "
             "compared with training data (value 999.0), which pushed the decision toward benign.")
    claims = extract_claims(prose, FEATURES)
    score = score_rationale({"verdict": "benign", "probability_stated": 0.054,
                             "claims": claims, "summary": prose}, ev, 0.05, 0.02)
    per_claim = {c["feature"]: c for c in score["per_claim"]}
    assert per_claim["handles.nmutant"]["status"] == "contradicted"
    assert "value_ok" in per_claim["handles.nmutant"]["failed_checks"]
    assert not score["FS_pass"]


def test_compact_parenthesis_still_binds():
    """'handles.nfile (658.0, typical)' keeps working (study prose style)."""
    claims = extract_claims("handles.nfile (658.0, typical) also pushed toward ransomware.", FEATURES)
    assert claims[0]["feature"] == "handles.nfile"
    assert claims[0]["values_all"] == [658.0]
    assert claims[0]["direction"] == "toward_ransomware"


def test_percentiles_and_sentence_final_mentions():
    """Percentile numbers are not values; a mention ending the sentence is still found."""
    assert find_numbers("handles.nmutant sits at the 48th percentile of training data.") == []
    claims = extract_claims("The level of handles.nmutant is typical.", FEATURES)
    assert [c["feature"] for c in claims] == ["handles.nmutant"]
    assert len(split_sentences("One. Two.")) == 2
