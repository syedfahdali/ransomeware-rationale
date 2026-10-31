"""Laya decision layer: the contract that makes it safe to ship at all.

Three things must hold on a machine without the optional `laya` extra, and they
are the whole reason the layer is allowed anywhere near the study:

1. Soft-fail. Every entry point returns ``{"available": False, ...}`` instead of
   raising, so `make all` and `make check` behave identically with and without it.
2. The stage is a no-op when disabled, and says so.
3. FS is untouched. `score_rationale` must produce a byte-identical score with
   Laya's extra fields present on the claims -- the `laya_fs_independence`
   acceptance check, asserted here at unit level too.
"""
from __future__ import annotations

import importlib.util
import json

import pytest

from src.faithfulness import laya_gate
from src.faithfulness.verify import score_rationale
from src.utils.config import Paths, load_config

HAVE_LAYA = importlib.util.find_spec("laya") is not None


@pytest.fixture
def no_laya(monkeypatch):
    """Force the absent-dependency branch even when `laya` IS installed.

    The soft-fail contract is the whole reason this optional layer is safe to
    ship, so it must be exercised on every host — not skipped on the hosts that
    happen to have the extra installed.
    """
    monkeypatch.setattr(laya_gate, "available",
                        lambda: (False, "simulated: the `laya` package is not installed"))
    monkeypatch.setattr(laya_gate, "_LAYA", None)
    monkeypatch.setattr(laya_gate, "_ROUTER", None)
    monkeypatch.setattr(laya_gate, "_ROUTER_KEY", None)
    yield


def evidence() -> dict:
    return {
        "sample_id": "test_000001",
        "verdict": "ransomware",
        "predicted_label": 1,
        "probability_ransomware": 0.93,
        "threshold": 0.5,
        "base_value": 0.0,
        "k": 2,
        "shap_mass_covered_by_top_k": 0.9,
        "all_feature_names": ["handles.nhandles", "dlllist.ndlls", "svcscan.nservices"],
        "top_features": [
            {"rank": 1, "name": "handles.nhandles", "value": 42000.0, "shap": 2.5,
             "direction": "toward_ransomware", "percentile_vs_train": 99.0,
             "train_median": 100.0, "z_score": 3.0, "description": "handles",
             "qualitative_level": "very_high"},
            {"rank": 2, "name": "dlllist.ndlls", "value": 80.0, "shap": 1.1,
             "direction": "toward_ransomware", "percentile_vs_train": 90.0,
             "train_median": 40.0, "z_score": 1.5, "description": "dlls",
             "qualitative_level": "high"},
        ],
    }


def rationale() -> dict:
    claims = [
        {"feature": "handles.nhandles", "direction": "toward_ransomware",
         "level": "very_high", "value_stated": 42000.0, "text": "handles are very high"},
        {"feature": "svcscan.nservices", "direction": "toward_benign",
         "level": "low", "value_stated": 10.0, "text": "few services"},
    ]
    return {"verdict": "ransomware", "probability_stated": 0.93, "claims": claims,
            "summary": "explanation"}


# --------------------------------------------------------------------------- #
# 1. soft-fail
# --------------------------------------------------------------------------- #
def test_soft_fail_contract_without_laya(no_laya):
    cfg = load_config()
    ok, reason = laya_gate.available()
    assert ok is False and reason

    for out in (laya_gate.gate_verdict(evidence(), cfg),
                laya_gate.judge_claims(rationale()["claims"], evidence(), cfg),
                laya_gate.review_repair(rationale()["claims"][0], None, evidence(), cfg)):
        assert out["available"] is False
        assert out["reason"]


def test_soft_fail_carries_no_laya_fields(no_laya):
    """A soft-failed result must not invent a verdict or a confidence."""
    cfg = load_config()
    gate = laya_gate.gate_verdict(evidence(), cfg)
    assert "laya_verdict" not in gate and "confidence" not in gate
    judged = laya_gate.judge_claims(rationale()["claims"], evidence(), cfg)
    assert judged["per_claim"] == []
    assert judged.get("agreement_rate") is None


def test_ask_is_total_and_never_raises(no_laya):
    cfg = load_config()
    r = laya_gate.ask("some state", laya_gate.gate_question(), cfg)
    assert r["available"] is False and r["answers"] == {}


def test_router_is_none_not_an_exception(no_laya):
    assert laya_gate.get_router(load_config()) is None


def test_fs_independence_does_not_need_laya_at_all():
    """The FS guarantee must hold with the layer installed *or* absent."""
    cfg = load_config()
    ev, rat = evidence(), rationale()
    score = score_rationale(rat, ev, cfg.faithfulness.value_tolerance,
                            cfg.faithfulness.prob_tolerance)
    stored = {"sample_id": ev["sample_id"], "verdict": rat["verdict"],
              "probability_stated": rat["probability_stated"], "score": score}
    ok, detail = laya_gate.fs_independence({"TEST": {ev["sample_id"]: stored}},
                                           {ev["sample_id"]: ev}, cfg)
    assert ok, detail


def test_environment_state_is_reported_consistently():
    """Whatever this host has, `available()` and the layer must agree."""
    ok, reason = laya_gate.available()
    assert ok == HAVE_LAYA, f"available()={ok} but find_spec says {HAVE_LAYA} ({reason})"


def test_available_returns_a_pair():
    ok, reason = laya_gate.available()
    assert isinstance(ok, bool) and isinstance(reason, str)


# --------------------------------------------------------------------------- #
# 2. stage no-op
# --------------------------------------------------------------------------- #
def test_stage_is_a_noop_when_disabled(tmp_path, monkeypatch):
    """Must be a no-op *and* must not clobber a real run's `summary.json`."""
    cfg = load_config()
    assert cfg.laya.get("enabled") is False, "the layer must ship OFF by default"
    # point the stage at scratch files: the acceptance table and the artifacts
    # a real run left behind are not the test's to overwrite
    from src.utils import checks as checks_mod

    monkeypatch.setattr(Paths, "laya", tmp_path)
    monkeypatch.setattr(checks_mod, "CHECKS_PATH", tmp_path / "phase_checks.json")
    (tmp_path / "summary.json").write_text('{"status": "ran", "arms": {}}')

    rc = laya_gate._skipped(cfg, "test: disabled")
    assert rc == 0
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["status"] == "skipped"
    assert payload["enabled"] is False
    # the skipped branch must still re-affirm any preserved full-run record
    recorded = json.loads((tmp_path / "phase_checks.json").read_text())
    names = {c["name"] for c in recorded}
    assert {"laya_available_fallback", "laya_gate_latency",
            "laya_fs_independence"} <= names


# --------------------------------------------------------------------------- #
# 3. FS independence
# --------------------------------------------------------------------------- #
def test_adding_laya_fields_to_claims_cannot_move_fs():
    cfg = load_config()
    ev, rat = evidence(), rationale()
    vt, pt = cfg.faithfulness.value_tolerance, cfg.faithfulness.prob_tolerance

    base = score_rationale(rat, ev, vt, pt)
    annotated = [dict(c, laya_status="contradicted", laya_confidence=0.99)
                 for c in rat["claims"]]
    with_laya = score_rationale(dict(rat, claims=annotated), ev, vt, pt)

    for key in ("FS", "claim_precision", "evidence_coverage", "verdict_ok", "n_claims"):
        assert base[key] == with_laya[key], f"{key} moved when Laya fields were added"
    assert base["status_counts"] == with_laya["status_counts"]
    assert [c["status"] for c in base["per_claim"]] == \
           [c["status"] for c in with_laya["per_claim"]]


def test_fs_independence_helper_round_trips_a_stored_record():
    cfg = load_config()
    ev, rat = evidence(), rationale()
    score = score_rationale(rat, ev, cfg.faithfulness.value_tolerance,
                            cfg.faithfulness.prob_tolerance)
    stored = {"sample_id": ev["sample_id"], "verdict": rat["verdict"],
              "probability_stated": rat["probability_stated"], "score": score}
    ok, detail = laya_gate.fs_independence({"TEST": {ev["sample_id"]: stored}},
                                           {ev["sample_id"]: ev}, cfg)
    assert ok, detail


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_collate_status_merges_the_two_not_in_evidence_states():
    assert laya_gate.collate_status("unsupported") == "not_in_evidence"
    assert laya_gate.collate_status("out_of_topk") == "not_in_evidence"
    assert laya_gate.collate_status("supported") == "supported"
    assert laya_gate.collate_status("contradicted") == "contradicted"


def test_render_claim_row_handles_listed_and_unlisted_features():
    ev = evidence()
    listed = laya_gate.render_claim_row(ev, "handles.nhandles")
    assert "42,000" in listed and "very_high" in listed and "99.0" in listed
    missing = laya_gate.render_claim_row(ev, "not.a.feature")
    assert "not among the ranked evidence features" in missing


def test_temperature_scaling_is_identity_at_one_and_monotone():
    assert laya_gate.apply_temperature(0.8, 1.0) == 0.8
    assert laya_gate.apply_temperature(0.8, 0.0) == 0.8
    hot = laya_gate.apply_temperature(0.9, 4.0)
    cold = laya_gate.apply_temperature(0.9, 0.25)
    assert 0.5 < hot < 0.9 < cold < 1.0


def test_questions_never_use_noul_and_stay_small():
    """Laya's own documented pitfall: `noul` latches onto its option labels."""
    for q in (laya_gate.gate_question(), laya_gate.claim_question(),
              laya_gate.repair_question()):
        (spec,) = q.values()
        assert spec["type"] == "choice"
        assert 2 <= len(spec["criteria"]) <= 3


def test_checkpoint_spec_maps_config_names():
    cfg = load_config()
    name, model_dir = laya_gate._checkpoint_spec(cfg)
    assert name == "typed-decisions" and model_dir is None


# --------------------------------------------------------------------------- #
# a partial run must not erase an earlier run's work
# --------------------------------------------------------------------------- #
def test_partial_run_keeps_the_other_arms_in_disagreements(tmp_path, monkeypatch):
    """`--arms X` writes only X's records; the spot-check file must keep the rest."""
    from src.utils.config import Paths

    monkeypatch.setattr(Paths, "laya", tmp_path)
    (tmp_path / "T_gate.jsonl").write_text(json.dumps(
        {"sample_id": "s1", "arm": "T", "agrees": True, "hard_disagreement": False,
         "latency_ms": 10.0}) + "\n")
    (tmp_path / "T_claims.jsonl").write_text(json.dumps(
        {"sample_id": "s1", "arm": "T", "n_compared": 2, "n_agreed": 2,
         "agreement_rate": 1.0, "agreement_rate_strict": 1.0,
         "median_claim_latency_ms": 5.0, "per_claim": []}) + "\n")

    # the arm this invocation is scoring (fresh, on disk, but not merged yet)
    x_gate = [{"sample_id": "s2", "arm": "X", "agrees": True,
               "hard_disagreement": False, "latency_ms": 12.0}]
    by_arm = {"X": laya_gate._arm_stats(x_gate, [])}

    stored = laya_gate._stored_by_arm(set(by_arm))
    assert set(stored) == {"T"}, "arms from earlier runs must be re-read from disk"

    out = tmp_path / "disagreements.md"
    laya_gate._write_disagreements(out, {**stored, **by_arm})
    text = out.read_text()
    assert "## arm T" in text and "## arm X" in text


def test_arm_stats_is_the_single_source_of_the_summary_row():
    gate = [{"sample_id": "s", "agrees": True, "hard_disagreement": False,
             "latency_ms": 10.0},
            {"sample_id": "s2", "agrees": False, "hard_disagreement": True,
             "latency_ms": 30.0}]
    claims = [{"n_compared": 4, "n_agreed": 3, "agreement_rate": 0.75,
               "agreement_rate_strict": 0.5}]
    row = laya_gate._arm_stats(gate, claims)
    assert row["n_samples"] == 2
    assert row["gate_agreement_rate"] == 0.5
    assert row["gate_hard_disagreements"] == 1
    assert row["LAR"] == 0.75 and row["LAR_strict"] == 0.5
    assert row["n_claims_compared"] == 4
    assert row["median_gate_latency_ms"] == 20.0


@pytest.mark.skipif(not (Paths.faithfulness / "X.jsonl").exists()
                   or not (Paths.faithfulness / "T.jsonl").exists(),
                   reason="verify stage has not stored every arm yet")
def test_fs_independence_covers_every_stored_arm_not_just_the_requested_one():
    """`--arms X` must still re-prove FS independence on all 7 arms on disk."""
    arms = laya_gate._all_faithfulness_arms()
    assert "X" in arms and "T" in arms, "every arm file belongs in the check"
    assert sum(len(v) for v in arms.values()) >= 350
    ok, detail = laya_gate.fs_independence(
        arms, laya_gate.load_evidence(), load_config())
    assert ok, detail
