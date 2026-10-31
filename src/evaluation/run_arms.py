"""Extraction + faithfulness scoring + causal tests for every arm (spec 8).

Arms: T, P0, P2, S2, S2R, P0R (generated) and X (negative control: S2
rationale paired with another sample's evidence of the same verdict,
spec 8.3.1). P0R is the extra prose-repair arm added so RQ2 is measurable.

Writes outputs/faithfulness/{arm}.jsonl + {arm}_summary.json + combined
all_arms_summary.{json,csv}. Prose arms go through the rule extractor with the
LLM fallback (which may emit agent_queue requests -> exit 20).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np
import pandas as pd

from src.faithfulness.causal import compute_causal, summarize_causal
from src.faithfulness.extract_llm import extract_claims_smart
from src.faithfulness.extract_rules import infer_verdict_and_probability
from src.faithfulness.judge import run_judge
from src.faithfulness.metrics import aggregate, sensitivity_table
from src.faithfulness.verify import score_rationale
from src.llm.base import WaitingForAgent, get_backend
from src.models.calibrate import load
from src.utils.config import Paths, ensure_dirs, load_config, now_iso
from src.utils.logging import setup_logger
from src.utils.seed import set_seed

log = setup_logger("run_arms")

ARMS = ["T", "P0", "P2", "S2", "S2R", "P0R", "X"]
# P0R is prose too: its repaired text goes through the same extractor + verifier.
PROSE_ARMS = {"P0", "P2", "P0R"}


def load_jsonl(path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[rec["sample_id"]] = rec
    return out


def build_mismatch_pairs(index: dict) -> dict[str, str]:
    """For each sample, the id of another sample with the SAME evidence verdict."""
    ids = index["sample_ids"]
    verdicts = index["verdicts"]
    pairs: dict[str, str] = {}
    for v in ("ransomware", "benign"):
        group = sorted([s for s in ids if verdicts.get(s) == v])
        if len(group) < 2:
            continue
        for i, sid in enumerate(group):
            pairs[sid] = group[(i + 1) % len(group)]
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--llm-extract", action="store_true", default=True,
                    help="use the LLM extractor fallback for prose (default on)")
    ap.add_argument("--no-llm-extract", dest="llm_extract", action="store_false")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config()
    set_seed(int(cfg.seed))
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    index = json.loads((Paths.evidence / "index.json").read_text())
    sample_ids = index["sample_ids"]
    evidence = {sid: json.loads((Paths.evidence / f"{sid}.json").read_text())
                for sid in sample_ids}
    strata = index.get("strata", {})

    test = pd.read_parquet(Paths.processed / "test.parquet").set_index("row_uid")
    features = json.loads((Paths.processed / "features.json").read_text())
    train = pd.read_parquet(Paths.processed / "train.parquet")
    train_X = train[features].to_numpy(dtype=np.float32)
    train_y = train["y_binary"].to_numpy(dtype=int)
    stats = json.loads((Paths.models / "train_stats.json").read_text())
    medians = np.array([stats["features"][f]["median"] for f in features], dtype=np.float32)

    uid_of = {sid: sid.replace("test_", "row_") for sid in sample_ids}
    X_eval = test.loc[[uid_of[sid] for sid in sample_ids]][features].to_numpy(dtype=np.float32)
    calibrator = load(Paths.models / "calibrator.joblib")
    npz = np.load(Paths.outputs / "shap_eval.npz", allow_pickle=True)
    sv = npz["shap"]

    backend = get_backend(cfg)
    extract_ctx = {"calls": 0, "cap": 60}
    results: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}

    s2 = load_jsonl(Paths.rationales / "S2.jsonl")
    mismatch = build_mismatch_pairs(index)

    for arm in arms:
        out_path = Paths.faithfulness / f"{arm}.jsonl"
        if out_path.exists() and not args.force and arm != "X":
            log.info("arm %s already scored (use --force)", arm)
            recs = load_jsonl(out_path)
            results[arm] = [recs[sid] for sid in sample_ids if sid in recs]
            summary[arm] = json.loads((Paths.faithfulness / f"{arm}_summary.json").read_text())
            continue

        recs = load_jsonl(Paths.rationales / f"{arm}.jsonl") if arm != "X" else s2
        if not recs:
            log.warning("arm %s: no rationales found; skipping", arm)
            continue

        per_sample: list[dict] = []
        misses = 0
        batch = int(getattr(backend, "batch_size", 25))
        for sid in sample_ids:
            ev = evidence[sid]
            rec = recs.get(sid)
            if rec is None:
                continue
            extraction = {"extractor": "given"}
            prose = rec.get("text") or ""
            if arm in PROSE_ARMS:
                try:
                    claims, extraction = extract_claims_smart(
                        prose, ev, backend, allow_llm=args.llm_extract, ctx=extract_ctx)
                except WaitingForAgent:
                    misses += 1
                    if misses >= batch:
                        status = backend.emit_batch()
                        print("\n" + "=" * 72)
                        print(f"WAITING_FOR_AGENT — extraction fallback (arm {arm})")
                        print(f"unanswered requests: {status['n_unanswered_total']} "
                              f"(active batch: {status['n_in_active_batch']})")
                        print(f"answers -> {status['responses_file']}")
                        print("Then re-run the same `make verify` command.")
                        print("=" * 72 + "\n")
                        sys.exit(20)
                    continue
                verdict, prob = infer_verdict_and_probability(prose)
                rationale = {"verdict": verdict, "probability_stated": prob,
                             "claims": claims, "summary": prose}
            elif arm == "X":
                partner_id = mismatch.get(sid)
                partner = s2.get(partner_id) if partner_id else None
                if partner is None or partner.get("rationale") is None:
                    continue
                rationale = partner["rationale"]
                prose = partner.get("text") or ""
                extraction = {"extractor": "given", "paired_from": partner_id}
            else:
                rationale = rec.get("rationale")
                if rationale is None:  # unparseable structured output
                    rationale = {"verdict": None, "claims": [], "summary": prose}
            score = score_rationale(rationale, ev, float(cfg.faithfulness.value_tolerance),
                                    float(cfg.faithfulness.prob_tolerance))
            cited: list[str] = []
            for c in score["per_claim"]:
                f = c["feature"]
                if f not in cited:
                    cited.append(f)
            per_sample.append({
                "sample_id": sid, "arm": arm, "stratum": strata.get(sid),
                "verdict": ev["verdict"], "text": prose,
                "verdict_stated": rationale.get("verdict"),
                "probability_stated": rationale.get("probability_stated"),
                "extraction": extraction, "cited_features": cited,
                "score": score,
            })

        if misses:
            status = backend.emit_batch()
            print("\n" + "=" * 72)
            print(f"WAITING_FOR_AGENT — extraction fallback (arm {arm})")
            print(f"unanswered requests: {status['n_unanswered_total']} "
                  f"(active batch: {status['n_in_active_batch']})")
            print(f"answers -> {status['responses_file']}")
            print("Then re-run the same `make verify` command.")
            print("=" * 72 + "\n")
            sys.exit(20)
        if not per_sample:
            log.warning("arm %s: zero scored samples", arm)
            continue

        causal = compute_causal(calibrator, X_eval, [r["sample_id"] for r in per_sample],
                                [r["cited_features"] for r in per_sample], features,
                                train_X, medians, shap_values=sv, train_y=train_y,
                                n_random=int(cfg.faithfulness.causal_random_draws),
                                n_marginal=int(cfg.faithfulness.causal_marginal_draws),
                                seed=int(cfg.seed))
        for r in per_sample:
            r["causal"] = causal.get(r["sample_id"])

        with out_path.open("w", encoding="utf-8") as fh:
            for r in per_sample:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        scores = [r["score"] for r in per_sample]
        agg = aggregate(scores)
        agg["causal"] = summarize_causal({r["sample_id"]: r["causal"] for r in per_sample
                                          if r.get("causal")})
        agg["sensitivity"] = sensitivity_table(scores)
        agg["n_extraction_fallbacks"] = sum(
            1 for r in per_sample if r["extraction"].get("fallback"))
        agg["generated_at"] = now_iso()
        (Paths.faithfulness / f"{arm}_summary.json").write_text(json.dumps(agg, indent=2))
        summary[arm] = agg
        results[arm] = per_sample
        log.info("arm %s: n=%d FS=%.3f precision=%.3f halluc=%.3f contra=%.3f coverage=%.3f "
                 "pass=%.2f lift=%.3f", arm, agg["n"], agg["FS"], agg["claim_precision"],
                 agg["hallucination_rate"], agg["contradiction_rate"], agg["evidence_coverage"],
                 agg["FS_pass_rate"], (agg["causal"] or {}).get("causal_lift_mean", float("nan")))

    # combined summary
    if summary:
        (Paths.faithfulness / "all_arms_summary.json").write_text(json.dumps(summary, indent=2))
        cols = ["n", "FS", "claim_precision", "claim_precision_lenient", "hallucination_rate",
                "contradiction_rate", "evidence_coverage", "top1_mentioned_rate",
                "verdict_consistency_rate", "FS_pass_rate", "no_claims_rate",
                "remediation_rate", "family_speculation_rate", "mean_claims"]
        with (Paths.faithfulness / "all_arms_summary.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            causal_cols = ["causal_lift_mean", "comprehensiveness_median_mean",
                           "comprehensiveness_random_mean_mean",
                           "comprehensiveness_topn_shap_mean", "sufficiency_median_mean",
                           "comprehensiveness_marginal_mean", "comprehensiveness_otherclass_mean",
                           "causal_lift_otherclass_mean"]
            w.writerow(["arm"] + cols + causal_cols)
            for arm, agg in summary.items():
                causal = agg.get("causal") or {}
                w.writerow([arm] + [agg.get(c) for c in cols] +
                           [causal.get(c) for c in causal_cols])
        log.info("wrote %s", Paths.faithfulness / "all_arms_summary.csv")

    judge_out = run_judge([{"sample_id": r["sample_id"], "evidence": evidence[r["sample_id"]],
                            "per_claim": r["score"]["per_claim"]}
                           for r in results.get("S2", [])], cfg)
    log.info("judge: %s", judge_out.get("status"))

    # ---- acceptance bookkeeping (spec section 0.7) ---------------------- #
    from src.utils.checks import record_check

    t_fs = summary.get("T", {}).get("FS")
    record_check("phase6", "template arm FS >= 0.99", bool((t_fs or 0) >= 0.99),
                 f"T FS={t_fs:.3f}" if t_fs is not None else "T arm missing")
    s2r = json.loads((Paths.faithfulness / "S2R_summary.json").read_text()) \
        if (Paths.faithfulness / "S2R_summary.json").exists() else {}
    p0r_path = Paths.faithfulness / "P0R_summary.json"
    p0r = json.loads(p0r_path.read_text()) if p0r_path.exists() else {}
    if s2r:
        detail = f"S2R FS={s2r.get('FS', float('nan')):.3f}"
        if p0r:
            detail += f"; P0R FS={p0r.get('FS', float('nan')):.3f}"
        record_check("phase8", "repair before/after table produced", True, detail)
    else:
        record_check("phase8", "repair before/after table produced", False, "S2R arm missing")
    record_check("phase5", "agent_queue round trip used",
                 any(r["extraction"].get("fallback") for arm in results for r in results[arm]),
                 "LLM extractor fallback ran on agent_queue answers")
    log.info("checks: %s", {"template_FS_ge_099": bool((t_fs or 0) >= 0.99),
                             "repair_table": bool(s2r)})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except WaitingForAgent:
        sys.exit(20)
