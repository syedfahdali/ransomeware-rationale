"""Metric validity tests (spec section 8.3) + `make validity`.

1. Corrupted-evidence control: rationale paired with another sample's evidence
   (arm X) must lose FS relative to the original (arm S2).
2. Synthetic perturbations of faithful rationales (template arm): flip a
   direction, swap in an out-of-top-k feature, change a value by +30%, drop the
   top feature. Detection recall must be >= 95%.
3. Fluency-vs-faithfulness decoupling: length, Flesch-Kincaid grade, ROUGE-L
   against the template reference; Spearman correlation with FS; and the same
   fluency metrics for corrupted rationales (high fluency, low FS).
4. Spot check: 10 random rationale/evidence/verifier triples to outputs/spot_check.md.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys

import numpy as np

from src.faithfulness.extract_rules import extract_claims
from src.faithfulness.verify import score_rationale
from src.generation.template_baseline import template_rationale
from src.utils.config import Paths, ensure_dirs, load_config
from src.utils.logging import setup_logger
from src.utils.seed import set_seed

log = setup_logger("validity")


# --------------------------------------------------------------------------- #
def load_arm(arm: str) -> dict[str, dict]:
    p = Paths.faithfulness / f"{arm}.jsonl"
    out = {}
    if not p.exists():
        return out
    with p.open() as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["sample_id"]] = rec
    return out


def load_evidence() -> dict[str, dict]:
    index = json.loads((Paths.evidence / "index.json").read_text())
    return {sid: json.loads((Paths.evidence / f"{sid}.json").read_text())
            for sid in index["sample_ids"]}


# --------------------------------------------------------------------------- #
def perturb(rationale: dict, ev: dict, kind: str, rng: np.random.RandomState) -> dict:
    r = copy.deepcopy(rationale)
    topk = [f["name"] for f in ev["top_features"]]
    if kind == "flip_direction" and r["claims"]:
        c = r["claims"][0]
        c["direction"] = ("toward_benign" if c["direction"] == "toward_ransomware"
                          else "toward_ransomware")
    elif kind == "out_of_topk_feature" and r["claims"]:
        outside = [f for f in ev["all_feature_names"] if f not in topk]
        c = r["claims"][min(1, len(r["claims"]) - 1)]
        c["feature"] = rng.choice(outside)
    elif kind == "value_30pct" and r["claims"]:
        c = r["claims"][0]
        base = c.get("value_stated") or float(ev["top_features"][0]["value"])
        c["value_stated"] = float(base) * 1.30
    elif kind == "drop_top" and r["claims"]:
        top_name = topk[0]
        r["claims"] = [c for c in r["claims"] if c["feature"] != top_name] or r["claims"][1:]
    return r


def perturbation_recall(t_records: dict[str, dict], evidence: dict[str, dict],
                        cfg) -> dict:
    rng = np.random.RandomState(42)
    kinds = ["flip_direction", "out_of_topk_feature", "value_30pct", "drop_top"]
    out = {}
    for kind in kinds:
        detected = total = 0
        for sid, rec in t_records.items():
            ev = evidence.get(sid)
            if ev is None:
                continue
            # The T arm is deterministic, so the faithful base rationale can be
            # rebuilt exactly when the stored record keeps only prose + score.
            base = rec.get("rationale") or template_rationale(ev)
            before = score_rationale(base, ev, float(cfg.faithfulness.value_tolerance),
                                     float(cfg.faithfulness.prob_tolerance))
            after = score_rationale(perturb(base, ev, kind, rng), ev,
                                    float(cfg.faithfulness.value_tolerance),
                                    float(cfg.faithfulness.prob_tolerance))
            detected += int(after["FS"] < before["FS"] or (before["FS_pass"] and not after["FS_pass"]))
            total += 1
        out[kind] = {"recall": detected / max(total, 1), "n": total,
                     "passed": (detected / max(total, 1)) >= 0.95}
    out["all_passed"] = all(v.get("passed", False) for k, v in out.items()
                            if isinstance(v, dict))
    return out


# --------------------------------------------------------------------------- #
_SYLL = re.compile(r"[aeiouy]+")


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    n = len(_SYLL.findall(w))
    if w.endswith("e") and n > 1:
        n -= 1
    return max(n, 1)


def flesch_kincaid(text: str) -> float:
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[A-Za-z]+", text)
    if not sentences or not words:
        return 0.0
    syl = sum(_syllables(w) for w in words)
    return 0.39 * (len(words) / len(sentences)) + 11.8 * (syl / len(words)) - 15.59


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[-1], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(hyp: str, ref: str) -> float:
    h = re.findall(r"[a-z0-9_]+", hyp.lower())
    r = re.findall(r"[a-z0-9_]+", ref.lower())
    if not h or not r:
        return 0.0
    lcs = _lcs(h, r)
    prec = lcs / len(h)
    rec = lcs / len(r)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def fluency_metrics(record: dict, reference_text: str) -> dict:
    text = record.get("text") or ""
    return {
        "n_tokens": len(re.findall(r"\S+", text)),
        "n_sentences": max(len([s for s in re.split(r"[.!?]+", text) if s.strip()]), 1),
        "flesch_kincaid": flesch_kincaid(text),
        "rouge_l_vs_template": rouge_l(text, reference_text),
    }


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    from scipy.stats import spearmanr
    rho, _p = spearmanr(x, y)
    return float(rho)


# --------------------------------------------------------------------------- #
def extraction_accuracy(gold_path, arm_records: dict, evidence: dict,
                        features: list[str], arm: str | None = None) -> dict:
    if not gold_path.exists():
        return {"status": "missing", "note": f"{gold_path} not found"}
    gold = [json.loads(line) for line in gold_path.read_text().splitlines() if line.strip()]
    # The gold set may label the same sample in several arms (different prose),
    # so only items labelled for this arm may be scored against it.
    if arm is not None:
        gold = [g for g in gold if (g.get("arm") or arm) == arm]
    tp = fp = fn = 0
    per_item = []
    for item in gold:
        sid = item["sample_id"]
        rec = arm_records.get(sid)
        if rec is None:
            continue
        claims = extract_claims(rec.get("text") or "", features)
        got = {(c["feature"], c.get("direction")) for c in claims}
        want = {(g["feature"], (g.get("direction") or None)) for g in item["gold_claims"]}
        tp_i = len(got & want)
        fp_i = len(got - want)
        fn_i = len(want - got)
        tp += tp_i
        fp += fp_i
        fn += fn_i
        per_item.append({"sample_id": sid, "arm": item.get("arm"), "tp": tp_i, "fp": fp_i,
                         "fn": fn_i, "got": [list(g) for g in sorted(got)],
                         "want": [list(w) for w in sorted(want)]})
    prec = tp / max(tp + fp, 1)
    rec_ = tp / max(tp + fn, 1)
    return {"status": "ran", "n_rationales": len(per_item), "precision": prec, "recall": rec_,
            "f1": 2 * prec * rec_ / max(prec + rec_, 1e-9), "items": per_item,
            "gold_source": "agent-drafted (not independently annotated)"}


# --------------------------------------------------------------------------- #
def spot_check(records: dict, evidence: dict, out_path, n: int = 10,
               seed: int = 42) -> None:
    rng = np.random.RandomState(seed)
    ids = sorted(records)
    pick = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
    lines = ["# Spot check — rationale / evidence / verifier triples", "",
             f"Random sample (seed={seed}); review by hand.", ""]
    for i in pick:
        sid = ids[i]
        r = records[sid]
        ev = evidence[sid]
        lines += [f"## {sid} (arm {r['arm']}, stratum {r.get('stratum')})", "",
                  "**Evidence table**", "", "```",
                  "\n".join(f"{f['rank']}. {f['name']} = {f['value']:.5g} | {f['qualitative_level']}"
                            f" | {f['direction']} | {f['percentile_vs_train']:.1f}pct"
                            for f in ev["top_features"]), "```", "",
                  f"**Rationale** ({r['text'].strip()[:1200]})", "", "",
                  "**Verifier**", "",
                  f"- stated verdict/prob: {r.get('verdict_stated')} / {r.get('probability_stated')}",
                  f"- FS={r['score']['FS']:.3f} pass={r['score']['FS_pass']} "
                  f"precision={r['score']['claim_precision']:.2f} "
                  f"coverage={r['score']['evidence_coverage']:.2f}",
                  "- claims:"]
        for c in r["score"]["per_claim"]:
            lines.append(f"  - `{c['feature']}` -> {c['status']} "
                         f"(failed: {', '.join(c['failed_checks']) or 'none'})")
        lines.append("")
    out_path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config()
    set_seed(int(cfg.seed))
    evidence = load_evidence()
    features = json.loads((Paths.processed / "features.json").read_text())

    t_recs = load_arm("T")
    s2 = load_arm("S2")
    x = load_arm("X")
    p0 = load_arm("P0")
    p2 = load_arm("P2")

    out: dict = {}

    # 1. corrupted-evidence control
    if s2 and x:
        common = sorted(set(s2) & set(x))
        fs_orig = float(np.mean([s2[s]["score"]["FS"] for s in common]))
        fs_corrupt = float(np.mean([x[s]["score"]["FS"] for s in common]))
        out["corrupted_evidence_control"] = {
            "n": len(common), "FS_original_S2": fs_orig, "FS_corrupted_X": fs_corrupt,
            "FS_drop": fs_orig - fs_corrupt,
            "hallucination_rate_X": float(np.mean(
                [x[s]["score"]["hallucination_rate"] for s in common])),
            "contradiction_rate_X": float(np.mean(
                [x[s]["score"]["contradiction_rate"] for s in common])),
        }

    # 2. perturbation recall
    if t_recs:
        out["perturbations"] = perturbation_recall(t_recs, evidence, cfg)

    # 3. fluency vs faithfulness
    if s2 and t_recs:
        # Pool S2 with the prose arms (P0) and the corrupted control (X): the S2
        # FS distribution is constant at the ceiling, so a correlation computed
        # on S2 alone is undefined (nan). Pooling is what makes RQ3 answerable.
        rows = []
        for arm_recs in (s2, p0, x):
            for sid, rec in arm_recs.items():
                t = t_recs.get(sid)
                if t is None:
                    continue
                fm = fluency_metrics(rec, t.get("text") or "")
                rows.append({"sample_id": sid, "arm": rec.get("arm"),
                             "FS": rec["score"]["FS"], **fm})
        if rows:
            fs = [r["FS"] for r in rows]
            out["fluency"] = {
                "n": len(rows),
                "spearman_FS_vs_tokens": spearman(fs, [r["n_tokens"] for r in rows]),
                "spearman_FS_vs_flesch_kincaid": spearman(fs, [r["flesch_kincaid"] for r in rows]),
                "spearman_FS_vs_rouge_l": spearman(fs, [r["rouge_l_vs_template"] for r in rows]),
                "mean_tokens": float(np.mean([r["n_tokens"] for r in rows])),
                "mean_flesch_kincaid": float(np.mean([r["flesch_kincaid"] for r in rows])),
                "mean_rouge_l_vs_template": float(np.mean([r["rouge_l_vs_template"] for r in rows])),
                "rows": rows,
            }
            # Cleaner RQ3 view: within a single natural arm only P0 has FS
            # variance (S2 sits at the ceiling), so the P0-only correlations
            # are the non-confounded test of a fluency proxy vs faithfulness.
            p0_rows = [r for r in rows if r["arm"] == "P0"]
            if len(p0_rows) >= 3:
                fs_p0 = [r["FS"] for r in p0_rows]
                out["fluency"].update({
                    "P0_only_n": len(p0_rows),
                    "P0_only_FS_vs_tokens": spearman(fs_p0, [r["n_tokens"] for r in p0_rows]),
                    "P0_only_FS_vs_flesch_kincaid": spearman(
                        fs_p0, [r["flesch_kincaid"] for r in p0_rows]),
                    "P0_only_FS_vs_rouge_l": spearman(
                        fs_p0, [r["rouge_l_vs_template"] for r in p0_rows]),
                })
        # same fluency metrics for the corrupted arm
        if x:
            rows_x = []
            for sid, rec in x.items():
                t = t_recs.get(sid)
                if t is None:
                    continue
                fm = fluency_metrics(rec, t.get("text") or "")
                rows_x.append({"sample_id": sid, "FS": rec["score"]["FS"], **fm})
            if rows_x:
                out["fluency"]["corrupted_mean_tokens"] = float(
                    np.mean([r["n_tokens"] for r in rows_x]))
                out["fluency"]["corrupted_mean_rouge_l_vs_template"] = float(
                    np.mean([r["rouge_l_vs_template"] for r in rows_x]))
                out["fluency"]["corrupted_mean_FS"] = float(np.mean([r["FS"] for r in rows_x]))
                out["fluency"]["rows_corrupted"] = rows_x

    # 4. extractor gold accuracy
    gold_path = Paths.eval / "extractor_gold.jsonl"
    out["extractor_accuracy"] = {}
    for arm, recs in (("P0", p0), ("P2", p2)):
        if recs:
            out["extractor_accuracy"][arm] = extraction_accuracy(gold_path, recs, evidence,
                                                                 features, arm=arm)

    # 5. spot check (S2 arm by default)
    if s2:
        spot_check(s2, evidence, Paths.outputs / "spot_check.md")

    # ---- acceptance bookkeeping (spec section 0.7) ---------------------- #
    from src.utils.checks import record_check

    pert = out.get("perturbations") or {}
    record_check("phase7", "perturbation detection recall >= 0.95",
                 bool(pert.get("all_passed")),
                 ", ".join(f"{k}={v['recall']:.2f}" for k, v in pert.items()
                           if isinstance(v, dict)) or "not run")
    ctrl = out.get("corrupted_evidence_control") or {}
    record_check("phase7", "corrupted-evidence control drops FS",
                 bool((ctrl or {}).get("FS_drop", 0) >= 0.2),
                 f"S2 {ctrl.get('FS_original_S2', float('nan')):.3f} -> "
                 f"X {ctrl.get('FS_corrupted_X', float('nan')):.3f}"
                 f" (drop {ctrl.get('FS_drop', float('nan')):.3f})" if ctrl else "not run")
    acc = out.get("extractor_accuracy") or {}
    record_check("phase7", "extractor gold set produced",
                 all(v.get("status") == "ran" for v in acc.values()) and bool(acc),
                 "; ".join(f"{a} P/R={v.get('precision', float('nan')):.2f}/"
                           f"{v.get('recall', float('nan')):.2f}" for a, v in acc.items())
                 or "not run")

    (Paths.outputs / "validity.json").write_text(json.dumps(out, indent=2))
    log.info("validity summary: %s", json.dumps({k: v for k, v in out.items()
                                                 if k not in ("fluency",)}, indent=2)[:2000])
    if "perturbations" in out:
        log.info("perturbation recall: %s", {k: v for k, v in out["perturbations"].items()
                                             if isinstance(v, dict)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
