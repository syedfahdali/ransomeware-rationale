"""Statistics for the evaluation (spec section 10).

- mean ± 95% bootstrap CI (10,000 resamples) per metric per arm
- paired bootstrap difference + Wilcoxon signed-rank for P2 vs S2 vs S2R
- stratified by TP/TN/FN/FP
"""
from __future__ import annotations

import json

import numpy as np

from src.utils.config import Paths
from src.utils.logging import setup_logger

log = setup_logger("stats")

N_BOOT = 10_000


def load_arm(arm: str) -> dict[str, dict]:
    path = Paths.faithfulness / f"{arm}.jsonl"
    out = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["sample_id"]] = rec
    return out


def bootstrap_ci(values, n_boot: int = N_BOOT, seed: int = 42, alpha: float = 0.05):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    means = [v[rng.randint(0, v.size, v.size)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap_diff(a: list[float], b: list[float], n_boot: int = N_BOOT,
                          seed: int = 42, alpha: float = 0.05):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return None
    d = a - b
    rng = np.random.RandomState(seed)
    means = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    out = {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
           "n_pairs": int(len(d))}
    try:
        from scipy.stats import wilcoxon
        if np.any(d != 0):
            stat, p = wilcoxon(d)
            out["wilcoxon"] = {"stat": float(stat), "p": float(p)}
        else:
            out["wilcoxon"] = {"stat": None, "p": 1.0}
    except Exception:  # noqa: BLE001
        out["wilcoxon"] = None
    return out


def metric_values(records: dict[str, dict], metric: str) -> dict[str, float]:
    return {sid: float(r["score"][metric]) for sid, r in records.items()}


def causal_values(records: dict[str, dict], key: str) -> dict[str, float]:
    out = {}
    for sid, r in records.items():
        c = r.get("causal") or {}
        v = c.get(key)
        if v is not None:
            out[sid] = float(v)
    return out


def build_stats(arms: list[str]) -> dict:
    records = {arm: load_arm(arm) for arm in arms}
    records = {k: v for k, v in records.items() if v}
    out: dict = {"arms": {}, "comparisons": {}, "strata": {}, "n_boot": N_BOOT}

    metrics = ["FS", "claim_precision", "claim_precision_lenient", "hallucination_rate",
               "contradiction_rate", "evidence_coverage", "FS_pass_rate"]
    for arm, recs in records.items():
        arm_out = {}
        for m in metrics:
            if m == "FS_pass_rate":
                vals = [1.0 if r["score"]["FS_pass"] else 0.0 for r in recs.values()]
            else:
                vals = list(metric_values(recs, m).values())
            mean, lo, hi = bootstrap_ci(vals)
            arm_out[m] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)}
        for key in ("causal_lift", "comprehensiveness_median",
                    "comprehensiveness_random_mean", "comprehensiveness_topn_shap",
                    "sufficiency_median", "comprehensiveness_marginal", "sufficiency_marginal",
                    "comprehensiveness_otherclass", "causal_lift_otherclass"):
            vals = list(causal_values(recs, key).values())
            mean, lo, hi = bootstrap_ci(vals)
            arm_out[key] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)}
        out["arms"][arm] = arm_out

    for a, b in (("P0", "P0R"), ("P0R", "P2"), ("P0", "P2"), ("P2", "S2"),
                 ("S2", "S2R"), ("S2", "X")):
        if a in records and b in records:
            common = sorted(set(records[a]) & set(records[b]))
            if not common:
                continue
            out["comparisons"][f"{a}_vs_{b}"] = {
                m: paired_bootstrap_diff([records[a][s]["score"][m] for s in common],
                                         [records[b][s]["score"][m] for s in common])
                for m in ("FS", "claim_precision", "hallucination_rate",
                          "contradiction_rate", "evidence_coverage")
            }

    for arm, recs in records.items():
        strata = {}
        for r in recs.values():
            st = r.get("stratum") or "unknown"
            strata.setdefault(st, []).append(r["score"])
        out["strata"][arm] = {
            st: {"n": len(ss), "FS": float(np.mean([s["FS"] for s in ss])),
                 "claim_precision": float(np.mean([s["claim_precision"] for s in ss])),
                 "hallucination_rate": float(np.mean([s["hallucination_rate"] for s in ss])),
                 "FS_pass_rate": float(np.mean([bool(s["FS_pass"]) for s in ss]))}
            for st, ss in strata.items()}

    (Paths.faithfulness / "stats.json").write_text(json.dumps(out, indent=2))
    log.info("wrote %s", Paths.faithfulness / "stats.json")
    return out


def main() -> int:
    arms = ["T", "P0", "P0R", "P2", "S2", "S2R", "X"]
    out = build_stats(arms)
    for arm, m in out["arms"].items():
        fs = m["FS"]
        log.info("%s: FS %.3f [%.3f, %.3f] pass %.3f lift %.3f", arm, fs["mean"],
                 fs["ci_low"], fs["ci_high"], m["FS_pass_rate"]["mean"],
                 m["causal_lift"]["mean"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
