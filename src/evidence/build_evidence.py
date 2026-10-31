"""Evidence builder (spec section 5).

For every eval sample: calibrated probability, raw margin, exact SHAP
(log-odds), top-k features with direction, train percentile, qualitative level
and glossary description. Validates against schemas/evidence.schema.json and
checks additivity: sum(shap) + base_value == raw margin (<= 1e-3).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

from src.evidence.shap_builder import additivity_error, compute_shap, raw_margin
from src.models.calibrate import calibrated_proba, load
from src.utils.checks import record_check
from src.utils.config import Paths, ensure_dirs, load_config
from src.utils.logging import setup_logger
from src.utils.seed import set_seed

log = setup_logger("build_evidence")

LEVELS = ["very_low", "low", "typical", "high", "very_high"]


def qualitative_level(pct: float) -> str:
    if pct < 5:
        return "very_low"
    if pct < 25:
        return "low"
    if pct <= 75:
        return "typical"
    if pct <= 95:
        return "high"
    return "very_high"


def percentile_vs_train(value: float, grid: list[float]) -> float:
    g = np.asarray(grid, dtype=float)
    if g.size != 101:
        raise ValueError("percentile grid must have 101 points")
    pct = float(np.interp(value, g, np.arange(101, dtype=float)))
    return float(np.clip(pct, 0.0, 100.0))


def load_glossary() -> dict:
    path = Paths.root / "src" / "evidence" / "feature_glossary.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)["features"]


def build_one(sample: dict, x_row: np.ndarray, sv_row: np.ndarray, base: float,
              p: float, k: int, features: list[str], stats: dict, glossary: dict,
              threshold: float = 0.5) -> dict:
    order = np.argsort(-np.abs(sv_row))[:k]
    mass_total = float(np.abs(sv_row).sum()) or 1.0
    mass_topk = float(np.abs(sv_row[order]).sum()) / mass_total
    top = []
    for rank, idx in enumerate(order, start=1):
        name = features[int(idx)]
        value = float(x_row[int(idx)])
        shap_v = float(sv_row[int(idx)])
        st = stats["features"][name]
        pct = percentile_vs_train(value, st["percentile_grid"])
        desc = (glossary.get(name) or {}).get("description") or f"Memory-forensics feature {name}"
        std = st["std"]
        top.append({
            "rank": rank,
            "name": name,
            "value": value,
            "shap": shap_v,
            "direction": "toward_ransomware" if shap_v > 0 else "toward_benign",
            "percentile_vs_train": pct,
            "train_median": float(st["median"]),
            "z_score": float((value - st["mean"]) / std) if std else 0.0,
            "description": desc,
            "qualitative_level": qualitative_level(pct),
        })
    verdict = "ransomware" if p >= threshold else "benign"
    return {
        "sample_id": sample["sample_id"],
        "verdict": verdict,
        "predicted_label": int(sample["pred"]),
        "probability_ransomware": float(p),
        "threshold": float(threshold),
        "base_value": float(base),
        "top_features": top,
        "shap_mass_covered_by_top_k": float(mass_topk),
        "k": int(k),
        "all_feature_names": features,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config()
    set_seed(int(cfg.seed))

    index_path = Paths.evidence / "index.json"
    if index_path.exists() and not args.force:
        log.info("evidence already built (use --force to redo)")
        return 0

    eval_path = Paths.eval / "eval_ids.json"
    if not eval_path.exists():
        log.info("eval set missing; selecting it now")
        from src.evaluation.select_eval_set import main as select_main
        select_main()

    eval_payload = json.loads(eval_path.read_text())
    eval_samples = eval_payload["samples"]
    features = json.loads((Paths.processed / "features.json").read_text())
    stats = json.loads((Paths.models / "train_stats.json").read_text())
    glossary = load_glossary()
    missing_desc = [f for f in features if f not in glossary]
    if missing_desc:
        log.warning("glossary missing %d features (neutral fallback used): %s",
                    len(missing_desc), missing_desc)

    test = pd.read_parquet(Paths.processed / "test.parquet").set_index("row_uid")
    booster = load(Paths.models / "clf.joblib")
    calibrator = load(Paths.models / "calibrator.joblib")

    rows = test.loc[[s["row_uid"] for s in eval_samples]]
    X = rows[features].to_numpy(dtype=np.float32)
    p = calibrated_proba(calibrator, X)
    margins = raw_margin(booster, X)

    shap_path = Paths.outputs / "shap_eval.npz"
    ids = [s["sample_id"] for s in eval_samples]
    if shap_path.exists() and not args.force:
        npz = np.load(shap_path, allow_pickle=True)
        if list(npz["sample_ids"]) == ids:
            sv, base_values, margins = npz["shap"], npz["base_values"], npz["margins"]
            log.info("loaded cached SHAP from %s", shap_path)
        else:
            log.info("cache ids differ; recomputing SHAP")
            sv, base_values, _impl = compute_shap(booster, X, features)
            np.savez_compressed(shap_path, sample_ids=np.array(ids), shap=sv,
                                base_values=base_values, margins=margins)
    else:
        sv, base_values, _impl = compute_shap(booster, X, features)
        np.savez_compressed(shap_path, sample_ids=np.array(ids), shap=sv,
                            base_values=base_values, margins=margins)

    err = additivity_error(sv, base_values, margins)
    log.info("SHAP additivity max error: %.3e", err)
    record_check("phase3", "SHAP additivity <= 1e-3", err <= 1e-3, f"max|sum(shap)+base-margin|={err:.2e}")

    import jsonschema
    schema = json.loads((Paths.schemas / "evidence.schema.json").read_text())
    written, invalid = 0, 0
    k = int(cfg.evidence.top_k)
    for i, sample in enumerate(eval_samples):
        ev = build_one(sample, X[i], sv[i], float(base_values[i]), float(p[i]),
                       k, features, stats, glossary)
        try:
            jsonschema.validate(ev, schema)
        except jsonschema.ValidationError as e:
            invalid += 1
            log.error("evidence schema violation for %s: %s", sample["sample_id"], e.message)
        (Paths.evidence / f"{sample['sample_id']}.json").write_text(json.dumps(ev, indent=2))
        written += 1
    index_path.write_text(json.dumps({
        "sample_ids": ids, "k": k, "n": len(ids),
        "verdicts": {s["sample_id"]: ("ransomware" if s["pred"] == 1 else "benign") for s in eval_samples},
        "strata": {s["sample_id"]: s["stratum"] for s in eval_samples},
        "additivity_max_error": err, "schema_invalid": invalid,
    }, indent=2))
    log.info("wrote %d evidence files (%d schema-invalid)", written, invalid)
    record_check("phase3", "evidence JSON validates against schema", invalid == 0,
                 f"{written} files, {invalid} invalid")
    if err > 1e-3 or invalid:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
