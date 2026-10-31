"""Pick two train-split examples (one ransomware, one benign) and render their
evidence tables, so the few-shot examples in prompts/fewshot.json can be
hand-written against real evidence. Drafts go to prompts/fewshot_draft.json.

Run after `make evidence` (needs models/ + train_stats.json).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.evidence.shap_builder import compute_shap, raw_margin
from src.generation.prompts import render_evidence_table
from src.models.calibrate import calibrated_proba, load
from src.utils.config import Paths, load_config
from src.utils.logging import setup_logger
from src.utils.seed import set_seed

log = setup_logger("make_fewshot")


def build_evidence_row(sample_id, x_row, sv_row, base, p, features, stats, glossary, k=6):
    order = np.argsort(-np.abs(sv_row))[:k]
    mass_total = float(np.abs(sv_row).sum()) or 1.0
    top = []
    for rank, idx in enumerate(order, start=1):
        name = features[int(idx)]
        value = float(x_row[int(idx)])
        st = stats["features"][name]
        grid = np.asarray(st["percentile_grid"], dtype=float)
        pct = float(np.clip(np.interp(value, grid, np.arange(101, dtype=float)), 0, 100))
        level = ("very_low" if pct < 5 else "low" if pct < 25 else "typical" if pct <= 75
                 else "high" if pct <= 95 else "very_high")
        top.append({
            "rank": rank, "name": name, "value": value, "shap": float(sv_row[int(idx)]),
            "direction": "toward_ransomware" if sv_row[int(idx)] > 0 else "toward_benign",
            "percentile_vs_train": pct, "train_median": float(st["median"]),
            "z_score": float((value - st["mean"]) / st["std"]) if st["std"] else 0.0,
            "description": (glossary.get(name) or {}).get("description", f"feature {name}"),
            "qualitative_level": level,
        })
    return {
        "sample_id": sample_id, "verdict": "ransomware" if p >= 0.5 else "benign",
        "predicted_label": int(p >= 0.5), "probability_ransomware": float(p),
        "threshold": 0.5, "base_value": float(base), "top_features": top,
        "shap_mass_covered_by_top_k": float(np.abs(sv_row[order]).sum() / mass_total),
        "k": k, "all_feature_names": features,
    }


def main() -> int:
    cfg = load_config()
    set_seed(int(cfg.seed))
    features = json.loads((Paths.processed / "features.json").read_text())
    stats = json.loads((Paths.models / "train_stats.json").read_text())
    import yaml
    glossary = yaml.safe_load((Paths.root / "src" / "evidence" / "feature_glossary.yaml").read_text())["features"]
    train = pd.read_parquet(Paths.processed / "train.parquet")
    booster = load(Paths.models / "clf.joblib")
    calibrator = load(Paths.models / "calibrator.joblib")

    X = train[features].to_numpy(dtype=np.float32)
    p = calibrated_proba(calibrator, X)
    rng = np.random.RandomState(int(cfg.seed))

    picks = {}
    for label, name in ((1, "ransomware"), (0, "benign")):
        want = np.where((train["y_binary"].to_numpy() == label) & (p > 0.9 if label else p < 0.1))[0]
        if len(want) == 0:
            want = np.where(train["y_binary"].to_numpy() == label)[0]
        idx = int(rng.choice(want))
        picks[name] = idx

    idxs = np.array(list(picks.values()))
    sv, base_values, _impl = compute_shap(booster, X[idxs], features)
    draft = []
    for j, (name, idx) in enumerate(picks.items()):
        ev = build_evidence_row(train["row_uid"].iloc[idx], X[idx], sv[j],
                                float(base_values[j]), float(p[idx]), features, stats, glossary)
        draft.append({
            "role": f"fewshot_{name}",
            "row_uid": train["row_uid"].iloc[idx],
            "evidence_table": render_evidence_table(ev),
            "evidence": ev,
        })
        log.info("picked %s example %s p=%.4f", name, train["row_uid"].iloc[idx], p[idx])
    (Paths.prompts / "fewshot_draft.json").write_text(json.dumps(draft, indent=2))
    log.info("wrote %s — now hand-write prompts/fewshot.json", Paths.prompts / "fewshot_draft.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
