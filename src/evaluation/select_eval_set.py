"""Select the N-sample evaluation set from the test split (spec section 3.3).

Strata are (true label, predicted label) = TP / TN / FN / FP. Errors are
deliberately included: faithful rationales for wrong verdicts are the hardest
case. If a stratum is smaller than its quota, take what exists and fill from
the remaining strata; the actual counts are recorded and reported.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.models.calibrate import calibrated_proba, load
from src.utils.checks import record_check
from src.utils.config import Paths, ensure_dirs, load_config, now_iso
from src.utils.logging import setup_logger
from src.utils.seed import set_seed

log = setup_logger("select_eval_set")


def sample_id_from_uid(uid: str) -> str:
    return "test_" + str(uid).replace("row_", "")


def stratum_of(y: int, pred: int) -> str:
    return {(1, 1): "tp", (0, 0): "tn", (1, 0): "fn", (0, 1): "fp"}[(int(y), int(pred))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config()
    set_seed(int(cfg.seed))

    out_path = Paths.eval / "eval_ids.json"
    if out_path.exists() and not args.force:
        log.info("eval set already selected: %s (use --force to redo)", out_path)
        return 0

    test = pd.read_parquet(Paths.processed / "test.parquet")
    booster = load(Paths.models / "clf.joblib")
    calibrator = load(Paths.models / "calibrator.joblib")
    features = json.loads((Paths.processed / "features.json").read_text())
    X = test[features].to_numpy(dtype=np.float32)
    p = calibrated_proba(calibrator, X)
    pred = (p >= 0.5).astype(int)
    y = test["y_binary"].to_numpy()

    rows = []
    for i, (yy, pp, prob) in enumerate(zip(y, pred, p)):
        rows.append({"sample_id": sample_id_from_uid(test["row_uid"].iloc[i]),
                     "row_uid": test["row_uid"].iloc[i], "y": int(yy), "pred": int(pp),
                     "p_ransomware": float(prob),
                     "stratum": stratum_of(yy, pp)})
    preds_df = pd.DataFrame(rows)
    preds_df.to_csv(Paths.outputs / "eval_set_predictions.csv", index=False)
    log.info("test-set stratum sizes:\n%s", preds_df["stratum"].value_counts().to_string())

    n_target = int(cfg.eval.n_samples)
    quotas_cfg = {k: int(v) for k, v in dict(cfg.eval.strata).items()}
    total_q = sum(quotas_cfg.values())
    if total_q != n_target:  # scale quotas to n_target
        scale = n_target / total_q
        quotas = {k: int(round(v * scale)) for k, v in quotas_cfg.items()}
    else:
        quotas = dict(quotas_cfg)

    rng = np.random.RandomState(int(cfg.seed))
    chosen: list[dict] = []
    chosen_ids: set[str] = set()
    notes: list[str] = []
    for stratum in ("tp", "tn", "fn", "fp"):
        pool = preds_df[preds_df["stratum"] == stratum]
        want = quotas[stratum]
        if len(pool) < want:
            notes.append(f"stratum {stratum}: only {len(pool)} samples available, quota {want}")
        take = pool.sample(n=min(want, len(pool)), random_state=int(cfg.seed))
        chosen.extend(take.to_dict("records"))
        chosen_ids.update(take["sample_id"].tolist())

    # fill if short
    if len(chosen) < n_target:
        rest = preds_df[~preds_df["sample_id"].isin(chosen_ids)]
        fill_order = ["tp", "tn", "fp", "fn"]
        need = n_target - len(chosen)
        per = {s: len(rest[rest["stratum"] == s]) for s in fill_order}
        log.info("fill needed=%d from pools %s", need, per)
        for stratum in fill_order:
            if need <= 0:
                break
            pool = rest[rest["stratum"] == stratum]
            take = pool.sample(n=min(need, len(pool)), random_state=int(cfg.seed) + ord(stratum[0]))
            chosen.extend(take.to_dict("records"))
            chosen_ids.update(take["sample_id"].tolist())
            need -= len(take)
            if len(take):
                notes.append(f"filled {len(take)} extra sample(s) from {stratum}")

    chosen.sort(key=lambda r: r["sample_id"])
    actual = pd.Series([c["stratum"] for c in chosen]).value_counts().to_dict()
    payload = {
        "n_samples": len(chosen),
        "n_target": n_target,
        "quotas": quotas,
        "actual_counts": {k: int(actual.get(k, 0)) for k in ("tp", "tn", "fn", "fp")},
        "notes": notes,
        "seed": int(cfg.seed),
        "threshold": 0.5,
        "generated": now_iso(),
        "samples": chosen,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("eval set: n=%d, composition=%s, notes=%s", len(chosen), payload["actual_counts"], notes)
    record_check("phase1/3", "eval set selected",
                 len(chosen) >= 50, f"n={len(chosen)} composition={payload['actual_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
