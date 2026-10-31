"""Causal faithfulness: do the cited features actually drive the model?

Spec section 8.1 C. Ablations, each with controls:

- Comprehensiveness = q(x) - q(x with cited features set to train medians),
  where q = probability of the model's PREDICTED class, held FIXED for all
  ablated rows (the reference class is the class predicted for x; using each
  ablated row's own argmax instead would switch the sign whenever an ablation
  flips the verdict, hiding the very effect being measured).
- Sufficiency = q(x) - q(background with only cited features restored to x),
  background = train medians.

Controls: n_cited random features (100 draws, average) and the top-n by |SHAP|.
Variants, because median imputation can create off-distribution points:
- marginal resampling: cited values replaced by per-feature draws from the train
  marginals (averaged over draws), with a joint-background sufficiency variant;
- class-conditional ablation: cited features are set to the medians of the
  OPPOSITE class (ransomware vs rest). The plain median background is strongly
  benign-like on this dataset, which biases the plain variant toward benign
  samples; the class-conditional variant is symmetric.
"""
from __future__ import annotations

import numpy as np

from src.models.calibrate import calibrated_proba


def compute_causal(calibrator, X_eval: np.ndarray, sample_ids: list[str],
                   cited: list[list[str]], feature_names: list[str],
                   train_X: np.ndarray, medians: np.ndarray,
                   shap_values: np.ndarray | None = None,
                   train_y: np.ndarray | None = None,
                   n_random: int = 100, n_marginal: int = 20, seed: int = 42) -> dict:
    n, f = X_eval.shape
    idx_of = {name: i for i, name in enumerate(feature_names)}
    rng = np.random.RandomState(seed)

    cited_idx = [[idx_of[name] for name in names if name in idx_of] for names in cited]

    # ---- build all prediction matrices up front (batched) --------------- #
    rows: list[np.ndarray] = []
    slices: dict[str, tuple[int, int]] = {}

    def add(mat: np.ndarray) -> tuple[int, int]:
        start = len(rows)
        rows.extend(mat)
        return start, len(rows)

    slices["x"] = add(X_eval)

    abl_med = X_eval.copy()
    for i, idxs in enumerate(cited_idx):
        if idxs:
            abl_med[i, idxs] = medians[idxs]
    slices["comp_median"] = add(abl_med)

    bg_med = np.tile(medians, (n, 1))
    for i, idxs in enumerate(cited_idx):
        if idxs:
            bg_med[i, idxs] = X_eval[i, idxs]
    slices["suff_median"] = add(bg_med)

    # random-feature control: per sample n_random draws of len(cited) features
    rand_mats = np.zeros((n * n_random, f), dtype=X_eval.dtype)
    for i, idxs in enumerate(cited_idx):
        k = len(idxs)
        base = X_eval[i].copy()
        for r in range(n_random):
            if k == 0:
                rand_mats[i * n_random + r] = base
                continue
            pool = np.setdiff1d(np.arange(f), np.array(idxs), assume_unique=False)
            pick = rng.choice(pool, size=min(k, len(pool)), replace=False)
            row = base.copy()
            row[pick] = medians[pick]
            rand_mats[i * n_random + r] = row
    slices["comp_random"] = add(rand_mats)

    # top-n |SHAP| control (n = number of cited features)
    topn_mat = X_eval.copy()
    if shap_values is not None:
        for i, idxs in enumerate(cited_idx):
            k = len(idxs)
            if k == 0:
                continue
            order = np.argsort(-np.abs(shap_values[i]))[:k]
            topn_mat[i, order] = medians[order]
    slices["comp_topn"] = add(topn_mat)

    # marginal-resampling variant
    marg_comp = np.zeros((n * n_marginal, f), dtype=X_eval.dtype)
    marg_suff = np.zeros((n * n_marginal, f), dtype=X_eval.dtype)
    for i, idxs in enumerate(cited_idx):
        for r in range(n_marginal):
            row = X_eval[i].copy()
            for j in idxs:
                row[j] = train_X[rng.randint(0, len(train_X)), j]  # per-feature draw
            marg_comp[i * n_marginal + r] = row
            bg = train_X[rng.randint(0, len(train_X))].copy()      # joint background row
            if idxs:
                bg[idxs] = X_eval[i, idxs]
            marg_suff[i * n_marginal + r] = bg
    slices["comp_marginal"] = add(marg_comp)
    slices["suff_marginal"] = add(marg_suff)

    all_rows = np.vstack(rows)
    q_all = calibrated_proba(calibrator, all_rows)
    s_x, e_x = slices["x"]
    p_x = q_all[s_x:e_x]
    pred = (p_x >= 0.5).astype(int)
    ref = np.where(pred == 1, p_x, 1.0 - p_x)  # q(x) for the predicted class

    def q_ref(key: str, pred_vec: np.ndarray) -> np.ndarray:
        """Probability of the ORIGINAL predicted class on the ablated rows."""
        s, e = slices[key]
        p = q_all[s:e]
        return np.where(np.asarray(pred_vec) == 1, p, 1.0 - p)

    comp_med = ref - q_ref("comp_median", pred)
    suff_med = ref - q_ref("suff_median", pred)
    comp_rand = ref - q_ref("comp_random", np.repeat(pred, n_random)).reshape(
        n, n_random).mean(axis=1)
    comp_topn = ref - q_ref("comp_topn", pred)
    comp_marg = ref - q_ref("comp_marginal", np.repeat(pred, n_marginal)).reshape(
        n, n_marginal).mean(axis=1)
    suff_marg = ref - q_ref("suff_marginal", np.repeat(pred, n_marginal)).reshape(
        n, n_marginal).mean(axis=1)

    # ---- class-conditional variant (opposite-class medians as background) ---- #
    comp_other = np.full(n, np.nan)
    comp_other_rand = np.full(n, np.nan)
    if train_y is not None:
        y = np.asarray(train_y).astype(int)
        med_by_class = {c: np.median(train_X[y == c], axis=0) for c in (0, 1)}
        other_rows = [X_eval[i].copy() for i in range(n)]
        other_rand = np.zeros((n * n_random, f), dtype=X_eval.dtype)
        for i, idxs in enumerate(cited_idx):
            other = 1 - pred[i]
            bg = med_by_class.get(other, medians)
            if idxs:
                other_rows[i][idxs] = bg[idxs]
            k = len(idxs)
            base = X_eval[i].copy()
            for r in range(n_random):
                if k == 0:
                    other_rand[i * n_random + r] = base
                    continue
                pool = np.setdiff1d(np.arange(f), np.array(idxs), assume_unique=False)
                pick = rng.choice(pool, size=min(k, len(pool)), replace=False)
                row = base.copy()
                row[pick] = bg[pick]
                other_rand[i * n_random + r] = row
        q2 = calibrated_proba(calibrator, np.vstack([np.array(other_rows), other_rand]))
        p_other = q2[:n]
        q_other = np.where(pred == 1, p_other, 1.0 - p_other)
        q_or = q2[n:]
        q_other_rand = np.where(np.repeat(pred, n_random) == 1, q_or, 1.0 - q_or).reshape(
            n, n_random).mean(axis=1)
        comp_other = ref - q_other
        comp_other_rand = ref - q_other_rand

    out = {}
    for i, sid in enumerate(sample_ids):
        k = len(cited_idx[i])
        rec = {
            "n_cited": k,
            "q_pred_class": float(ref[i]),
            "comprehensiveness_median": float(comp_med[i]),
            "sufficiency_median": float(suff_med[i]),
            "comprehensiveness_random_mean": float(comp_rand[i]),
            "causal_lift": float(comp_med[i] - comp_rand[i]),
            "comprehensiveness_topn_shap": float(comp_topn[i]) if shap_values is not None else None,
            "comprehensiveness_marginal": float(comp_marg[i]),
            "comprehensiveness_marginal_random": float(comp_marg[i] - comp_rand[i]),
            "sufficiency_marginal": float(suff_marg[i]),
            "comp_marginal_minus_median": float(comp_marg[i] - comp_med[i]),
        }
        rec["comprehensiveness_otherclass"] = float(comp_other[i]) if np.isfinite(comp_other[i]) else None
        rec["comprehensiveness_otherclass_random"] = (
            float(comp_other_rand[i]) if np.isfinite(comp_other_rand[i]) else None)
        rec["causal_lift_otherclass"] = (
            float(comp_other[i] - comp_other_rand[i]) if np.isfinite(comp_other[i]) else None)
        out[sid] = rec
    return out


CAUSAL_KEYS = ["comprehensiveness_median", "sufficiency_median",
               "comprehensiveness_random_mean", "causal_lift",
               "comprehensiveness_topn_shap", "comprehensiveness_marginal",
               "comprehensiveness_marginal_random", "sufficiency_marginal",
               "comp_marginal_minus_median", "comprehensiveness_otherclass",
               "comprehensiveness_otherclass_random", "causal_lift_otherclass"]


def summarize_causal(results: dict) -> dict:
    vals = [v for v in results.values() if v["n_cited"] > 0]
    if not vals:
        return {"n": 0, "note": "no samples with cited features"}
    out = {"n": len(vals)}
    for k in CAUSAL_KEYS:
        xs = [v[k] for v in vals if v.get(k) is not None]
        out[k + "_mean"] = float(np.mean(xs)) if xs else None
    return out
