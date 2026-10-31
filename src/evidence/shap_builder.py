"""SHAP computation (log-odds space) with an exact XGBoost fallback.

`shap.TreeExplainer` is the primary path (spec section 5). If the `shap`
package is unavailable/incompatible, XGBoost's own `pred_contribs` produces the
same exact TreeSHAP values, so the pipeline degrades safely.
"""
from __future__ import annotations

import numpy as np

from src.utils.logging import setup_logger

log = setup_logger("shap_builder")


def compute_shap(booster, X: np.ndarray, feature_names: list[str]):
    """Return (shap_values [n, f], base_values [n], implementation)."""
    X = np.asarray(X)
    try:
        import shap  # type: ignore
        explainer = shap.TreeExplainer(booster)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):  # some versions return per-class lists
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv, dtype=float)
        if sv.ndim == 3:  # (n, f, classes)
            sv = sv[:, :, -1]
        base = np.asarray(explainer.expected_value, dtype=float).ravel()
        base_values = np.full(len(X), float(base[-1] if base.size > 1 else base[0]))
        log.info("SHAP via shap.TreeExplainer (n=%d, f=%d)", *sv.shape)
        return sv, base_values, "shap.TreeExplainer"
    except Exception as e:  # noqa: BLE001
        log.warning("shap package unavailable/failed (%s); using xgboost pred_contribs", e)
        return _fallback_xgb(booster, X, feature_names)


def _fallback_xgb(booster, X, feature_names):
    from xgboost import DMatrix
    if hasattr(booster, "get_booster"):
        impl_booster = booster.get_booster()
    else:
        impl_booster = booster
    dm = DMatrix(np.asarray(X), feature_names=feature_names)
    contribs = impl_booster.predict(dm, pred_contribs=True)
    contribs = np.asarray(contribs)
    sv = contribs[:, :-1]
    base_values = contribs[:, -1]
    log.info("SHAP via xgboost.pred_contribs (n=%d, f=%d)", *sv.shape)
    return sv, base_values, "xgboost.pred_contribs"


def raw_margin(booster, X: np.ndarray) -> np.ndarray:
    """Model output before the sigmoid (log-odds margin)."""
    X = np.asarray(X)
    if hasattr(booster, "predict"):
        try:
            return np.asarray(booster.predict(X, output_margin=True), dtype=float)
        except TypeError:
            pass
    from xgboost import DMatrix
    b = booster.get_booster() if hasattr(booster, "get_booster") else booster
    return np.asarray(b.predict(DMatrix(X), output_margin=True), dtype=float)


def additivity_error(sv: np.ndarray, base_values: np.ndarray, margins: np.ndarray) -> float:
    recon = sv.sum(axis=1) + base_values
    return float(np.max(np.abs(recon - margins)))
