"""Evidence additivity test (spec phase 3 acceptance).

Skipped until the evidence stage has produced models/ and outputs/shap_eval.npz.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.evidence.shap_builder import additivity_error, raw_margin
from src.utils.config import Paths

NPZ = Paths.outputs / "shap_eval.npz"
CLF = Paths.models / "clf.joblib"


@pytest.mark.skipif(not (NPZ.exists() and CLF.exists()), reason="evidence not built")
def test_additivity_within_tolerance():
    import joblib
    booster = joblib.load(CLF)
    npz = np.load(NPZ, allow_pickle=True)
    sv, base, margins = npz["shap"], npz["base_values"], npz["margins"]
    assert additivity_error(sv, base, margins) <= 1e-3


@pytest.mark.skipif(not (NPZ.exists() and CLF.exists()), reason="evidence not built")
def test_margins_match_model_recomputation():
    import joblib
    import pandas as pd
    booster = joblib.load(CLF)
    npz = np.load(NPZ, allow_pickle=True)
    idx = json.loads((Paths.evidence / "index.json").read_text())
    features = json.loads((Paths.processed / "features.json").read_text())
    test = pd.read_parquet(Paths.processed / "test.parquet").set_index("row_uid")
    uids = [sid.replace("test_", "row_") for sid in idx["sample_ids"]]
    X = test.loc[uids][features].to_numpy(dtype=np.float32)
    recomputed = raw_margin(booster, X)
    assert np.max(np.abs(recomputed - npz["margins"])) <= 1e-3


@pytest.mark.skipif(not (Paths.evidence / "index.json").exists(), reason="evidence not built")
def test_evidence_index_and_files():
    idx = json.loads((Paths.evidence / "index.json").read_text())
    assert idx["n"] >= 50
    assert idx["schema_invalid"] == 0
    for sid in idx["sample_ids"][:5]:
        ev = json.loads((Paths.evidence / f"{sid}.json").read_text())
        assert len(ev["top_features"]) == idx["k"]
        assert ev["verdict"] in ("ransomware", "benign")
        names = [f["name"] for f in ev["top_features"]]
        assert names == sorted(names, key=lambda n: -abs(
            next(f["shap"] for f in ev["top_features"] if f["name"] == n)))
