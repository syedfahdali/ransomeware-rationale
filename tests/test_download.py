"""Dataset checks (spec phase 1 acceptance). Skipped if the data is absent."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.utils.config import Paths

RAW = Paths.raw / "CIC-MalMem-2022.csv"


@pytest.mark.skipif(not RAW.exists(), reason="raw CSV not downloaded")
def test_raw_file_shape_and_labels():
    n = sum(1 for _ in RAW.open("r", encoding="utf-8", errors="ignore")) - 1
    assert n > 50_000, f"expected >50k rows, got {n}"
    df = pd.read_csv(RAW, nrows=5)
    cols = {c.strip() for c in df.columns}
    assert {"Category", "Class"} & cols, "label columns missing"
    assert len(cols) >= 50, "too few columns"


@pytest.mark.skipif(not (Paths.processed / "all.parquet").exists(),
                    reason="preprocess not run")
def test_processed_labels_and_features():
    df = pd.read_parquet(Paths.processed / "all.parquet")
    feats = json.loads((Paths.processed / "features.json").read_text())
    assert len(feats) == 55
    assert set(df["y_binary"].unique()) == {0, 1}
    assert df["y_binary"].sum() > 5_000, "implausibly few ransomware rows"
    assert not df[feats].isna().any().any(), "NaN features after preprocess"
    assert df["row_uid"].is_unique


@pytest.mark.skipif(not (Paths.processed / "splits.json").exists(), reason="split not run")
def test_split_leakage():
    meta = json.loads((Paths.processed / "splits.json").read_text())
    assert meta["group_aware"] is True
    assert all(v == 0 for v in meta["leak_check"]["group_overlap"].values())
    assert all(v == 0 for v in meta["leak_check"]["identical_rows"].values())
    rows = meta["rows"]
    assert abs(rows["all"] - sum(rows[k] for k in ("train", "val", "test"))) == 0
